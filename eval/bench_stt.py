"""STT benchmark: thread scaling, quantization, decoding, timestamps.

Run from the project root:
    python -m eval.bench_stt --audio data/test/clip16k.wav --ref data/test/clip16k.ref.txt
    python -m eval.bench_stt --audio ... --ref ... --only A_q5_t4
"""
import argparse
import csv
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf

from eval.wer import score

WHISPER_CLI = Path.home() / "src/whisper.cpp/build/bin/whisper-cli"
F16 = Path.home() / "src/whisper.cpp/models/ggml-large-v3-turbo.bin"
Q8 = Path("models/ggml-large-v3-turbo-q8_0.bin")
Q5 = Path("models/ggml-large-v3-turbo-q5_0.bin")
OUT_DIR = Path("data/bench")
N_CORES = 4


@dataclass
class Run:
    name: str
    model: Path
    threads: int
    args: list[str] = field(default_factory=list)


RUNS = [
    # A: thread scaling (q5_0, greedy) -> decides the core split
    Run("A_q5_t1", Q5, 1, ["-bs", "1"]),
    Run("A_q5_t2", Q5, 2, ["-bs", "1"]),
    Run("A_q5_t3", Q5, 3, ["-bs", "1"]),
    Run("A_q5_t4", Q5, 4, ["-bs", "1"]),
    # B: quantization at the production thread count (q5_t3 is A_q5_t3)
    Run("B_f16_t3", F16, 3, ["-bs", "1"]),
    Run("B_q8_t3", Q8, 3, ["-bs", "1"]),
    # C: beam search vs greedy
    Run("C_q5_t3_bs5", Q5, 3, ["-bs", "5"]),
    Run("C_q8_t3_bs5", Q8, 3, ["-bs", "5"]),
    # D: DTW word timestamps (needs flash attention off; check flag names with --help)
    Run("D_q5_t3_dtw", Q5, 3, ["-bs", "1", "-nfa", "--dtw", "large.v3.turbo"]),
]

TIMING_RE = re.compile(r"whisper_print_timings:\s+(\w+) time =\s+([\d.]+) ms")


def soc_temp_c() -> float:
    return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000


def throttled() -> str:
    out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True).stdout
    return out.strip().split("=")[-1]


def cooldown(limit_c: float) -> None:
    # Every run starts from the same temperature, so runs are comparable.
    while (t := soc_temp_c()) > limit_c:
        print(f"  cooling down: {t:.1f} °C")
        time.sleep(10)


def pinned_cores(threads: int) -> str:
    # Use the highest cores, like production (STT on 1-3, core 0 stays free).
    return f"{N_CORES - threads}-{N_CORES - 1}"


def run_whisper(cmd: list[str]) -> tuple[float, int, str]:
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    stderr = proc.stderr.read()  # read fully before waiting, avoids a pipe deadlock
    _, status, rusage = os.wait4(proc.pid, 0)  # wait4 gives this child's own peak memory
    wall_s = time.perf_counter() - t0
    proc.returncode = os.waitstatus_to_exitcode(status)
    if proc.returncode != 0:
        raise RuntimeError(f"whisper-cli failed ({proc.returncode}):\n{stderr[-2000:]}")
    return wall_s, rusage.ru_maxrss // 1024, stderr


def read_text(json_path: Path) -> str:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return "".join(seg["text"] for seg in data["transcription"])


def bench(run: Run, audio: Path, reference: str, audio_s: float) -> dict:
    out_base = OUT_DIR / run.name
    cmd = [
        "taskset", "-c", pinned_cores(run.threads),
        str(WHISPER_CLI),
        "-m", str(run.model),
        "-f", str(audio),
        "-t", str(run.threads),
        "-l", "tr",
        "-ojf", "-of", str(out_base),
        *run.args,
    ]
    wall_s, rss_mb, stderr = run_whisper(cmd)
    (OUT_DIR / f"{run.name}.log").write_text(stderr, encoding="utf-8")

    t = {name: float(ms) / 1000 for name, ms in TIMING_RE.findall(stderr)}
    load_s = t.get("load", 0.0)
    # whisper-server keeps the model loaded, so the production cost is total - load.
    compute_s = t.get("total", wall_s) - load_s

    hyp = read_text(out_base.with_suffix(".json"))
    return {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "name": run.name,
        "model": run.model.name,
        "threads": run.threads,
        "args": " ".join(run.args),
        "audio_s": round(audio_s, 1),
        "wall_s": round(wall_s, 2),
        "load_s": round(load_s, 2),
        "encode_s": round(t.get("encode", 0.0), 2),
        "decode_s": round(t.get("decode", 0.0) + t.get("batchd", 0.0), 2),
        "compute_s": round(compute_s, 2),
        "rtf": round(compute_s / audio_s, 3),
        "rss_mb": rss_mb,
        **{k: round(v, 4) for k, v in score(reference, hyp).items()},
        "temp_c": soc_temp_c(),
        "throttled": throttled(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--only", nargs="*", help="run only these run names")
    ap.add_argument("--cool", type=float, default=60.0, help="start each run below this °C")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reference = args.ref.read_text(encoding="utf-8")
    audio_s = sf.info(str(args.audio)).duration
    runs = [r for r in RUNS if not args.only or r.name in args.only]
    if not runs:
        raise SystemExit("no runs selected")

    print(f"audio {audio_s:.1f} s | {len(runs)} runs | throttled at start: {throttled()}")
    rows = []
    for run in runs:
        cooldown(args.cool)
        print(f"> {run.name}")
        row = bench(run, args.audio, reference, audio_s)
        print(f"  RTF {row['rtf']:.2f} | CER {row['cer']:.3f} | WER {row['wer']:.3f} "
              f"| RSS {row['rss_mb']} MB | {row['temp_c']:.0f} °C | {row['throttled']}")
        rows.append(row)

    out_csv = OUT_DIR / "results.csv"
    new_file = not out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
    print(f"saved -> {out_csv}")


if __name__ == "__main__":
    main()
