"""Run the pipeline on an audio file, stage by stage.

    python -m pipeline.run data/test/meeting.m4a              # all stages
    python -m pipeline.run data/test/meeting.m4a --from vad   # rerun from one stage
    python -m pipeline.run data/test/meeting.m4a --to vad     # stop after one stage
    python -m pipeline.run data/test/meeting.m4a --from diar  # only rediarize (e.g. after changing [diar])
    python -m pipeline.run data/test/meeting.m4a --from summary  # only resummarize
    taskset -c 0 python -m pipeline.run data/test/meeting.m4a --parallel   # stt and diar at the same time
"""
import argparse
import json
import io
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import soundfile as sf

from pipeline.config import Config, load_config
from pipeline.diar import Turn, diarize
from pipeline.merge import merge
from pipeline.stt import Chunk, Word, pack, transcribe
from pipeline.summarize import summarize
from pipeline.vad import SpeechRegion, vad_file


# Stages: each reads the previous stage's files and writes its own

def stage_convert(src: Path, sdir: Path, cfg: Config) -> None:
    """Any audio/video file -> 16 kHz mono s16 WAV, the format every model expects."""
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(sdir / "audio.wav"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()[-1000:]}")


def stage_vad(src: Path, sdir: Path, cfg: Config) -> None:
    regions = vad_file(sdir / "audio.wav", cfg.vad)
    write_json(sdir / "vad.json", {"regions": [r.to_dict() for r in regions]})
    write_labels(sdir / "vad.labels.txt", [(r.start_ms, r.end_ms, "speech") for r in regions])

    total = duration_ms(sdir / "audio.wav")
    speech = sum(r.duration_ms for r in regions)
    print(f"  {len(regions)} regions | speech {speech / 1000:.0f} s of {total / 1000:.0f} s "
          f"({speech / total:.0%})")


def stage_pack(src: Path, sdir: Path, cfg: Config) -> None:
    regions = [SpeechRegion(**r) for r in read_json(sdir / "vad.json")["regions"]]
    chunks = pack(regions, cfg.stt, sdir / "audio.wav")
    write_json(sdir / "chunks.json", {"chunks": [c.to_dict() for c in chunks]})
    # chunk labels for Audacity: listen at the cuts, "forced" ones are mid-speech
    write_labels(sdir / "chunks.labels.txt", [(c.start_ms, c.end_ms, f"{c.index} {c.reason}") for c in chunks])

    covered = sum(c.duration_ms for c in chunks)
    fill = covered / (len(chunks) * cfg.stt.max_chunk_ms) if chunks else 0
    reasons = {r: sum(c.reason == r for c in chunks) for r in ("gap", "full", "forced")}
    # fill = how much of each 30 s encoder window is real audio (100% = no waste)
    print(f"  {len(chunks)} chunks | {covered / 1000:.0f} s audio | encoder fill {fill:.0%} | cuts: "
          + " ".join(f"{r} {n}" for r, n in reasons.items()))


def stage_stt(src: Path, sdir: Path, cfg: Config) -> None:
    chunks = [Chunk(**c) for c in read_json(sdir / "chunks.json")["chunks"]]
    words = transcribe(sdir / "audio.wav", chunks, cfg.stt)

    write_json(sdir / "words.json", {"words": [w.to_dict() for w in words]})
    (sdir / "words.txt").write_text(" ".join(w.text for w in words) + "\n", encoding="utf-8")
    # word labels for Audacity: check the word timestamps against the waveform
    write_labels(sdir / "words.labels.txt", [(w.start_ms, w.end_ms, w.text) for w in words])
    print(f"  {len(words)} words")


def stage_diar(src: Path, sdir: Path, cfg: Config) -> None:
    regions = [SpeechRegion(**r) for r in read_json(sdir / "vad.json")["regions"]]
    turns, report = diarize(sdir / "audio.wav", regions, cfg.diar)

    write_json(sdir / "turns.json", {"turns": [t.to_dict() for t in turns]})
    write_labels(sdir / "turns.labels.txt", [(t.start_ms, t.end_ms, t.speaker) for t in turns])
    # RTTM: the standard diarization format, used for DER scoring later
    (sdir / "turns.rttm").write_text("".join(
        f"SPEAKER {sdir.name} 1 {t.start_ms / 1000:.3f} {t.duration_ms / 1000:.3f} "
        f"<NA> <NA> {t.speaker} <NA> <NA>\n" for t in turns
    ), encoding="utf-8")

    for s in sorted({t.speaker for t in turns}):
        mine = [t for t in turns if t.speaker == s]
        print(f"  {s}: {sum(t.duration_ms for t in mine) / 1000:.0f} s in {len(mine)} turns")
    # centroid similarity: close to 1 between two speakers = probably one person split in two
    print(f"  {report['method']} | {report['windows']} windows | {report['speakers']} speakers "
          f"| centroid similarity:")
    for row in report["centroid_similarity"]:
        print("    " + " ".join(f"{v:5.2f}" for v in row))


def stage_merge(src: Path, sdir: Path, cfg: Config) -> None:
    words = [Word(**w) for w in read_json(sdir / "words.json")["words"]]
    turns = [Turn(**t) for t in read_json(sdir / "turns.json")["turns"]]
    utterances = merge(words, turns)

    write_json(sdir / "transcript.json", {"utterances": [u.to_dict() for u in utterances]})
    lines = [f"[{clock(u.start_ms)}] {u.speaker}: {u.text}" for u in utterances]
    (sdir / "transcript.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  {len(utterances)} utterances")


def stage_summary(src: Path, sdir: Path, cfg: Config) -> None:
    lines = (sdir / "transcript.txt").read_text(encoding="utf-8").splitlines()
    # only what was said counts, not the "[0:01:23] SPK_0:" prefixes
    n_words = sum(len(u["text"].split()) for u in read_json(sdir / "transcript.json")["utterances"])

    summary = summarize(lines, n_words, cfg.summary, cfg.stt.language)
    (sdir / "summary.md").write_text(summary + "\n", encoding="utf-8")
    print(f"  {cfg.summary.backend} | {len(summary.split())} words -> summary.md")


STAGES = {
    "convert": stage_convert,
    "vad": stage_vad,
    "pack": stage_pack,
    "stt": stage_stt,
    "diar": stage_diar,
    "merge": stage_merge,
    "summary": stage_summary,
}


# File helpers

def write_json(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    # ensure_ascii=False keeps Turkish characters readable in the file ("ş", not "ş")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)  # atomic: a crash never leaves a half-written file


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_labels(path: Path, rows) -> None:
    """Audacity labels (File → Import → Labels) to check results against the waveform."""
    path.write_text("".join(f"{s / 1000:.3f}\t{e / 1000:.3f}\t{label}\n" for s, e, label in rows),
                    encoding="utf-8")


def duration_ms(path: Path) -> int:
    return int(sf.info(str(path)).duration * 1000)


def clock(ms: int) -> str:
    s = ms // 1000
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"


# Runner

def main() -> None:
    names = list(STAGES)
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--from", dest="first", choices=names, default=names[0])
    ap.add_argument("--to", dest="last", choices=names, default=names[-1])
    ap.add_argument("--language", help="override [stt] language: auto, en, tr, ...")
    ap.add_argument("--parallel", action="store_true", help="run stt and diar at the same time")
    args = ap.parse_args()

    cfg = load_config()
    if args.language:
        cfg = replace(cfg, stt=replace(cfg.stt, language=args.language))
    sdir = cfg.data_dir / args.audio.stem
    sdir.mkdir(parents=True, exist_ok=True)
    print(f"session: {sdir}")

    selected = names[names.index(args.first): names.index(args.last) + 1]
    together = args.parallel and {"stt", "diar"} <= set(selected)
    for name in selected:
        if together and name == "stt":
            run_stt_and_diar(args, sdir, cfg)
        elif not (together and name == "diar"):
            run_stage(name, args.audio, sdir, cfg)


def run_stage(name: str, src: Path, sdir: Path, cfg: Config) -> None:
    print(f"> {name}", flush=True)
    t0 = time.perf_counter()
    STAGES[name](src, sdir, cfg)
    took = time.perf_counter() - t0
    wav = sdir / "audio.wav"
    rtf = f" | RTF {took / (duration_ms(wav) / 1000):.3f}" if wav.exists() else ""
    print(f"  {name}: {took:.1f} s{rtf}", flush=True)


def run_stt_and_diar(args, sdir: Path, cfg: Config) -> None:
    """stt mostly waits on whisper-server (cores 1-3) while diar computes on this process's core.
    They only share vad.json, chunks.json and audio.wav. diar runs as its own process, so the two
    outputs don't mix line by line: its output is printed in one piece after stt.

    For main.py's progress screen: "+ diar" says diar starts next to the stage that follows,
    "= diar" (printed as soon as diar ends, in the middle of stt's lines) that it is done."""
    cmd = [sys.executable, "-m", "pipeline.run", str(args.audio), "--from", "diar", "--to", "diar"]
    diar = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = []

    def wait_diar():
        out.append(diar.communicate()[0])
        print(f"= diar done (exit code {diar.returncode})", flush=True)

    stdout = sys.stdout
    sys.stdout = WholeLines(stdout)     # the "= diar" line must not land inside a line of stt's
    watcher = threading.Thread(target=wait_diar, daemon=True)
    try:
        print("+ diar next to stt", flush=True)
        watcher.start()
        try:
            run_stage("stt", args.audio, sdir, cfg)
        except BaseException:
            diar.kill()
            raise

        t0 = time.perf_counter()
        watcher.join()
    finally:
        sys.stdout = stdout
    print(out[0], end="")
    print(f"  stt + diar in parallel: waited {time.perf_counter() - t0:.1f} s for diar after stt", flush=True)
    if diar.returncode != 0:
        raise RuntimeError(f"diar failed with exit code {diar.returncode}")


class WholeLines(io.TextIOBase):
    """stdout shared by threads: each thread's text is held until its line is complete, then the
    whole line is written at once. print() writes the text and the newline separately, so without
    this another thread's line could land between them."""

    def __init__(self, out):
        self.out = out
        self.lock = threading.Lock()
        self.local = threading.local()

    def write(self, s):
        *lines, self.local.rest = (getattr(self.local, "rest", "") + s).split("\n")
        if lines:
            with self.lock:
                self.out.write("".join(line + "\n" for line in lines))
                self.out.flush()
        return len(s)

    def flush(self):
        with self.lock:
            self.out.flush()


if __name__ == "__main__":
    main()
