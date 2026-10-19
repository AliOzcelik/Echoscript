"""Cost of one whisper-server request by chunk length: how short can live chunks be?

The encoder works on a 30 s window whatever the chunk length, so a short chunk costs nearly as
much as a full one. Live mode keeps up only while each chunk takes less time than its own audio.

Run from the project root (whisper-server must be running):
    python -m eval.bench_cost recordings/htalks_gs_barcelona
    python -m eval.bench_cost recordings/htalks_gs_barcelona --lengths 5 10 20 30 --repeat 3
    python -m eval.bench_cost recordings/htalks_gs_barcelona --audio-ctx   # also with a shorter encoder window
"""
import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import httpx
import soundfile as sf

from pipeline.config import SAMPLE_RATE, load_config
from eval.wer import score
from pipeline.stt import inference, parse

FRAMES_PER_S = 50       # encoder frames: 1500 for 30 s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rec", type=Path, help="folder with audio.wav and vad.json")
    ap.add_argument("--lengths", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30], help="chunk lengths, s")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--audio-ctx", action="store_true", help="also run each length with audio_ctx = its length")
    args = ap.parse_args()
    cfg = load_config().stt

    # start inside speech, so every chunk has words to decode like a real one
    regions = json.loads((args.rec / "vad.json").read_text())["regions"]
    start = regions[len(regions) // 3]["start_ms"] * SAMPLE_RATE // 1000

    rows = []
    with httpx.Client(base_url=cfg.server_url, timeout=cfg.timeout_s) as client:
        warm, _ = sf.read(args.rec / "audio.wav", start=start, frames=5 * SAMPLE_RATE, dtype="float32")
        inference(client, warm, "", cfg)        # the first request after a server start is slower

        for length in args.lengths:
            audio, _ = sf.read(args.rec / "audio.wav", start=start, frames=length * SAMPLE_RATE, dtype="float32")
            full_text = None
            for ctx in ([0, length * FRAMES_PER_S + 64] if args.audio_ctx and length < 30 else [0]):
                took = []
                for _ in range(args.repeat):
                    t0 = time.perf_counter()
                    data = inference(client, audio, "", cfg, audio_ctx=ctx)
                    took.append(time.perf_counter() - t0)
                words = parse(data, cfg.word_time, length * 1000)
                text = " ".join(w.text for w in words)
                full_text = full_text or text
                # how far the shorter window's text is from the full window's (0 = same text)
                diff = score(full_text, text)["cer"] if ctx else 0.0
                mean = statistics.mean(took)
                rows.append({"length_s": length, "audio_ctx": ctx, "took_s": round(mean, 2), "min_s": round(min(took), 2),
                             "max_s": round(max(took), 2), "rtf": round(mean / length, 2), "words": len(words),
                             "cer_vs_full": round(diff, 3)})
                print(f"{length:3d} s chunk | ctx {ctx or 'full':>4} | {mean:5.1f} s ({min(took):.1f}-{max(took):.1f}) | "
                      f"RTF {mean / length:.2f} | {len(words)} words | CER vs full {diff:.1%}", flush=True)

    out = Path("data/bench/cost.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for ctx_used in sorted({bool(r["audio_ctx"]) for r in rows}):
        ok = [r["length_s"] for r in rows if r["rtf"] < 0.8 and bool(r["audio_ctx"]) == ctx_used]
        label = "short audio_ctx" if ctx_used else "full window"
        print(f"\n{label}: shortest chunk with RTF < 0.8: {min(ok)} s" if ok else f"\n{label}: no length reaches RTF < 0.8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
