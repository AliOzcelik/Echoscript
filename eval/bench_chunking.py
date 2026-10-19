"""Chunking strategies on the same recording: transcribe, then score overall and at the cuts.

    A           pack(): <= 30 s, cut at silences (the pipeline today). Read from runs/A, not rerun.
    A_noprompt  the same chunks, no previous text as prompt: what the prompt is worth
    B           fixed 30 s windows, cut anywhere: the worst case
    C           30 s windows every 25 s (5 s overlap), each keeps its words up to the middle of the overlap
    E           live: LiveChunker on the simulated clock (eval.live_sim), min_send 0, forced cuts at 30 s
    E2          E, forced cuts at the quietest point of the last 5 s
    E3          E, forced cuts at 30 s, the next chunk starts 3-6 s earlier (quietest point there)
    E2_noprompt E2 without the previous text as prompt
    A2          pack() with the wav: speech without a pause cut at quiet points, like E2
    A2_noprompt A2 without the prompt

Only the chunks touching --from/--to are transcribed (plus the one before, so the prompt is warm),
so a 10-minute section takes ~10 minutes per strategy. Scored against ref.txt if it has text,
otherwise against runs/A (agreement, not accuracy: shows where a strategy differs from today's).

    python -m eval.bench_chunking recordings/oscar_da_bizi_gorecek_mi --from 510 --to 1110
    python -m eval.bench_chunking recordings/oscar_da_bizi_gorecek_mi --from 510 --to 1110 --only B C
"""
import argparse
import csv
import json
import time
from pathlib import Path

import httpx
import soundfile as sf

from eval.live_sim import simulate
from eval.wer import cuts_from_chunks, score, score_boundaries
from pipeline.config import SAMPLE_RATE, load_config
from pipeline.stt import Chunk, Word, inference, pack, parse, tail
from pipeline.vad import SpeechRegion

SAMPLES_PER_MS = SAMPLE_RATE // 1000
STRATEGIES = ["A", "A_noprompt", "B", "C", "E", "E2", "E3", "E2_noprompt", "A2", "A2_noprompt"]
LIVE_CUT = {"E": "hard", "E2": "quiet", "E3": "overlap", "E2_noprompt": "quiet"}
WINDOW_MS = 30000
OVERLAP_MS = 5000


def has_speech(start, end, regions):
    return any(r.start_ms < end and r.end_ms > start for r in regions)


def make_chunks(name, regions, total_ms, cfg, wav=None):
    """-> chunks to transcribe, and per chunk the time range whose words it keeps."""
    if name in ("A", "A_noprompt"):
        chunks = pack(regions, cfg.stt)
    elif name in ("A2", "A2_noprompt"):
        chunks = pack(regions, cfg.stt, wav)
    elif name in LIVE_CUT:
        chunks = [c for c, _, _ in simulate(regions, total_ms, cfg, 0, LIVE_CUT[name], wav)]
    else:
        step = WINDOW_MS - (OVERLAP_MS if name == "C" else 0)
        starts = [s for s in range(0, total_ms, step) if has_speech(s, s + WINDOW_MS, regions)]
        chunks = [Chunk(i, s, min(s + WINDOW_MS, total_ms), "fixed" if name == "B" else "overlap")
                  for i, s in enumerate(starts)]
    # overlapping chunks: C cuts in the middle of the overlap; E3 at the next chunk's start,
    # because the first chunk's last seconds are the ones whisper drops
    def border(a, b):
        return (b.start_ms + a.end_ms) // 2 if name == "C" else b.start_ms
    keep = []
    for i, c in enumerate(chunks):
        lo = border(chunks[i - 1], c) if i and chunks[i - 1].end_ms > c.start_ms else c.start_ms
        hi = border(c, chunks[i + 1]) if i + 1 < len(chunks) and chunks[i + 1].start_ms < c.end_ms else c.end_ms
        keep.append((lo, hi))
    return chunks, keep


def transcribe(wav, chunks, keep, cfg, use_prompt):
    """stt.transcribe(), but a chunk only keeps words inside its keep range, and prompt can be off."""
    words, prompt, took = [], "", 0.0
    with httpx.Client(base_url=cfg.server_url, timeout=cfg.timeout_s) as client:
        for n, (c, (lo, hi)) in enumerate(zip(chunks, keep)):
            audio, _ = sf.read(wav, start=c.start_ms * SAMPLES_PER_MS, stop=c.end_ms * SAMPLES_PER_MS, dtype="float32")
            t0 = time.perf_counter()
            data = inference(client, audio, prompt if use_prompt else "", cfg)
            took += time.perf_counter() - t0
            chunk_ms = len(audio) * 1000 // SAMPLE_RATE
            new = [Word(w.start_ms + c.start_ms, w.end_ms + c.start_ms, w.text, w.prob)
                   for w in parse(data, cfg.word_time, chunk_ms)]
            new = [w for w in new if lo <= (w.start_ms + w.end_ms) // 2 < hi]
            words += new
            prompt = tail(f"{prompt} {' '.join(w.text for w in new)}", cfg.prompt_chars)
            print(f"    chunk {n + 1}/{len(chunks)} | {c.reason:7s} | {chunk_ms / 1000:4.1f} s | {len(new):3d} words", flush=True)
    return words, took


def in_range(words, t_from, t_to):
    return [w for w in words if t_from <= w["start_ms"] < t_to]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rec", type=Path, help="recording folder: audio.wav, vad.json, ref.txt, runs/A")
    ap.add_argument("--from", dest="t_from", type=float, required=True, help="section start, s")
    ap.add_argument("--to", dest="t_to", type=float, required=True, help="section end, s")
    ap.add_argument("--only", nargs="+", choices=STRATEGIES, default=STRATEGIES)
    ap.add_argument("--rerun", action="store_true", help="transcribe again even if the run is there")
    args = ap.parse_args()
    cfg = load_config()
    t_from, t_to = int(args.t_from * 1000), int(args.t_to * 1000)

    regions = [SpeechRegion(**r) for r in json.loads((args.rec / "vad.json").read_text())["regions"]]
    total_ms = int(sf.info(str(args.rec / "audio.wav")).duration * 1000)
    meta = {"from_ms": t_from, "to_ms": t_to}

    for name in args.only:
        out = args.rec / "runs" / name
        if name == "A":
            if not (out / "words.json").exists():
                raise SystemExit(f"{out}/words.json missing: run the pipeline's pack + stt on the recording first")
            continue
        old = json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else None
        if old and old["from_ms"] == t_from and old["to_ms"] == t_to and not args.rerun:
            print(f"{name}: already there, scoring only")
            continue

        chunks, keep = make_chunks(name, regions, total_ms, cfg, args.rec / "audio.wav")
        touch = [i for i, c in enumerate(chunks) if c.end_ms > t_from and c.start_ms < t_to]
        sel = list(range(max(touch[0] - 1, 0), touch[-1] + 1))      # one chunk before: warm prompt
        print(f"{name}: {len(sel)} of {len(chunks)} chunks", flush=True)
        words, took = transcribe(args.rec / "audio.wav", [chunks[i] for i in sel], [keep[i] for i in sel],
                                 cfg.stt, use_prompt=not name.endswith("_noprompt"))

        out.mkdir(parents=True, exist_ok=True)
        # chunks.json holds the kept ranges: for C these are the effective cuts, not the windows
        rows = [{**chunks[i].to_dict(), "start_ms": keep[i][0], "end_ms": keep[i][1],
                 "audio_start_ms": chunks[i].start_ms, "audio_end_ms": chunks[i].end_ms} for i in sel]
        (out / "chunks.json").write_text(json.dumps({"chunks": rows}, indent=1), encoding="utf-8")
        (out / "words.json").write_text(json.dumps({"words": [w.to_dict() for w in words]}, indent=1, ensure_ascii=False), encoding="utf-8")
        (out / "words.txt").write_text(" ".join(w.text for w in words) + "\n", encoding="utf-8")
        for f, labels in [("chunks.labels.txt", [(r["start_ms"], r["end_ms"], f"{r['index']} {r['reason']}") for r in rows]),
                          ("words.labels.txt", [(w.start_ms, w.end_ms, w.text) for w in words])]:
            (out / f).write_text("".join(f"{s / 1000:.3f}\t{e / 1000:.3f}\t{t}\n" for s, e, t in labels), encoding="utf-8")
        audio_s = sum(chunks[i].duration_ms for i in sel) / 1000
        (out / "meta.json").write_text(json.dumps({**meta, "chunks": len(sel), "audio_s": audio_s,
                                                   "took_s": round(took, 1)}, indent=1), encoding="utf-8")

    # scoring: the section only, against the reference or else runs/A
    ref_file = args.rec / "ref.txt"
    has_ref = ref_file.exists() and ref_file.read_text(encoding="utf-8").strip()
    if has_ref:
        reference, against = ref_file.read_text(encoding="utf-8"), "ref.txt"
    else:
        a_words = json.loads((args.rec / "runs/A/words.json").read_text())["words"]
        reference, against = " ".join(w["text"] for w in in_range(a_words, t_from, t_to)), "runs/A (agreement)"
    print(f"\n{args.rec.name}  {args.t_from:g}-{args.t_to:g} s, scored against {against}")
    print(f"{'strategy':11s} {'chunks':>6s} {'RTF':>5s} {'WER':>6s} {'CER':>6s} {'CER cut':>8s} {'CER in':>7s} {'cut/in':>6s}  cut zones")

    rows = []
    for name in args.only:
        out = args.rec / "runs" / name
        if not (out / "words.json").exists():
            continue
        words = in_range(json.loads((out / "words.json").read_text())["words"], t_from, t_to)
        chunks = json.loads((out / "chunks.json").read_text())["chunks"]
        total = score(reference, " ".join(w["text"] for w in words))
        zones = score_boundaries(reference, words, cuts_from_chunks(chunks))
        # all cut zones together: errors weighted by their words
        cut = [z for k, z in zones.items() if k != "interior"]
        cut_cer = (sum(z["cer"] * z["ref_words"] for z in cut) / sum(z["ref_words"] for z in cut)) if cut else float("nan")
        in_cer = zones.get("interior", {}).get("cer", float("nan"))
        meta_f = out / "meta.json"
        rtf = (lambda m: m["took_s"] / m["audio_s"])(json.loads(meta_f.read_text())) if meta_f.exists() else float("nan")
        per_zone = " ".join(f"{k} {z['cer']:.1%}" for k, z in sorted(zones.items()) if k != "interior")
        ratio = cut_cer / in_cer if in_cer else float("nan")
        n_chunks = sum(c["end_ms"] > t_from and c["start_ms"] < t_to for c in chunks)
        print(f"{name:11s} {n_chunks:6d} {rtf:5.2f} {total['wer']:6.1%} {total['cer']:6.1%} {cut_cer:8.1%} {in_cer:7.1%} {ratio:6.2f}  {per_zone}")
        rows.append({"recording": args.rec.name, "from_s": args.t_from, "to_s": args.t_to, "against": against,
                     "strategy": name, "chunks": n_chunks, "rtf": round(rtf, 3), "wer": round(total["wer"], 4),
                     "cer": round(total["cer"], 4), "cer_cut": round(cut_cer, 4), "cer_interior": round(in_cer, 4),
                     "cut_over_interior": round(ratio, 2), **{f"cer_{k}": round(z["cer"], 4) for k, z in zones.items()}})

    csv_path = args.rec / "runs" / "bench_chunking.csv"
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"-> {csv_path}")


if __name__ == "__main__":
    main()
