"""Reference draft for a section: a stronger, independent transcription plus a list of the places
where the systems under test disagree with it, so only those need listening.

    R   whisper-cli, f16 model (not quantized), beam 5, whisper's own long-form decoding (it seeks
        by its timestamps, so it shares none of our chunk cuts) -> runs/R/words.json
Then for every run in runs/ (A, E, E3, ...): align to R, and list the time spans where any run
differs from R. R's text goes to ref.txt (if empty); check.txt lists the spans to listen to with
R's reading and each run's, check.labels.txt the same spans for Audacity.

Correct ref.txt at those spots by ear. Where every system agrees with R it is taken as right:
errors all whisper models share stay in (we measure cut losses, which do not).

    python -m eval.reference recordings/oscar_da_bizi_gorecek_mi --from 510 --to 1110
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import jiwer

from eval.textnorm import normalize_tr

WHISPER_CLI = Path.home() / "src/whisper.cpp/build/bin/whisper-cli"
F16 = Path.home() / "src/whisper.cpp/models/ggml-large-v3-turbo.bin"
MARGIN_MS = 3000        # decode a little around the section, so its first and last words are whole
JOIN_TOKENS = 2         # disagreements this close together are one spot to listen to


def transcribe_reference(rec: Path, t_from: int, t_to: int, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    start = max(t_from - MARGIN_MS, 0)
    cmd = [str(WHISPER_CLI), "-m", str(F16), "-f", str(rec / "audio.wav"), "-l", "tr", "-t", "4",
           "-bs", "5", "-ot", str(start), "-d", str(t_to + MARGIN_MS - start),
           "-ml", "1", "-sow", "-oj", "-of", str(out / "cli"), "-np"]
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    took = time.perf_counter() - t0
    segs = json.loads((out / "cli.json").read_text())["transcription"]
    words = [{"start_ms": s["offsets"]["from"], "end_ms": s["offsets"]["to"], "text": s["text"].strip(), "prob": 0.0}
             for s in segs if s["text"].strip()]
    (out / "words.json").write_text(json.dumps({"words": words}, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps({"from_ms": t_from, "to_ms": t_to, "took_s": round(took, 1),
                                               "model": F16.name, "beam": 5}, indent=1), encoding="utf-8")
    print(f"R: {len(words)} words in {took / 60:.1f} min")


def tokens(words):
    """Normalized tokens, each with the index of the word it came from."""
    out = []
    for i, w in enumerate(words):
        out += [(t, i) for t in normalize_tr(w["text"]).split()]
    return out


def compare(ref_toks, hyp_toks):
    """-> per reference token: the run's tokens aligned to it, and whether they differ."""
    got = [[] for _ in ref_toks]
    bad = [False] * len(ref_toks)
    out = jiwer.process_words(" ".join(t for t, _ in ref_toks), " ".join(t for t, _ in hyp_toks) or "")
    for op in out.alignments[0]:
        if op.type in ("equal", "substitute"):
            for k in range(op.ref_end_idx - op.ref_start_idx):
                got[op.ref_start_idx + k].append(hyp_toks[op.hyp_start_idx + k][0])
                bad[op.ref_start_idx + k] |= op.type == "substitute"
        elif op.type == "delete":
            for i in range(op.ref_start_idx, op.ref_end_idx):
                bad[i] = True
        else:   # insert: belongs to the reference token before it
            i = max(op.ref_start_idx - 1, 0)
            got[i] += [hyp_toks[j][0] for j in range(op.hyp_start_idx, op.hyp_end_idx)]
            bad[i] = True
    return got, bad


def clock(ms):
    return f"{ms // 60000:02d}:{ms // 1000 % 60:02d}.{ms % 1000 // 100}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rec", type=Path)
    ap.add_argument("--from", dest="t_from", type=float, required=True)
    ap.add_argument("--to", dest="t_to", type=float, required=True)
    ap.add_argument("--runs", nargs="+", default=["A", "A_noprompt", "E", "E2", "E3"], help="runs to compare with R")
    args = ap.parse_args()
    t_from, t_to = int(args.t_from * 1000), int(args.t_to * 1000)
    runs = args.rec / "runs"

    if not (runs / "R/words.json").exists():
        transcribe_reference(args.rec, t_from, t_to, runs / "R")
    in_range = lambda ws: [w for w in ws if t_from <= w["start_ms"] < t_to]
    ref_words = in_range(json.loads((runs / "R/words.json").read_text())["words"])
    ref_toks = tokens(ref_words)

    ref_file = args.rec / "ref.txt"
    if not (ref_file.exists() and ref_file.read_text(encoding="utf-8").strip()):
        ref_file.write_text(" ".join(w["text"] for w in ref_words) + "\n", encoding="utf-8")
        print(f"R's text -> {ref_file} (correct it at the spots in check.txt)")

    systems = {}
    for d in (runs / name for name in args.runs):
        if (d / "words.json").exists():
            systems[d.name] = compare(ref_toks, tokens(in_range(json.loads((d / "words.json").read_text())["words"])))

    # spots: runs of reference tokens where at least one system differs, nearby ones joined
    bad = [any(b[i] for _, b in systems.values()) for i in range(len(ref_toks))]
    spots, i = [], 0
    while i < len(bad):
        if not bad[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(bad) and any(bad[j + 1:j + 2 + JOIN_TOKENS]):
            j += 1
        spots.append((i, j))
        i = j + 1

    lines, labels = [], []
    for n, (i, j) in enumerate(spots, 1):
        w0, w1 = ref_toks[i][1], ref_toks[j][1]
        lo, hi = ref_words[w0]["start_ms"], ref_words[w1]["end_ms"]
        before = " ".join(w["text"] for w in ref_words[max(w0 - 4, 0):w0])
        after = " ".join(w["text"] for w in ref_words[w1 + 1:w1 + 5])
        lines.append(f"#{n}  {clock(lo)} - {clock(hi)}")
        lines.append(f"    ...{before} [{' '.join(w['text'] for w in ref_words[w0:w1 + 1])}] {after}...")
        for name, (got, b) in systems.items():
            if any(b[i:j + 1]):
                lines.append(f"    {name:11s} {' '.join(t for k in range(i, j + 1) for t in got[k]) or '(nothing)'}")
        lines.append("")
        labels.append(f"{lo / 1000:.3f}\t{hi / 1000:.3f}\t#{n}\n")

    (args.rec / "check.txt").write_text(
        f"{len(spots)} spots in {clock(t_from)} - {clock(t_to)}: [R's words]; below, what the runs heard instead\n\n"
        + "\n".join(lines), encoding="utf-8")
    (args.rec / "check.labels.txt").write_text("".join(labels), encoding="utf-8")
    print(f"{len(spots)} spots to check ({sum(bad)} of {len(ref_toks)} reference tokens) -> {args.rec / 'check.txt'}")


if __name__ == "__main__":
    main()
