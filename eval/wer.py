"""WER/CER against a reference, overall and around chunk cuts.

    python -m eval.wer <run dir with words.json + chunks.json> --ref recordings/x/ref.txt
    python -m eval.wer <run dir> --ref <other run>/words.txt     # no reference: compare two runs

Scoring part of a recording (a reference for only that part):
    python -m eval.wer <run dir> --from 300 --to 900 --draft recordings/x/ref.txt   # run's text there, to correct
    python -m eval.wer <run dir> --from 300 --to 900 --ref recordings/x/ref.txt
"""
import argparse
import bisect
import json
from collections import Counter
from pathlib import Path

import jiwer
from rapidfuzz.distance import Levenshtein

from eval.textnorm import normalize_tr

def score(reference, hypothesis) -> dict[str, float]:
    ref = normalize_tr(reference)
    hyp = normalize_tr(hypothesis)
    return {"wer": jiwer.wer(ref, hyp), "cer": jiwer.cer(ref, hyp)}


# for turkish, trust CER more than WER (character error rate vs word error rate). One wrong suffix makes the whole word wrong in wer, but its only one character in cer
# örnek: geliyorum vs geliyoruz


def cuts_from_chunks(chunks) -> list[tuple[int, int, str]]:
    """chunks.json rows -> (end of one chunk, start of the next, why it was cut)."""
    return [(c["end_ms"], n["start_ms"], c.get("reason") or "cut") for c, n in zip(chunks, chunks[1:])]


def score_boundaries(reference, words, cuts, window_ms=2000) -> dict[str, dict]:
    """Errors split by where they happen: within window_ms of a cut (zone = the cut's reason:
    "forced", "full", "gap") or "interior". Words get their time from the hypothesis; a deleted
    reference word gets the time between its hypothesis neighbours.
    CER here is per aligned word (spaces not counted), so it is close to, not equal to, jiwer's."""
    ref = normalize_tr(reference).split()
    hyp, hyp_ms = [], []
    for w in words:
        for token in normalize_tr(w["text"]).split():     # "Ali'nin," -> "alinin"; rarely 2 tokens
            hyp.append(token)
            hyp_ms.append((w["start_ms"] + w["end_ms"]) // 2)
    if not ref:
        raise ValueError("empty reference")

    cuts = sorted(cuts)
    starts = [lo - window_ms for lo, _, _ in cuts]

    def zone(t):
        i = bisect.bisect_right(starts, t) - 1
        # cuts are seconds apart, so only the last one starting before t can contain it
        if i >= 0 and t <= cuts[i][1] + window_ms:
            return cuts[i][2]
        return "interior"

    def gap_ms(j):
        """Time between hypothesis words j-1 and j: where a deleted reference word belongs."""
        around = hyp_ms[max(j - 1, 0):j + 1]
        return sum(around) // len(around) if around else 0

    stats = {}
    def add(t, **counts):
        stats.setdefault(zone(t), Counter()).update(counts)

    # hypothesis can be empty (nothing recognized): every reference word is then a deletion
    out = jiwer.process_words(" ".join(ref), " ".join(hyp) if hyp else "")
    for op in out.alignments[0]:
        if op.type in ("equal", "substitute"):
            for i, j in zip(range(op.ref_start_idx, op.ref_end_idx), range(op.hyp_start_idx, op.hyp_end_idx)):
                wrong = op.type == "substitute"
                add(hyp_ms[j], ref_words=1, ref_chars=len(ref[i]), sub=wrong,
                    char_err=Levenshtein.distance(ref[i], hyp[j]) if wrong else 0)
        elif op.type == "delete":
            for i in range(op.ref_start_idx, op.ref_end_idx):
                add(gap_ms(op.hyp_start_idx), ref_words=1, ref_chars=len(ref[i]), dele=1, char_err=len(ref[i]))
        else:   # insert
            for j in range(op.hyp_start_idx, op.hyp_end_idx):
                add(hyp_ms[j], ins=1, char_err=len(hyp[j]))

    result = {}
    for z, c in stats.items():
        errors = c["sub"] + c["dele"] + c["ins"]
        result[z] = {
            "ref_words": c["ref_words"], "sub": c["sub"], "del": c["dele"], "ins": c["ins"],
            "wer": errors / c["ref_words"] if c["ref_words"] else float("nan"),
            "cer": c["char_err"] / c["ref_chars"] if c["ref_chars"] else float("nan"),
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="folder with words.json and chunks.json")
    ap.add_argument("--ref", type=Path, help="reference text (or another run's words.txt)")
    ap.add_argument("--window", type=int, default=2000, help="ms around a cut that count as 'at the cut'")
    ap.add_argument("--from", dest="t_from", type=float, default=0, help="score only words starting from this second")
    ap.add_argument("--to", dest="t_to", type=float, default=float("inf"), help="... up to this second")
    ap.add_argument("--draft", type=Path, help="write the run's text in --from/--to to this file (to correct into a reference)")
    args = ap.parse_args()

    # a word belongs to the range by its start, so a reference cut the same way lines up
    words = [w for w in json.loads((args.run / "words.json").read_text())["words"]
             if args.t_from * 1000 <= w["start_ms"] < args.t_to * 1000]
    chunks = json.loads((args.run / "chunks.json").read_text())["chunks"]

    if args.draft:
        if args.draft.exists() and args.draft.read_text(encoding="utf-8").strip():
            raise SystemExit(f"{args.draft} is not empty: not overwriting a reference")
        args.draft.write_text(" ".join(w["text"] for w in words) + "\n", encoding="utf-8")
        print(f"{len(words)} words -> {args.draft}")
        return
    if not args.ref:
        ap.error("--ref or --draft is required")

    reference = args.ref.read_text(encoding="utf-8")

    total = score(reference, " ".join(w["text"] for w in words))
    print(f"overall: WER {total['wer']:.1%}  CER {total['cer']:.1%}  ({len(chunks)} chunks)")

    zones = score_boundaries(reference, words, cuts_from_chunks(chunks), args.window)
    print(f"\n{'zone':10s} {'ref words':>9s} {'sub':>5s} {'del':>5s} {'ins':>5s} {'WER':>7s} {'CER':>7s}")
    for z in sorted(zones, key=lambda z: (z == "interior", z)):
        r = zones[z]
        print(f"{z:10s} {r['ref_words']:9d} {r['sub']:5d} {r['del']:5d} {r['ins']:5d} {r['wer']:7.1%} {r['cer']:7.1%}")
    if "interior" in zones and zones["interior"]["cer"] > 0:
        for z in zones:
            if z != "interior":
                print(f"{z} / interior CER: {zones[z]['cer'] / zones['interior']['cer']:.2f}x")


if __name__ == "__main__":
    main()
