"""DER (diarization error rate): how much of the reference speech gets the wrong speaker, or none.
WDER (word DER): how many words of the transcript get the wrong speaker, after merge.

Run from the project root:
    python -m eval.der --ref data/test/meeting.ref.rttm --hyp data/sessions/meeting/turns.rttm
    python -m eval.der --ref data/test/meeting.ref.txt --hyp data/sessions/meeting/turns.rttm --collar 0
    python -m eval.der --ref recordings/x/ref.labels.txt --hyp <turns.rttm> --from 510 --to 1110 \
        --words recordings/x/runs/E2/words.json        # a reference for one section only, + WDER

DER = (missed speech + false alarm + speaker confusion) / reference speech, counted per 10 ms frame
after the best one-to-one mapping between reference and hypothesis speakers, like NIST md-eval.
Overlapping speech counts once per speaker.

WDER: every word gets its speaker as in the transcript (merge(), with the speaker change snapped
to a pause), and its reference speaker from the reference turns it overlaps. A word in overlapping
speech is right if it got any of the speakers talking. Words outside every reference turn are not
scored (VAD/word timing, not diarization). Speakers are mapped one-to-one by word counts.

Making a reference in Audacity: select each speaker turn, add a label (Ctrl+B) named after the
speaker, then File → Export → Export Labels. Overlapping speech: one label per speaker, overlapping
in time. RTTM files work too.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

FRAME_S = 0.01


def read_turns(path: Path) -> list[tuple[float, float, str]]:
    """RTTM (SPEAKER <file> 1 <start> <duration> <NA> <NA> <speaker> ...) or Audacity labels
    (start <tab> end <tab> speaker) -> (start_s, end_s, speaker)."""
    turns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if path.suffix == ".rttm":
            f = line.split()
            if f and f[0] == "SPEAKER":
                turns.append((float(f[3]), float(f[3]) + float(f[4]), f[7]))
        elif line.strip() and not line.startswith("\\"):    # "\" lines: Audacity's spectral selections
            start, end, speaker = line.split("\t")[:3]
            turns.append((float(start), float(end), speaker.strip()))
    return turns


def clip(turns, t_from: float, t_to: float):
    """Only the part of each turn inside [t_from, t_to] seconds."""
    out = []
    for start, end, speaker in turns:
        start, end = max(start, t_from), min(end, t_to)
        if end > start:
            out.append((start, end, speaker))
    return out


def activity(turns, speakers, n_frames: int) -> np.ndarray:
    """(speakers, frames) bool: who is speaking in each 10 ms frame."""
    a = np.zeros((len(speakers), n_frames), dtype=bool)
    for start, end, speaker in turns:
        a[speakers.index(speaker), round(start / FRAME_S):round(end / FRAME_S)] = True
    return a


def scored(ref_turns, n_frames: int, collar_s: float) -> np.ndarray:
    """False within ±collar of every reference boundary: boundaries placed by hand are never exact."""
    keep = np.ones(n_frames, dtype=bool)
    c = round(collar_s / FRAME_S)
    for start, end, _ in ref_turns:
        for b in (round(start / FRAME_S), round(end / FRAME_S)):
            keep[max(0, b - c):b + c] = False
    return keep


def der(ref_turns, hyp_turns, collar_s: float = 0.25) -> dict:
    n_frames = round(max(end for _, end, _ in ref_turns + hyp_turns) / FRAME_S) + 1
    ref_spk = sorted({s for *_, s in ref_turns})
    hyp_spk = sorted({s for *_, s in hyp_turns})
    keep = scored(ref_turns, n_frames, collar_s)
    ref = activity(ref_turns, ref_spk, n_frames)[:, keep]
    hyp = activity(hyp_turns, hyp_spk, n_frames)[:, keep]

    # best one-to-one mapping: the speaker pairs that talk at the same time the most (Hungarian algorithm)
    overlap = ref.astype(np.float64) @ hyp.T.astype(np.float64)    # (ref, hyp) frames both active
    r_idx, h_idx = linear_sum_assignment(overlap, maximize=True)
    mapped = np.zeros_like(ref)         # hypothesis activity renamed to the reference speakers
    mapped[r_idx] = hyp[h_idx]

    n_ref = ref.sum(axis=0)
    n_hyp = hyp.sum(axis=0)
    overlap_share = (n_ref >= 2).sum() / max((n_ref >= 1).sum(), 1)    # no system here can score this
    total = n_ref.sum()
    if total == 0:
        raise ValueError("the reference has no scored speech")
    miss = np.maximum(n_ref - n_hyp, 0).sum()               # fewer speakers found than talking
    false_alarm = np.maximum(n_hyp - n_ref, 0).sum()        # more speakers found than talking
    confusion = (np.minimum(n_ref, n_hyp) - (ref & mapped).sum(axis=0)).sum()  # found, but the wrong one

    return {
        "der": float((miss + false_alarm + confusion) / total),
        "miss": float(miss / total),
        "false_alarm": float(false_alarm / total),
        "confusion": float(confusion / total),
        "ref_speech_s": float(total * FRAME_S),
        "ref_speakers": len(ref_spk),
        "hyp_speakers": len(hyp_spk),
        "ref_overlap": float(overlap_share),
        "mapping": {hyp_spk[h]: ref_spk[r] for r, h in zip(r_idx, h_idx) if overlap[r, h] > 0},
    }


def wder(ref_turns, words, speakers) -> dict:
    """words: (start_s, end_s) per word; speakers: the speaker merge gave each word."""
    ref_spk = sorted({s for *_, s in ref_turns})
    hyp_spk = sorted(set(speakers))
    scored = []     # (reference speakers of the word, the main one, hypothesis speaker)
    for (w0, w1), h in zip(words, speakers):
        talk = {}
        for start, end, s in ref_turns:
            ov = min(w1, end) - max(w0, start)
            if ov > 0 or start <= (w0 + w1) / 2 < end:      # zero-length word: its time point
                talk[s] = talk.get(s, 0) + max(ov, 0)
        if talk:
            scored.append((set(talk), max(talk, key=talk.get), h))
    if not scored:
        raise ValueError("no word falls inside a reference turn")

    counts = np.zeros((len(ref_spk), len(hyp_spk)))
    for _, main, h in scored:
        counts[ref_spk.index(main), hyp_spk.index(h)] += 1
    r_idx, h_idx = linear_sum_assignment(counts, maximize=True)
    to_ref = {hyp_spk[h]: ref_spk[r] for r, h in zip(r_idx, h_idx)}

    wrong = sum(to_ref.get(h) not in talk for talk, _, h in scored)
    return {"wder": wrong / len(scored), "words": len(scored), "unscored_words": len(words) - len(scored)}


def word_speakers(words_json: Path, hyp_turns, t_from: float, t_to: float):
    """The transcript's words starting in [t_from, t_to) with the speaker merge gives them.
    merge() runs over all words, so snapping at the section edges is as in the pipeline."""
    from pipeline.diar import Turn
    from pipeline.merge import merge
    from pipeline.stt import Word

    words = [Word(**w) for w in json.loads(words_json.read_text(encoding="utf-8"))["words"]]
    turns = sorted((Turn(round(s * 1000), round(e * 1000), spk) for s, e, spk in hyp_turns),
                   key=lambda t: t.start_ms)
    pairs = [(w, u.speaker) for u in merge(words, turns) for w in u.words]
    pairs = [(w, s) for w, s in pairs if t_from * 1000 <= w.start_ms < t_to * 1000]
    return [(w.start_ms / 1000, w.end_ms / 1000) for w, _ in pairs], [s for _, s in pairs]


def main() -> None:
    ap = argparse.ArgumentParser(description="Diarization error rate of a hypothesis against a reference.")
    ap.add_argument("--ref", type=Path, required=True, help="reference: .rttm or Audacity labels (.txt)")
    ap.add_argument("--hyp", type=Path, required=True, help="hypothesis, e.g. data/sessions/<name>/turns.rttm")
    ap.add_argument("--collar", type=float, default=0.25, help="seconds ignored around reference boundaries")
    ap.add_argument("--from", dest="t_from", type=float, default=0, help="score only from this second")
    ap.add_argument("--to", dest="t_to", type=float, default=float("inf"), help="... up to this second")
    ap.add_argument("--words", type=Path, help="words.json of a transcription: adds WDER")
    args = ap.parse_args()

    ref_turns = clip(read_turns(args.ref), args.t_from, args.t_to)
    hyp_all = read_turns(args.hyp)
    r = der(ref_turns, clip(hyp_all, args.t_from, args.t_to), args.collar)
    print(f"reference: {r['ref_speakers']} speakers, {r['ref_speech_s']:.1f} s scored speech, "
          f"{r['ref_overlap']:.1%} overlapped | hypothesis: {r['hyp_speakers']} speakers | collar {args.collar} s")
    print(f"DER {r['der']:.1%} = missed {r['miss']:.1%} + false alarm {r['false_alarm']:.1%} "
          f"+ confusion {r['confusion']:.1%}")
    print("mapping: " + ", ".join(f"{h} -> {ref}" for h, ref in r["mapping"].items()))
    if args.words:
        w = wder(ref_turns, *word_speakers(args.words, hyp_all, args.t_from, args.t_to))
        print(f"WDER {w['wder']:.1%} of {w['words']} words ({w['unscored_words']} outside the reference turns)")


if __name__ == "__main__":
    main()
