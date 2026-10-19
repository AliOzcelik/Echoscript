"""Pairwise disagreement between diarization settings on whole recordings, from cached voiceprints.
Disagreement = DER of one against the other (collar 0, speakers mapped one-to-one): the share of speech
the two give to different people. Same VAD, so it is all confusion.

    python -m eval.diar_agree "resnet34 1500 750 window spectral 0.25" "campplus 1500 750 window ahc 0.8" ...

Each setting: model window step cmn method (threshold for ahc | p_ratio for spectral). Needs the
voiceprints bench_diar cached in recordings/*/runs/diar/.
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

from eval.bench_diar import emb_key, load_embs
from eval.der import der
from pipeline.config import load_config
from pipeline.diar import Tuning, cluster_turns
from pipeline.vad import SpeechRegion

RECS = ["htalks_gs_barcelona", "oscar_da_bizi_gorecek_mi", "balkan_asidik", "habersizler_168"]
cfg = load_config()


def run(rec, spec):
    model, window, step, cmn, method, knob = spec.split()
    window, step = int(window), int(step)
    r = Path("recordings") / rec
    regions = [SpeechRegion(**x) for x in json.loads((r / "vad.json").read_text())["regions"]]
    w, e, _ = load_embs(r / "runs" / "diar" / f"{emb_key(model, window, step, cmn, 'system')}.npz")
    if method == "ahc":
        d, t = replace(cfg.diar, method="ahc", threshold=float(knob), num_speakers=0), Tuning(window, step, cmn)
    else:
        d, t = replace(cfg.diar, method="spectral", num_speakers=0), Tuning(window, step, cmn, sc_max_p_ratio=float(knob))
    turns, rep = cluster_turns(w, e, regions, d, t)
    return [(x.start_ms / 1000, x.end_ms / 1000, x.speaker) for x in turns], rep["speakers"]


def main() -> None:
    specs = sys.argv[1:]
    names = [chr(ord("A") + i) for i in range(len(specs))]
    for n, s in zip(names, specs):
        print(f"{n} = {s}")
    for rec in RECS:
        res = [run(rec, s) for s in specs]
        print(f"\n{rec}: speakers " + " ".join(f"{n}={k}" for n, (_, k) in zip(names, res)))
        print("     " + "".join(f"{n:>7s}" for n in names))
        for i, (a, _) in enumerate(res):
            print(f"  {names[i]}  " + "".join(f"{der(a, b, 0)['der']:7.1%}" if i != j else f"{'':>7s}" for j, (b, _) in enumerate(res)))


if __name__ == "__main__":
    main()
