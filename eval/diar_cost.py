"""Clean cost: embed one recording from scratch, alone on core 0 (as the app's pipeline does).
Nothing cached: every setting embeds again.

    OPENBLAS_NUM_THREADS=1 taskset -c 0 python -m eval.diar_cost recordings/habersizler_168
"""
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import soundfile as sf

from eval.bench_diar import MODELS
from pipeline.config import load_config
from pipeline.diar import SpeakerEmbedder, Tuning, cluster_turns, embed_windows
from pipeline.vad import SpeechRegion

rec = Path(sys.argv[1])
cfg = load_config()
regions = [SpeechRegion(**x) for x in json.loads((rec / "vad.json").read_text())["regions"]]
audio_s = sf.info(str(rec / "audio.wav")).duration
speech_s = sum(r.duration_ms for r in regions) / 1000
print(f"{rec.name}: {audio_s / 60:.1f} min audio, {speech_s / 60:.1f} min speech")
print(f"{'model':9s} {'win':>5s} {'step':>5s} {'windows':>7s} {'embed s':>8s} {'RTF':>6s} {'ahc s':>6s} {'spec s':>6s}")
for model in ("resnet34", "campplus"):
    emb = SpeakerEmbedder(MODELS[model])
    for window, step in ((1000, 750), (1500, 750), (2000, 750), (3000, 750), (1500, 500)):
        tun = Tuning(window, step)
        t0 = time.perf_counter()
        w, e = embed_windows(rec / "audio.wav", regions, emb, tun)
        took = time.perf_counter() - t0
        t0 = time.perf_counter(); cluster_turns(w, e, regions, replace(cfg.diar, method="ahc"), tun); ahc_s = time.perf_counter() - t0
        t0 = time.perf_counter(); cluster_turns(w, e, regions, replace(cfg.diar, method="spectral"), tun); sc_s = time.perf_counter() - t0
        print(f"{model:9s} {window:5d} {step:5d} {len(w):7d} {took:8.1f} {took / audio_s:6.3f} {ahc_s:6.1f} {sc_s:6.1f}", flush=True)
