"""Diarization settings on the same recordings: voiceprints computed once per embedding setting,
then every clustering setting on them, scored against hand-made speaker labels (ref.labels.txt).

Embedding settings (each one is a pass over the whole recording, minutes; cached in runs/diar/):
    --model     resnet34 (WeSpeaker, 256-d, the pipeline today) | campplus (3D-Speaker CAM++, 512-d)
    --window    ms of audio per voiceprint            --step   ms between windows
    --cmn       mean removal: window | region | none  --vad    system (vad.json) | oracle (see below)
Clustering settings (each one takes seconds):
    --method ahc | spectral    --threshold (ahc)    --linkage average | complete | ward (ahc)
    --p-ratio (spectral)       --min-cluster        --min-turn    --merge-gap
Every list is crossed with every other. The speaker count is never given: num_speakers stays 0.

The whole recording is diarized, as in the app, and only its section is scored (the STT sections
of DOCUMENTATION.md, so the E2 words there give WDER for free).
    oracle VAD: inside the section the speech regions are the reference turns, outside it vad.json.
                DER minus oracle DER = what VAD costs; oracle confusion = the clustering alone.

    python -m eval.bench_diar recordings/htalks_gs_barcelona --draft     # ref.labels.txt to fill in
    python -m eval.bench_diar recordings/*/                               # the pipeline's settings
    python -m eval.bench_diar recordings/*/ --model resnet34 campplus --threshold 0.5 0.6 0.7 0.8
    python -m eval.bench_diar recordings/*/ --method ahc spectral --vad system oracle --jobs 3
"""
import os

# --jobs runs several workers at once: each must use one BLAS thread, as the app's pipeline does
# under taskset -c 0. With the default (all cores each) spectral took 220 s instead of 10 s.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import csv
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

from eval.der import clip, der, read_turns, wder, word_speakers
from pipeline.config import load_config
from pipeline.diar import SpeakerEmbedder, Tuning, Window, cluster_turns, embed_windows
from pipeline.vad import SpeechRegion

# the STT sections (s); --from/--to for any other recording
SECTIONS = {
    "htalks_gs_barcelona": (60, 660),
    "oscar_da_bizi_gorecek_mi": (510, 1110),
    "balkan_asidik": (90, 690),
}
MODELS = {
    "resnet34": Path("models/wespeaker_en_voxceleb_resnet34.onnx"),
    "campplus": Path("models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"),
}
REF = "ref.labels.txt"
WORDS = "runs/E2/words.json"
CSV_OUT = Path("data/bench/diar.csv")
MAJOR_SHARE = 0.05      # a speaker with at least this share of the speech is a "major" one


# Reference

def draft(rec: Path, t_from: float, t_to: float) -> None:
    """VAD regions in the section, every one labelled "?": rename each to its speaker by ear,
    split it where the speaker changes, add a label for a voice VAD missed. Not the system's
    turns: correcting those would favour the settings that made them."""
    out = rec / REF
    if out.exists() and out.read_text(encoding="utf-8").strip():
        raise SystemExit(f"{out} is not empty: not overwriting a reference")
    regions = clip([(r["start_ms"] / 1000, r["end_ms"] / 1000, "?") for r in read_regions(rec)], t_from, t_to)
    out.write_text("".join(f"{s:.3f}\t{e:.3f}\t?\n" for s, e, _ in regions), encoding="utf-8")
    print(f"{len(regions)} regions -> {out}")


def load_ref(rec: Path, t_from: float, t_to: float):
    """Reference turns in the section, or None if there is no reference yet."""
    path = rec / REF
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return None
    turns = clip(read_turns(path), t_from, t_to)
    unnamed = [s for s, _, spk in turns if spk in ("", "?")]
    if len(unnamed) == len(turns):
        return None     # a draft nobody has started on
    if unnamed:
        raise SystemExit(f"{path}: {len(unnamed)} labels still unnamed, first at {unnamed[0]:.1f} s")
    return turns


def read_regions(rec: Path):
    return json.loads((rec / "vad.json").read_text())["regions"]


def oracle_regions(system, ref_turns, t_from: float, t_to: float):
    """vad.json outside the section; inside it, the reference turns (overlaps joined)."""
    lo, hi = round(t_from * 1000), round(t_to * 1000)
    out = []
    for r in system:
        if r.start_ms < lo:
            out.append(SpeechRegion(r.start_ms, min(r.end_ms, lo)))
        if r.end_ms > hi:
            out.append(SpeechRegion(max(r.start_ms, hi), r.end_ms))
    inside = []
    for s, e, _ in sorted(ref_turns):
        s, e = round(s * 1000), round(e * 1000)
        if inside and s <= inside[-1][1]:
            inside[-1][1] = max(inside[-1][1], e)
        else:
            inside.append([s, e])
    return sorted(out + [SpeechRegion(s, e) for s, e in inside], key=lambda r: r.start_ms)


# Voiceprints, cached

def emb_key(model, window, step, cmn, vad):
    return f"{model}_w{window}_s{step}_{cmn}_{vad}"


def embed_job(args):
    """One embedding setting on one recording -> runs/diar/<key>.npz. Runs in a worker process."""
    rec, model, window, step, cmn, vad, regions = args
    path = rec / "runs" / "diar" / f"{emb_key(model, window, step, cmn, vad)}.npz"
    r_arr = np.array([(r.start_ms, r.end_ms) for r in regions], dtype=np.int64).reshape(-1, 2)
    if path.exists():
        with np.load(path) as z:
            if np.array_equal(z["regions"], r_arr):     # oracle regions change when the reference does
                return path, None

    embedder = SpeakerEmbedder(MODELS.get(model, Path(model)))
    t0 = time.perf_counter()
    windows, embs = embed_windows(rec / "audio.wav", regions, embedder, Tuning(window, step, cmn))
    took = time.perf_counter() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, regions=r_arr, embs=embs,
             windows=np.array([(w.start_ms, w.end_ms, w.region) for w in windows], dtype=np.int64).reshape(-1, 3),
             took_s=took)
    return path, took


def load_embs(path: Path):
    with np.load(path) as z:
        windows = [Window(int(s), int(e), int(r)) for s, e, r in z["windows"]]
        return windows, z["embs"], float(z["took_s"])


# Runner

def cluster_job(work):
    """All clustering settings on one voiceprint set -> rows. Runs in a worker process."""
    (rec, model, window, step, cmn, vad, regions), settings, diar_cfg, ref, (t_from, t_to), labels = work
    windows, embs, took = load_embs(rec / "runs" / "diar" / f"{emb_key(model, window, step, cmn, vad)}.npz")
    audio_s = sf.info(str(rec / "audio.wav")).duration
    rows = []
    for s in settings:
        dcfg = replace(diar_cfg, method=s["method"], num_speakers=0,
                       threshold=s["threshold"] if s["method"] == "ahc" else diar_cfg.threshold,
                       min_cluster_ms=s["min_cluster"])
        tuning = Tuning(window, step, cmn, s["linkage"] if s["method"] == "ahc" else Tuning.linkage,
                        s["p_ratio"] if s["method"] == "spectral" else Tuning.sc_max_p_ratio,
                        s["merge_gap"], s["min_turn"])
        t0 = time.perf_counter()
        turns, report = cluster_turns(windows, embs, regions, dcfg, tuning)
        cluster_s = time.perf_counter() - t0

        row = {"recording": rec.name, "model": model, "window": window, "step": step, "cmn": cmn, "vad": vad, **s,
               "speakers": report["speakers"], **plausibility(turns, report, audio_s), "windows": len(windows),
               "embed_rtf": took / audio_s, "cluster_s": cluster_s, **score(rec, t_from, t_to, ref, turns)}
        rows.append(row)
        if labels:
            name = "_".join(str(row[k]) for k in SETTING_KEYS)
            out = rec / "runs" / "diar" / "turns" / f"{name}.labels.txt"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("".join(f"{t.start_ms / 1000:.3f}\t{t.end_ms / 1000:.3f}\t{t.speaker}\n" for t in turns),
                           encoding="utf-8")
    return rows


def plausibility(turns, report, audio_s):
    """Without a reference: does the result look like a conversation? A "major" speaker has at
    least MAJOR_SHARE of the speech; many small ones = a voice split into pieces. The highest
    similarity between two major speakers' centroids near 1 = probably one person twice."""
    per = {}
    for t in turns:
        per[t.speaker] = per.get(t.speaker, 0) + t.duration_ms
    total = sum(per.values()) or 1
    major = [s for s, ms in per.items() if ms / total >= MAJOR_SHARE]
    sim = np.array(report["centroid_similarity"])
    idx = [int(s.split("_")[1]) for s in major]
    pair = max((sim[i, j] for i in idx for j in idx if i < j), default=float("nan"))
    return {"major_speakers": len(major), "minor_share": 1 - sum(per[s] for s in major) / total,
            "top_share": max(per.values()) / total, "turns_per_min": len(turns) / (audio_s / 60),
            "max_major_sim": float(pair)}

def cluster_settings(args):
    """All clustering settings; the knobs a method ignores are '-', so it runs once."""
    out = []
    for method, thr, link, p, mc, mt, mg in itertools.product(
            args.method, args.threshold, args.linkage, args.p_ratio, args.min_cluster, args.min_turn, args.merge_gap):
        s = {"method": method, "threshold": thr if method == "ahc" else "-", "linkage": link if method == "ahc" else "-",
             "p_ratio": p if method == "spectral" else "-", "min_cluster": mc, "min_turn": mt, "merge_gap": mg}
        if s not in out:
            out.append(s)
    return out


def score(rec, t_from, t_to, ref, turns):
    hyp = [(t.start_ms / 1000, t.end_ms / 1000, t.speaker) for t in turns]
    if ref is None:
        return {}
    d = der(ref, clip(hyp, t_from, t_to), 0.25)
    d0 = der(ref, clip(hyp, t_from, t_to), 0.0)
    row = {"ref_speakers": d["ref_speakers"], "hyp_speakers_section": d["hyp_speakers"],
           "der": d["der"], "miss": d["miss"], "false_alarm": d["false_alarm"], "confusion": d["confusion"],
           "der_collar0": d0["der"], "ref_overlap": d["ref_overlap"]}
    if (rec / WORDS).exists():
        w = wder(ref, *word_speakers(rec / WORDS, hyp, t_from, t_to))
        row.update(wder=w["wder"], words=w["words"])
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recs", nargs="+", type=Path, help="recording folders: audio.wav, vad.json, ref.labels.txt")
    ap.add_argument("--from", dest="t_from", type=float, help="section start, s (default: SECTIONS)")
    ap.add_argument("--to", dest="t_to", type=float, help="section end, s")
    ap.add_argument("--draft", action="store_true", help="write ref.labels.txt from the VAD regions, then stop")
    ap.add_argument("--model", nargs="+", default=["resnet34"], help="resnet34 | campplus | path to an .onnx")
    ap.add_argument("--window", nargs="+", type=int, default=[Tuning.window_ms])
    ap.add_argument("--step", nargs="+", type=int, default=[Tuning.step_ms])
    ap.add_argument("--cmn", nargs="+", default=[Tuning.cmn], choices=["window", "region", "none"])
    ap.add_argument("--vad", nargs="+", default=["system"], choices=["system", "oracle"])
    ap.add_argument("--method", nargs="+", default=None, choices=["ahc", "spectral"])
    ap.add_argument("--threshold", nargs="+", type=float, default=None)
    ap.add_argument("--linkage", nargs="+", default=[Tuning.linkage], choices=["average", "complete", "ward"])
    ap.add_argument("--p-ratio", nargs="+", type=float, default=[Tuning.sc_max_p_ratio])
    ap.add_argument("--min-cluster", nargs="+", type=int, default=None)
    ap.add_argument("--min-turn", nargs="+", type=int, default=[Tuning.min_turn_ms])
    ap.add_argument("--merge-gap", nargs="+", type=int, default=[Tuning.merge_gap_ms])
    ap.add_argument("--jobs", type=int, default=1, help="embedding passes at once (1 thread each). "
                    ">1 is faster, but the RTF then includes the passes slowing each other")
    ap.add_argument("--top", type=int, default=20, help="settings printed, best first")
    ap.add_argument("--sort", default="der", choices=["der", "wder", "confusion"])
    ap.add_argument("--labels", action="store_true", help="also write each run's turns as Audacity labels")
    ap.add_argument("--out", type=Path, default=CSV_OUT)
    args = ap.parse_args()

    cfg = load_config()
    args.method = args.method or [cfg.diar.method]
    args.threshold = args.threshold or [cfg.diar.threshold]
    args.min_cluster = args.min_cluster or [cfg.diar.min_cluster_ms]
    recs = [r for r in args.recs if (r / "audio.wav").exists()]

    def section(rec):
        if args.t_from is not None and args.t_to is not None:
            return args.t_from, args.t_to
        return SECTIONS.get(rec.name, (0, float("inf")))     # no section: all of it

    if args.draft:
        for rec in recs:
            draft(rec, *section(rec))
        return

    refs = {rec: load_ref(rec, *section(rec)) for rec in recs}
    for rec, ref in refs.items():
        if ref is None:
            print(f"{rec.name}: no {REF} yet: speakers found and cost only")
    if "oracle" in args.vad:
        missing = [rec.name for rec, ref in refs.items() if ref is None]
        if missing:
            raise SystemExit(f"--vad oracle needs a reference: {', '.join(missing)}")

    # 1. voiceprints: one pass per recording x embedding setting, skipped if cached
    jobs = []
    for rec in recs:
        system = [SpeechRegion(**r) for r in read_regions(rec)]
        for model, window, step, cmn, vad in itertools.product(args.model, args.window, args.step, args.cmn, args.vad):
            regions = system if vad == "system" else oracle_regions(system, refs[rec], *section(rec))
            jobs.append((rec, model, window, step, cmn, vad, regions))
    print(f"{len(jobs)} embedding passes", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for job, (path, took) in zip(jobs, pool.map(embed_job, jobs)):
            print(f"  {job[0].name} {path.stem}: " + ("cached" if took is None else f"{took:.0f} s"), flush=True)

    # 2. every clustering setting on every voiceprint set
    settings = cluster_settings(args)
    print(f"{len(settings)} clustering settings", flush=True)
    rows = []
    work = [(job, settings, cfg.diar, refs[job[0]], section(job[0]), args.labels) for job in jobs]
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for job, out in zip(jobs, pool.map(cluster_job, work)):
            rows += out
            print(f"  {job[0].name} {emb_key(*job[1:6])}: {len(out)} clusterings", flush=True)

    report_rows(rows, recs, args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in rows)
    print(f"-> {args.out} ({len(rows)} rows)")


SETTING_KEYS = ["model", "window", "step", "cmn", "vad", "method", "threshold", "linkage", "p_ratio",
                "min_cluster", "min_turn", "merge_gap"]


def report_rows(rows, recs, args) -> None:
    """One line per setting: the mean over the recordings, then each recording's DER / speakers."""
    by_setting = {}
    for r in rows:
        by_setting.setdefault(tuple(r[k] for k in SETTING_KEYS), []).append(r)
    varying = [k for i, k in enumerate(SETTING_KEYS) if len({key[i] for key in by_setting}) > 1] or ["model"]
    scored = all("der" in r for r in rows)

    def mean(rs, k):
        vals = [r[k] for r in rs if k in r]
        return sum(vals) / len(vals) if vals else float("nan")

    order = sorted(by_setting.items(), key=lambda kv: mean(kv[1], args.sort)) if scored else list(by_setting.items())
    head = " ".join(f"{k:>11s}" for k in varying)
    names = [rec.name[:14] for rec in recs]
    if scored:
        print(f"\n{head} {'DER':>6s} {'conf':>6s} {'WDER':>6s} {'RTF':>5s}  " + "  ".join(f"{n:>21s}" for n in names))
        print(f"{'':{len(head)}s} {'':6s} {'':6s} {'':6s} {'':5s}  " + "  ".join(f"{'DER  WDER  spk/ref':>21s}" for _ in names))
    else:
        print(f"\n{head} {'RTF':>5s}  " + "  ".join(f"{n:>14s}" for n in names))
    for key, rs in order[:args.top]:
        cells = " ".join(f"{str(key[SETTING_KEYS.index(k)]):>11s}" for k in varying)
        per = {r["recording"]: r for r in rs}
        if scored:
            recs_txt = "  ".join(
                f"{per[rec.name]['der']:6.1%} {per[rec.name].get('wder', float('nan')):5.1%} "
                f"{per[rec.name]['hyp_speakers_section']:3d}/{per[rec.name]['ref_speakers']:<3d}" if rec.name in per else f"{'':21s}"
                for rec in recs)
            print(f"{cells} {mean(rs, 'der'):6.1%} {mean(rs, 'confusion'):6.1%} {mean(rs, 'wder'):6.1%} "
                  f"{mean(rs, 'embed_rtf'):5.3f}  {recs_txt}")
        else:
            recs_txt = "  ".join(f"{per[rec.name]['speakers']:>10d} spk" if rec.name in per else f"{'':14s}" for rec in recs)
            print(f"{cells} {mean(rs, 'embed_rtf'):5.3f}  {recs_txt}")


if __name__ == "__main__":
    main()
