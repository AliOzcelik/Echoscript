"""bench_diar CSVs -> the markdown tables of DOCUMENTATION.md §15.1–15.3 (no reference labels: plausibility only).

    python -m eval.diar_report                       # data/bench/diar_grid_*.csv
    python -m eval.diar_report data/bench/diar.csv
"""
import csv
import glob
import sys
from collections import defaultdict

files = sys.argv[1:] or sorted(glob.glob("data/bench/diar_grid_*.csv"))
rows = [r for f in files for r in csv.DictReader(open(f))]
for r in rows:
    for k in ("speakers", "major_speakers", "window", "step", "min_cluster"):
        r[k] = int(r[k])
    for k in ("minor_share", "top_share", "turns_per_min", "embed_rtf", "cluster_s"):
        r[k] = float(r[k])
    r["max_major_sim"] = float(r["max_major_sim"]) if r["max_major_sim"] not in ("", "nan") else float("nan")

VIDEOS = ["htalks_gs_barcelona", "oscar_da_bizi_gorecek_mi", "balkan_asidik"]
HAB = "habersizler_168"
RECS = VIDEOS + [HAB]
SHORT = {"htalks_gs_barcelona": "htalks", "oscar_da_bizi_gorecek_mi": "oscar", "balkan_asidik": "balkan", HAB: "habersizler"}

EMB = ("model", "window", "step", "cmn")
SET = EMB + ("method", "threshold", "linkage", "p_ratio", "min_cluster")
by = defaultdict(dict)       # setting -> recording -> row
for r in rows:
    by[tuple(r[k] for k in SET)][r["recording"]] = r

embs = sorted({tuple(r[k] for k in EMB) for r in rows}, key=lambda e: (e[0], e[3] != "window", e[2], e[1]))
thresholds = sorted({r["threshold"] for r in rows if r["method"] == "ahc"}, key=float)
p_ratios = sorted({r["p_ratio"] for r in rows if r["method"] == "spectral"}, key=float)


def emb_name(e):
    m, w, s, c = e
    return f"{m} {w / 1000:g}/{s / 1000:g}s cmn-{c}"


def cell(r):
    """speakers found (major speakers, if fewer)"""
    if r is None:
        return "–"
    return f"{r['speakers']}" + (f" ({r['major_speakers']})" if r["major_speakers"] != r["speakers"] else "")


def get(e, method, thr="-", link="-", p="-", mc=5000):
    return by.get(e + (method, thr, link, p, mc), {})


def ok3(rs):
    return all(v in rs and rs[v]["speakers"] == 3 for v in VIDEOS)


def table(head, lines):
    print("| " + " | ".join(head) + " |")
    print("|" + "|".join("---" for _ in head) + "|")
    for l in lines:
        print("| " + " | ".join(str(x) for x in l) + " |")
    print()


base = [e for e in embs if e[1] == 1500 and e[2] == 750 and e[3] == "window"]

print("## 1. AHC average linkage: speakers found vs threshold (1.5/0.75 s, min_cluster 5 s)\n")
print("Cells: speakers found (major ≥5% speech, if fewer). ✓ = all three videos 3.\n")
for e in base:
    print(f"**{e[0]}**\n")
    lines = []
    for t in thresholds:
        rs = get(e, "ahc", t, "average")
        if not rs:
            continue
        lines.append([t] + [cell(rs.get(v)) for v in RECS] + ["✓" if ok3(rs) else ""])
    table(["threshold"] + [SHORT[v] for v in RECS] + ["3/3/3"], lines)

print("## 2. Spectral (NME-SC): speakers found vs p_ratio (1.5/0.75 s, min_cluster 5 s)\n")
lines = []
for e in base:
    for p in p_ratios:
        rs = get(e, "spectral", p=p)
        lines.append([e[0], p] + [cell(rs.get(v)) for v in RECS] + ["✓" if ok3(rs) else ""])
table(["model", "p_ratio"] + [SHORT[v] for v in RECS] + ["3/3/3"], lines)

print("## 3. Every embedding setting\n")
print("AHC range = average-linkage thresholds where all three videos get exactly 3 (min_cluster 5 s); "
      "habersizler at the middle of that range. Spectral at p_ratio 0.25. "
      "Sim = highest centroid similarity between two major speakers (spectral, mean of the videos; model-specific scale).\n")
lines = []
for e in embs:
    good = [t for t in thresholds if ok3(get(e, "ahc", t, "average"))]
    mid = good[len(good) // 2] if good else None
    hab = cell(get(e, "ahc", mid, "average").get(HAB)) if mid else "–"
    sp = get(e, "spectral", p="0.25")
    sims = [sp[v]["max_major_sim"] for v in VIDEOS if v in sp]
    rtf = [r["embed_rtf"] for r in sp.values()]
    lines.append([emb_name(e), f"{good[0]}–{good[-1]} ({len(good)})" if good else "none", hab,
                  " / ".join(cell(sp.get(v)) for v in RECS), f"{sum(sims) / len(sims):.2f}" if sims else "–",
                  f"{sum(rtf) / len(rtf):.3f}" if rtf else "–"])
table(["embedding", "AHC 3/3/3 range (n)", "habersizler (AHC mid)", "spectral htalks/oscar/balkan/haber", "sim", "embed RTF*"], lines)

print("## 4. Complete linkage: speakers found vs threshold (1.5/0.75 s, min_cluster 5 s)\n")
lines = []
for e in base:
    for t in thresholds:
        rs = get(e, "ahc", t, "complete")
        if rs and float(t) >= 0.9:
            lines.append([e[0], t] + [cell(rs.get(v)) for v in RECS] + ["✓" if ok3(rs) else ""])
table(["model", "threshold"] + [SHORT[v] for v in RECS] + ["3/3/3"], lines)

print("## 5. min_cluster: width of the AHC-average 3/3/3 range, and spectral\n")
lines = []
for e in embs:
    line = [emb_name(e)]
    for mc in (0, 5000, 30000):
        good = [t for t in thresholds if ok3(get(e, "ahc", t, "average", mc=mc))]
        line.append(f"{good[0]}–{good[-1]} ({len(good)})" if good else "none")
    for mc in (0, 5000, 30000):
        line.append(" / ".join(cell(get(e, "spectral", p="0.25", mc=mc).get(v)) for v in RECS))
    lines.append(line)
table(["embedding", "AHC mc 0", "AHC mc 5 s", "AHC mc 30 s", "spectral mc 0", "spectral mc 5 s", "spectral mc 30 s"], lines)

print("## 6. Clustering time (s, mean per recording, 4 workers in parallel)\n")
ct = defaultdict(list)
for r in rows:
    ct[(r["model"], r["step"], r["method"])].append(r["cluster_s"])
table(["model", "step", "method", "cluster s"], [[*k, f"{sum(v) / len(v):.1f}"] for k, v in sorted(ct.items())])
