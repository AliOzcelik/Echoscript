"""Full-system test (DOCUMENTATION.md §21): upload a recording through the web API as the UI does,
with the automatic summary on, follow it to the end, and sample memory and temperature every second.
whisper-server and the app must be running.

    python eval/full_run.py http://127.0.0.1:8000 recordings/oscar_da_bizi_gorecek_mi/audio.wav runs/full

Writes <out_dir>/samples.csv (one row per second) and <out_dir>/events.txt (stage changes).
Memory: system = MemTotal - MemAvailable; per process = VmRSS, by role (whisper-server, app,
pipeline, diar, summary). Temperature: thermal_zone0."""
import csv
import subprocess
import sys
import time
from pathlib import Path

import httpx

base, audio, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
c = httpx.Client(base_url=base, timeout=60)


def meminfo():
    kv = dict(l.split(":", 1) for l in Path("/proc/meminfo").read_text().splitlines())
    kb = lambda k: int(kv[k].split()[0])
    return (kb("MemTotal") - kb("MemAvailable")) / 1024     # MiB in use (not reclaimable cache)


def procs():
    """MiB of RSS per role: whisper-server, app (uvicorn), pipeline, diar, summary."""
    rss = {"whisper": 0, "app": 0, "pipeline": 0, "diar": 0, "summary": 0}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            st = (d / "status").read_text()
        except OSError:
            continue
        if "VmRSS:" not in st:
            continue
        mib = int(st.split("VmRSS:")[1].split()[0]) / 1024
        if "whisper-server" in cmd and "taskset" not in cmd:
            rss["whisper"] += mib
        elif "uvicorn main:app" in cmd and "python" in cmd:
            rss["app"] += mib
        elif "pipeline.run" in cmd and "python" in cmd and "taskset" not in cmd:
            role = "diar" if "--from diar --to diar" in cmd else "summary" if "--from summary" in cmd else "pipeline"
            rss[role] += mib
    return rss


def temp():
    return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000


with open(audio, "rb") as f:
    t0 = time.time()
    mid = c.post("/api/meetings", files={"file": (audio.name, f, "audio/wav")},
                 data={"title": "Oscar — full test", "language": "tr", "auto_summary": "true"}).json()["id"]
upload_s = time.time() - t0
events = [f"{0:8.1f} s  uploaded {audio.name} -> {mid} ({upload_s:.1f} s)"]
print(events[0], flush=True)

last = None
with open(out / "samples.csv", "w", newline="") as fcsv:
    w = csv.writer(fcsv)
    w.writerow(["t_s", "system_used_mib", "whisper", "app", "pipeline", "diar", "summary", "temp_c", "status", "stage", "side", "summary_status", "summary_progress"])
    while True:
        m = next(x for x in c.get("/api/meetings").json() if x["id"] == mid)
        t = time.time() - t0
        side = m.get("side") or {}
        side_s = f"{side.get('stage')}:{'done' if side.get('done') else 'running'}" if side else ""
        state = (m["status"], m["stage"], side_s, m["summary_status"])
        if state != last:
            e = f"{t:8.1f} s  status={m['status']} stage={m['stage']} side={side_s or '-'} summary={m['summary_status']}"
            events.append(e)
            print(e, flush=True)
            last = state
        r = procs()
        w.writerow([round(t, 1), round(meminfo()), *(round(r[k]) for k in ("whisper", "app", "pipeline", "diar", "summary")),
                    temp(), m["status"], m["stage"], side_s, m["summary_status"], m["summary_progress"]])
        fcsv.flush()
        if m["status"] == "error" or m["summary_status"] in ("done", "error") or (m["status"] == "done" and not m["summary_status"]):
            break
        time.sleep(1)

d = c.get(f"/api/meetings/{mid}").json()
events.append(f"log tail: {d['log'][-6:]}")
events.append(f"error: {d['error']} | summary_error: {d['summary_error']}")
events.append(f"utterances: {len(d['utterances'])} | speakers: {list(d['speakers'])} | duration_s: {d['duration_s']}")
throttled = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True).stdout.strip()
events.append(f"after: {throttled}")
(out / "events.txt").write_text("\n".join(events) + "\n")
print("\n".join(events[-4:]))
print("MID", mid)
