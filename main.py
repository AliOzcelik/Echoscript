"""Echoscript web app: serves the UI, records from the Pi's mic, takes uploads, runs one pipeline job at a time.

    python main.py          # http://<pi>:8000

Every meeting is a session folder, data/sessions/<id>/, with meeting.json next to the pipeline's files.
A recording is transcribed while it goes on (live_stt): at stop only the last chunks are left,
then the pipeline goes on from diar. The pipeline runs as `python -m pipeline.run` in a subprocess: a crash can't take the server down,
and all model memory is freed when a job ends. whisper-server (and llama-server for
[summary] backend = "llama") must already be running:

The meeting is done when its transcript is: the summary is a separate job with its own queue, started
right after the transcript if the meeting was created with auto_summary, else from the UI's Summarize
button. A failed summary leaves the transcript as it is. Summaries run one at a time, next to the
pipeline (on the AI HAT+2 they hardly use the CPU).

    taskset -c 1-3 ~/src/whisper.cpp/build/bin/whisper-server -m models/ggml-large-v3-turbo-q8_0.bin -l tr -t 3 --port 8080

q8_0 encodes a 30 s chunk 2.4x faster than q5_0 on the Pi 5 with the same text, and a 4th thread only
gains 8%, so core 0 stays for the pipeline (diar runs there next to stt).
"""
import asyncio
import io
import json
import re
import shutil
import sys
import uuid
import wave
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from pipeline.config import SAMPLE_RATE, load_config
from pipeline.live import LiveChunker
from pipeline.stt import tail, transcribe_chunk
from pipeline.vad import StreamingVAD


ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "Echoscript_standalone.html"
CONFIG = load_config(ROOT / "config.toml")
SESSIONS = ROOT / CONFIG.data_dir
VAD_CONFIG = replace(CONFIG.vad, model=ROOT / CONFIG.vad.model)
LOG_LINES = 12          # pipeline output lines kept per meeting for the processing screen
MAX_CLIP_MS = 30000     # speaker samples only, never a whole recording in memory
# speech without a pause is cut at the quietest point of the last 5 s, with the previous text as
# prompt: E2 in eval/bench_chunking, the fewest words lost at the cuts on the test recordings
LIVE_CUT = "quiet"
LIVE_POLL_S = 0.2       # how often live_stt looks for a chunk while whisper-server is free
# Linux: the pipeline runs on core 0, whisper-server has 1-3. Elsewhere (no taskset) the OS places it.
PIN_CORE_0 = ["taskset", "-c", "0"] if shutil.which("taskset") else []

meetings = {}           # id -> meeting.json contents
queue = asyncio.Queue() # ids of meetings waiting for the pipeline
running = None          # the pipeline subprocess, while a job runs
summary_queue = asyncio.Queue()     # ids of meetings waiting for a summary
summarizing = None      # the summary subprocess, while one runs
recorder = None         # the recording in progress: {"id", "proc", "task", "bytes", "level", "vad", "regions",
                        #   "chunker", "chunks", "words", "stopped", "live", "live_error"}
finishing = set()       # stopped recordings whose last chunks live_stt is still transcribing


def save(m):
    path = SESSIONS / m["id"] / "meeting.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)       # atomic: a crash never leaves a half-written file


def find(mid):
    if mid not in meetings:
        raise HTTPException(404, "unknown meeting")
    return meetings[mid]


def new_meeting(title, language, suffix, status, first_stage="convert", auto_summary=False):
    """A session folder and its meeting.json. The input file is <id><suffix>: run.py names the folder after it.
    first_stage: where the pipeline starts (a recording already has audio.wav, and vad.json from the live VAD).
    auto_summary: summarize as soon as the transcript is ready."""
    if language not in ("auto", "en", "tr"):
        raise HTTPException(400, "language must be auto, en or tr")

    mid = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    (SESSIONS / mid).mkdir(parents=True)
    m = {
        "id": mid, "title": title, "created": datetime.now().isoformat(timespec="seconds"),
        "language": language, "input": f"{mid}{suffix}", "duration_s": None, "speakers": {},
        "first_stage": first_stage, "status": status, "stage": None, "progress": None, "side": None,
        "log": [], "error": None,
        # summary_status: None (not asked for) | "queued" | "running" | "done" | "error"
        # summary_progress: "3/9" (parts of a long transcript), "combining", or None
        "auto_summary": auto_summary, "summary_status": None, "summary_progress": None, "summary_error": None,
    }
    meetings[mid] = m
    save(m)
    return m


def drop(m):
    del meetings[m["id"]]
    shutil.rmtree(SESSIONS / m["id"])



# Pipeline worker

async def worker():
    while True:
        m = meetings[await queue.get()]
        try:
            await process(m)
        except Exception as exc:    # one broken meeting must not stop the queue
            m.update(status="error", error=str(exc))
            save(m)


async def process(m):
    global running
    sdir = SESSIONS / m["id"]
    m.update(status="processing", stage=None, progress=None, side=None, log=[], error=None)
    save(m)

    # core 0: whisper-server has cores 1-3. stt waits on it while diar computes here (--parallel).
    # Up to the transcript; the summary is its own job (summarize()).
    running = await asyncio.create_subprocess_exec(
        *PIN_CORE_0, sys.executable, "-u", "-m", "pipeline.run", str(sdir / m["input"]),
        "--from", m["first_stage"], "--to", "merge", "--language", m["language"], "--parallel",
        cwd=ROOT, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    beside = None       # the stage the side stage started next to
    async for raw in running.stdout:
        line = raw.decode(errors="replace").strip()
        side = m.get("side")
        if line.startswith("+ "):           # a stage starts next to the next one: "+ diar next to stt"
            m["side"], beside = {"stage": line[2:].split()[0], "done": False}, None
            save(m)
        elif line.startswith("= "):         # ...and is done: "= diar done"
            if side:
                side["done"] = True
                save(m)
        elif line.startswith("> "):         # run.py starts a stage
            stage = line[2:]
            if side and stage == side["stage"]:
                continue                    # the side stage's output, printed after the other one ended
            if side and beside is None:
                beside = stage
            elif side:
                m["side"] = None            # the stage it ran next to is over
            m.update(stage=stage, progress=None)
            if m["duration_s"] is None and (sdir / "audio.wav").exists():
                m["duration_s"] = round(sf.info(str(sdir / "audio.wav")).duration, 1)
            save(m)
        elif line:
            if n := re.search(r"chunk (\d+)/(\d+)", line):     # stt chunks
                m["progress"] = int(n[1]) / int(n[2])
            m["log"] = (m["log"] + [line])[-LOG_LINES:]
    code = await running.wait()
    running = None

    if code == 0:
        utterances = json.loads((sdir / "transcript.json").read_text(encoding="utf-8"))["utterances"]
        ids = dict.fromkeys(u["speaker"] for u in utterances)      # in order of first appearance
        m["speakers"] = {s: m["speakers"].get(s, {"name": "", "color": None}) for s in ids}
        m.update(status="done", stage=None, progress=None, side=None)
        if m.get("auto_summary"):
            queue_summary(m)
    else:
        m.update(status="error", side=None, error=m["log"][-1] if m["log"] else f"pipeline exit code {code}")
    save(m)



# Summary worker

def queue_summary(m):
    m.update(summary_status="queued", summary_progress=None, summary_error=None)
    save(m)
    summary_queue.put_nowait(m["id"])


async def summary_worker():
    while True:
        m = meetings[await summary_queue.get()]
        try:
            await summarize(m)
        except Exception as exc:    # the transcript stays; only the summary failed
            m.update(summary_status="error", summary_progress=None, summary_error=str(exc))
            save(m)


async def summarize(m):
    """run.py's summary stage for a done meeting. It writes summary.md; get_meeting reads it."""
    global summarizing
    sdir = SESSIONS / m["id"]
    m.update(summary_status="running", summary_progress=None, summary_error=None)
    save(m)

    summarizing = await asyncio.create_subprocess_exec(
        *PIN_CORE_0, sys.executable, "-u", "-m", "pipeline.run", str(sdir / m["input"]),
        "--from", "summary", "--to", "summary", "--language", m["language"],
        cwd=ROOT, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    last = ""
    async for raw in summarizing.stdout:
        if not (line := raw.decode(errors="replace").strip()):
            continue
        last = line
        if n := re.search(r"part (\d+)/(\d+)", line):     # a long transcript, in parts
            m["summary_progress"] = f"{n[1]}/{n[2]}"
        elif line.startswith("combining"):              # ...whose partial summaries are merged
            m["summary_progress"] = "combining"
    code = await summarizing.wait()
    summarizing = None

    if code == 0:
        m.update(summary_status="done", summary_progress=None)
    else:
        m.update(summary_status="error", summary_progress=None, summary_error=last or f"summary exit code {code}")
    save(m)


@asynccontextmanager
async def lifespan(app):
    SESSIONS.mkdir(parents=True, exist_ok=True)
    for path in sorted(SESSIONS.glob("*/meeting.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        meetings[m["id"]] = m
        if "summary_status" not in m:       # a meeting from before summaries were their own job
            done = (path.parent / "summary.md").exists()
            m.update(auto_summary=False, summary_status="done" if done else None, summary_progress=None, summary_error=None)
        # the server stopped mid-job: run it again. A cut-off recording is a valid WAV up to that point.
        if m["status"] in ("recording", "queued", "processing"):
            m.update(status="queued", stage=None, progress=None, side=None)
            save(m)
            queue.put_nowait(m["id"])
        elif m["status"] == "done" and m["summary_status"] in ("queued", "running"):
            queue_summary(m)

    task = asyncio.create_task(worker())
    summary_task = asyncio.create_task(summary_worker())
    yield
    task.cancel()
    summary_task.cancel()
    if running is not None:
        running.kill()              # requeued at the next start
    if summarizing is not None:
        summarizing.kill()          # requeued at the next start
    for t in finishing:
        t.cancel()                  # "processing": transcribed again from pack at the next start
    if recorder is not None:
        recorder["live"].cancel()
        recorder["proc"].terminate()
        await recorder["task"]      # closes the WAV; the meeting is queued at the next start


app = FastAPI(title="Echoscript", lifespan=lifespan)



# API

@app.get("/")
async def index():
    return FileResponse(FRONTEND)


@app.get("/api/meetings")
async def list_meetings():
    return sorted(meetings.values(), key=lambda m: m["created"], reverse=True)


@app.post("/api/meetings")
async def create_meeting(file: UploadFile, title: str = Form(""), language: str = Form("auto"),
                         auto_summary: bool = Form(False)):
    """An uploaded file: any format ffmpeg can read."""
    suffix = re.sub(r"[^a-z0-9.]", "", Path(file.filename or "").suffix.lower()) or ".bin"
    m = new_meeting(title.strip() or Path(file.filename or "upload").stem, language, suffix, "queued",
                    auto_summary=auto_summary)
    with open(SESSIONS / m["id"] / m["input"], "wb") as f:
        await asyncio.to_thread(shutil.copyfileobj, file.file, f)
    queue.put_nowait(m["id"])
    return {"id": m["id"]}


class RecordStart(BaseModel):
    title: str = ""
    language: str = "auto"
    auto_summary: bool = False


@app.post("/api/record/start")
async def record_start(req: RecordStart):
    """Records from the mic on the Pi ([record] device) until /api/record/stop."""
    global recorder
    if recorder is not None:
        raise HTTPException(409, "already recording")

    # arecord already gives the pipeline's format, so it writes audio.wav and convert is skipped.
    # Until the live VAD's regions are saved at stop, the pipeline would start at vad (after a crash).
    m = new_meeting(req.title.strip() or f"Recording {datetime.now():%Y-%m-%d %H:%M}", req.language, ".wav", "recording", "vad",
                    auto_summary=req.auto_summary)
    sdir = SESSIONS / m["id"]
    # raw 16 kHz mono s16: the pipeline's format. stderr goes to a file: an unread pipe could fill and block arecord.
    try:
        with open(sdir / "arecord.log", "wb") as log:
            proc = await asyncio.create_subprocess_exec(
                "arecord", "-q", "-D", CONFIG.record.device, "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "1", "-t", "raw",
                stdout=asyncio.subprocess.PIPE, stderr=log,
            )
    except FileNotFoundError:       # no ALSA (macOS, Windows): uploads still work
        drop(m)
        raise HTTPException(500, "arecord not found: recording needs Linux with ALSA. Upload a file instead.")
    await asyncio.sleep(0.3)        # a wrong device or a busy mic makes arecord quit at once
    if proc.returncode is not None:
        error = (sdir / "arecord.log").read_text(errors="replace").strip() or f"exit code {proc.returncode}"
        drop(m)
        raise HTTPException(500, f"arecord: {error}")

    recorder = {"id": m["id"], "proc": proc, "bytes": 0, "level": 0.0, "vad": StreamingVAD(VAD_CONFIG), "regions": [],
                "chunker": LiveChunker(CONFIG.stt, cut=LIVE_CUT, wav=sdir / "audio.wav"), "chunks": [], "words": [],
                "stopped": False, "live_error": None}
    recorder["task"] = asyncio.create_task(capture(recorder, sdir / "audio.wav"))
    stt = replace(CONFIG.stt, language=req.language)       # as run.py --language does
    recorder["live"] = asyncio.create_task(live_stt(recorder, sdir / "audio.wav", stt))
    return {"id": m["id"]}


async def capture(rec, path):
    """arecord's output -> the WAV file, the live VAD (speech regions are ready at stop),
    and the loudness for the UI's level bars."""
    rest = b""      # an odd byte left over from a read: half a sample
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        while chunk := await rec["proc"].stdout.read(SAMPLE_RATE // 5):     # up to 100 ms
            w.writeframes(chunk)    # rewrites the header each time: the file stays valid if the server dies
            rec["bytes"] += len(chunk)

            data = rest + chunk
            n = len(data) // 2 * 2
            pcm = np.frombuffer(data[:n], dtype=np.int16).astype(np.float32) / 32768
            rest = data[n:]
            new = rec["vad"].feed(pcm)                  # ~3 Silero frames per read: well under 1 ms
            rec["regions"] += new
            for r in new:
                rec["chunker"].add_region(r)
            rec["chunker"].set_open(rec["vad"].open_start_ms)
            rec["level"] = float(np.sqrt(np.mean(pcm ** 2))) if len(pcm) else 0.0


async def live_stt(rec, wav, stt):
    """Transcribes the recording while it goes on: whenever whisper-server is free, LiveChunker picks
    the next chunk from the speech so far (whisper takes ~25 s per chunk, so chunks fill up while it
    works). Once rec["stopped"], sends what is left and returns. On an error it stops and leaves
    the recording to the pipeline (from pack, after stop)."""
    chunker, m = rec["chunker"], meetings[rec["id"]]
    prompt = ""
    try:
        with httpx.Client(base_url=stt.server_url, timeout=stt.timeout_s) as client:
            while True:
                final = rec["stopped"]
                chunk = chunker.next_chunk(rec["bytes"] // 2 * 1000 // SAMPLE_RATE, final)
                if chunk is None:
                    if final and chunker.empty:
                        return
                    await asyncio.sleep(LIVE_POLL_S)
                    continue
                new, took = await asyncio.to_thread(transcribe_chunk, client, wav, chunk, prompt, stt)
                rec["chunks"].append(chunk)
                rec["words"] += new
                prompt = tail(f"{prompt} {' '.join(w.text for w in new)}", stt.prompt_chars)
                lag = rec["bytes"] / 2 / SAMPLE_RATE - chunk.end_ms / 1000
                m["log"] = (m["log"] + [f"chunk {chunk.index + 1} | {chunk.reason:6s} | {chunk.duration_ms / 1000:4.1f} s "
                                        f"| {len(new):3d} words | {took:5.1f} s | {lag:4.0f} s behind"])[-LOG_LINES:]
                if final:       # after stop the length is known: progress = the share of it transcribed
                    m["progress"] = min(1.0, chunk.end_ms * SAMPLE_RATE * 2 / 1000 / rec["bytes"])
    except Exception as exc:
        rec["live_error"] = f"{type(exc).__name__}: {exc}"


async def finish_live(rec, m):
    """After stop: waits for live_stt's last chunks, writes the files the pipeline's stt stage
    would have, and queues the meeting from diar. If live_stt failed, from pack."""
    sdir = SESSIONS / m["id"]
    await rec["live"]
    if rec["live_error"] is None:
        write_json(sdir / "chunks.json", {"chunks": [c.to_dict() for c in rec["chunks"]]})
        write_json(sdir / "words.json", {"words": [w.to_dict() for w in rec["words"]]})
        (sdir / "words.txt").write_text(" ".join(w.text for w in rec["words"]) + "\n", encoding="utf-8")
        write_labels(sdir / "chunks.labels.txt", [(c.start_ms, c.end_ms, f"{c.index} {c.reason}") for c in rec["chunks"]])
        write_labels(sdir / "words.labels.txt", [(w.start_ms, w.end_ms, w.text) for w in rec["words"]])
        m["first_stage"] = "diar"
    else:
        m["log"] = (m["log"] + [f"live transcription failed ({rec['live_error']}): transcribing again"])[-LOG_LINES:]
    m.update(status="queued", stage=None)
    save(m)
    queue.put_nowait(m["id"])


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def write_labels(path, rows):
    """Audacity labels, as pipeline/run.py writes them."""
    path.write_text("".join(f"{s / 1000:.3f}\t{e / 1000:.3f}\t{label}\n" for s, e, label in rows), encoding="utf-8")


@app.get("/api/record")
async def record_status():
    """The recording in progress, polled by the UI for its timer and level bars."""
    if recorder is None:
        return {"id": None}
    return {"id": recorder["id"], "seconds": recorder["bytes"] / 2 / SAMPLE_RATE, "level": recorder["level"]}


class RecordStop(BaseModel):
    auto_summary: bool | None = None    # the choice can change while recording; None = keep the one from start


@app.post("/api/record/stop")
async def record_stop(req: RecordStop | None = None):
    global recorder
    rec = recorder
    if rec is None:
        raise HTTPException(409, "not recording")
    recorder = None
    if req is not None and req.auto_summary is not None:
        meetings[rec["id"]]["auto_summary"] = req.auto_summary

    if rec["proc"].returncode is None:      # still running (a mic unplugged mid-recording ends it early)
        rec["proc"].terminate()
    await rec["task"]       # reads what arecord still had, then closes the WAV
    await rec["proc"].wait()

    m = meetings[rec["id"]]
    if rec["bytes"] < SAMPLE_RATE * 2:      # under a second: a mis-click, nothing to process
        rec["live"].cancel()
        drop(m)
        return {"id": None}

    # the same files run.py's vad stage writes
    sdir = SESSIONS / m["id"]
    last = rec["vad"].flush()
    regions = rec["regions"] + last
    write_json(sdir / "vad.json", {"regions": [r.to_dict() for r in regions]})
    write_labels(sdir / "vad.labels.txt", [(r.start_ms, r.end_ms, "speech") for r in regions])

    # live_stt sends the rest; until it is done, a server restart transcribes it all again from pack
    for r in last:
        rec["chunker"].add_region(r)
    rec["chunker"].set_open(None)
    rec["stopped"] = True
    m.update(first_stage="pack", status="processing", stage="stt", progress=None)
    save(m)
    task = asyncio.create_task(finish_live(rec, m))
    finishing.add(task)
    task.add_done_callback(finishing.discard)
    return {"id": m["id"]}


@app.get("/api/meetings/{mid}")
async def get_meeting(mid: str):
    """The meeting plus its transcript and summary, once the pipeline has written them."""
    m = find(mid)
    sdir = SESSIONS / mid
    utterances = []
    if (sdir / "transcript.json").exists():
        utterances = json.loads((sdir / "transcript.json").read_text(encoding="utf-8"))["utterances"]
    summary = (sdir / "summary.md").read_text(encoding="utf-8") if (sdir / "summary.md").exists() else ""
    return {
        **m,
        "utterances": [{k: u[k] for k in ("speaker", "start_ms", "end_ms", "text")} for u in utterances],
        "summary": summary_sections(summary),
    }


class Speaker(BaseModel):
    name: str = ""
    color: int | None = None    # index into the UI's palette; None = the default for its position


@app.put("/api/meetings/{mid}/speakers")
async def set_speakers(mid: str, speakers: dict[str, Speaker]):
    m = find(mid)
    m["speakers"].update({s: v.model_dump() for s, v in speakers.items() if s in m["speakers"]})
    save(m)
    return m["speakers"]


@app.post("/api/meetings/{mid}/summary")
async def start_summary(mid: str):
    """The UI's Summarize button: queue a summary of a done meeting (again, after an error)."""
    m = find(mid)
    if m["status"] != "done":
        raise HTTPException(409, "the transcript is not ready yet")
    if m.get("summary_status") not in ("queued", "running"):
        queue_summary(m)
    return m


@app.get("/api/meetings/{mid}/audio")
async def audio_clip(mid: str, start_ms: int, end_ms: int):
    """A short WAV clip of the meeting, e.g. a speaker sample."""
    find(mid)
    wav = SESSIONS / mid / "audio.wav"
    if not wav.exists():
        raise HTTPException(404, "no audio yet")

    per_ms = SAMPLE_RATE // 1000
    end_ms = min(end_ms, start_ms + MAX_CLIP_MS)
    audio, _ = sf.read(str(wav), start=max(0, start_ms) * per_ms, stop=end_ms * per_ms, dtype="int16")
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav")


def summary_sections(text):
    """summary.md -> the four sections the UI shows, in the order summarize.py asks for them."""
    sections = []
    for line in text.splitlines():
        if line.startswith("#"):
            sections.append([])
        elif line.strip() and sections:
            sections[-1].append(re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).replace("**", "").strip())

    sections = [s for s in sections if s]       # e.g. a "# Summary" title line the model added
    if not sections and text.strip():
        sections = [[text.strip()]]             # no headings: "(transcript too short to summarize)"
    overview, key, decisions, actions = (sections + [[], [], [], []])[:4]
    return {"overview": " ".join(overview), "key": key, "decisions": decisions, "actions": actions}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
