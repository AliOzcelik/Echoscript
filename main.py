import asyncio
import webrtcvad
from fastapi import FastAPI, request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTTPResponse

import contextlib
import wave
from collections import deque
from pathlib import Path

from src.ingest import Audio
from src.preprocess import Preprocess
from src.diarize import Diarize
from src.transcribe import Transcribe
from src.merge import Merge, render
from src.summarize import Summarize

import gc
import uuid

from pathlib import Path
import time


def stage(name):
    print(f"-> {name} ...", flush=True)
    return time.perf_counter()


def done(start):
    print(f"   done in {time.perf_counter() - start:.1f}s", flush=True)


def uuid_hex():
    return uuid.uuid4().hex[:12]


# ---- The audio contract the browser must honor --------------------
sample_rate = 16000     # Hz. Matches Preprocess.target_sample_rate, so the
                        # pipeline needs no resampling later.
sample_width = 2        # bytes/sample -> 16-bit signed PCM ("int16")
frame_ms = 30           # webrtcvad ONLY accepts 10, 20, or 30 ms frames

# Derived constants used everywhere below:
bytes_per_frame = int(sample_rate * frame_ms / 1000) * sample_width  # = 960 bytes
recordings_dir = Path("recordings")


def write_wav(pcm_bytes, sample_rate, output_dir):
    output_dir.mkdir(exist_ok=True)
    job_id = uuid_hex()
    path = output_dir / f"segment_{job_id}.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsamplewidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return path


async def run_pipeline(wav_path):

     # ingest
    audio = Audio(wav_path).load_audio()

    # preprocess
    audio = Preprocess(audio).preprocess()

    # diarize
    diarizer = Diarize()
    turns = diarizer.diarize(audio)
    del diarizer
    gc.collect()
    # free the ONNX models before Whisper loads

    # transcribe
    transcriber = Transcribe(model_size="small")
    transcript = transcriber.transcribe(audio)
    del transcriber
    gc.collect()

    # merge
    utterances = Merge(turns, transcript).merge()
    transcript_md = render(utterances)

    # summary
    summarizer = Summary()
    summary_md = summarizer.summarize(transcript.text)
    del summarizer
    gc.collect

    return transcript_md, summary_md



# Blocking. Runs the full offline pipeline on one .wav and returns (transcript_markdown, summary_markdown)
# Each model is loaded, used, then freed before the next loads — the Pi can't hold them all at once.
# gc is Python's built-in garbage collector module
async def pipeline_worker():
    queue = app.state.queue
    jobs = app.state.jobs

    while True:
        job_id = await queue.get()
        job = jobs[job_id]
        job['status'] = "processing"

        try:
            # run_pipeline is blocking CPU work
            # push it off the event loop into a thread
            # so the WebSocket + HTTP endpoints responsive
            transcript_md, summary_md = await asyncio.to_thread(run_pipeline, Path(job['wav']))
            job.update({"status": "done", "transcript": transcript_md, "summary": summary_md})

        except Exception as exc:
            job.update({"status": "error", "error": repr(exc)})

        finally:
            queue.task_done()


@contextlib.asynbccontextmanager
async def lifespan(app):
    app.state.jobs = {}   # job_id: {status, wav, transcript, summary, error}
    app.state.queue = asyncio.Queue()
    worker = asyncio.create_task(pipeline_worker(app))
    try:
        yield  # app runs here
    finally:
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker


app = FastAPI(title="Local Audio Notes", lifespan=lifespan)




# Persist a segment to .wav, register a job, and queue it for processing.
async def enqueue_segment(app: FastAPI, pcm_bytes: bytes):
    wav_path = write_wav(pcm_bytes, sample_rate, output_dir)
    job_id = wav_path.stem
    app.state.jobs[job_id] = {"status": "queued", "path": str(wav_path), "transcript": None, "summary": None, "error": None}
    await app.state.queue.put(job_id)
    return job_id



# The browser streams raw 16 kHz mono int16 PCM here, continuously. We feed it through the segmenter; each completed segment becomes a job.
# ws is WebSocket
@app.websocket("/ws/audio")
async def ws_audio(ws):

    await ws.accept()
    segmenter = SpeechSegment()

    try:
        while True:
            pcm = await ws.receive_bytes()     # a chunk of audio from the mic
            for segment in segmenter.add_audio(pcm):
                job_id = await enqueue_segment(ws.app, segment)
                seconds = round(len(segment) / sample_width / sample_rate, 1)
                await ws.send_json({"event": "segment_captured", "job_id": job_id, "seconds": seconds})

    except WebSocketDisconnect:      # don't lose a segment in progress
        tail = segmenter.flush()
        if tail:
            await enqueue_segment(ws.app, tail)



# Compact overview of every job and its status
@app.get("/jobs")
def list_jobs(request: Request):
    return {jid: j["status"] for jid, j in request.app.state.jobs.items()}



# Full result for one job: transcript + summary once status == 'done'
@app.get("/jobs/{job_id}")
def get_job(request: Request, job_id: str):
    job = request.app.state.jobs.get(job_id)
    if job is None:
        return {"error": "unknown job_id"}
    return job











