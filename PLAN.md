# Echoscript — Plan

Status: defaults (table below) confirmed on 2026-10-06. Current step: 1 (environment).

## Architecture for v1 (mode A: process after stop)

### Processes and cores

```
                        Raspberry Pi 5 (headless, edge device)
┌──────────────────────────────────────────────────────────────────────────────┐
│ core 0 ─────────────────────────────────┐  cores 1-3 ───────────────────────┐ │
│                                         │                                  │ │
│  FastAPI app (uvicorn)                  │  whisper-server (systemd service)│ │
│   • serves your index.html              │   • large-v3-turbo (q5_0/q8_0)   │ │
│   • REST + WebSocket                    │   • -l tr, -t 3, model kept      │ │
│   • SQLite (sessions, jobs, progress)   │     loaded, own VAD off          │ │
│                                         │   • pinned: taskset -c 1-3       │ │
│  Recorder  (high priority)              │                                  │ │
│   arecord → audio.wav (16k mono s16)    │                                  │ │
│   + live Silero VAD → vad.json          │                                  │ │
│                                         │                                  │ │
│  Pipeline job (low priority, nice +10)  │                                  │ │
│   ├─ STT worker ── HTTP chunks ─────────┼──► (does the actual compute)     │ │
│   └─ Diar worker (ONNX, 1 thread)       │                                  │ │
└─────────────────────────────────────────┴──────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
        ▲ HTTPS/WS (LAN now, remote access later)
        │
   Browser on PC/phone: your index.html
```

### Data flow for one session

```
 [Start] ─► Recorder ─► audio.wav ─┐
                └─► live VAD ─► vad.json (speech regions, ms)
 [Stop]  ─► job queued
                                   ▼
                     ┌──── Chunk packer (≤30 s, cut at silence) ──── chunks.json
                     │
          ┌──────────┴───────────┐        runs in parallel
          ▼                      ▼
   STT worker              Diar worker (DIY Stage 1)
   chunk → whisper-server   windows 1.5 s / step 0.75 s (VAD regions only)
   + previous-text prompt   → embeddings (batched ONNX)
   → words.json             → clustering (unknown N, threshold)
     [start_ms, end_ms,      → smoothing → turns.json (+ .rttm)
      text, prob]
          └──────────┬───────────┘
                     ▼
              Merge: word → speaker (max overlap),
              snap boundaries to word gaps, group into utterances
                     ▼
              transcript.json ─► SQLite status "done" ─► WebSocket ─► UI
              audio.wav → audio.flac (archive)
```

### Key design choices in this picture

- **VAD runs live during recording.** It costs very little and only needs one core. When you press stop, the speech regions are already done. It's also the first building block for live mode (B) later.
- **whisper-server is permanently loaded** (~1 GB RAM of 16). There's no model loading per job, and the STT worker is just a lightweight HTTP client.
- **Every stage writes its own file** (vad / chunks / words / turns / transcript). Any single stage can be rerun, e.g. diarization with sherpa instead of DIY, without redoing STT. Good for learning and debugging.
- **One processing job at a time.** The recorder always has priority, so you can start a new recording while the previous one is still processing.
- **Shared timeline:** all times are integer ms from session start. Word time = chunk start + Whisper's local time.

### Folder structure

```
echoscript/
├── config.toml              # all knobs: paths, threads, cores, VAD, packer, clustering
├── app/        main.py · api.py · ws.py · db.py
├── audio/      recorder.py · vad.py · io.py
├── pipeline/   orchestrator.py · packer.py · merge.py · types.py
├── stt/        base.py · whispercpp_server.py
├── diar/       base.py · diy/{windows,embed,cluster,smooth}.py · sherpa.py · pyannote.py (later)
├── eval/       bench_threads.py · wer.py · der.py
├── web/        index.html   (your existing file)
├── models/     ggml-turbo-q5_0.bin · silero_vad.onnx · wespeaker_resnet34.onnx
├── data/sessions/<id>/  audio.flac · vad.json · chunks.json · words.json · turns.json · transcript.json
└── deploy/     whisper-server.service · echoscript.service
```

## System details (confirmed defaults)

| # | Detail | Proposal | Why |
|---|---|---|---|
| 1 | Recording format | 16 kHz mono s16 WAV while recording → FLAC afterwards | All the models want 16 kHz; FLAC halves the storage |
| 2 | VAD timing | Live during recording | Ready when you press stop; prepares mode B |
| 3 | VAD parameters | threshold 0.5, min speech 250 ms, min silence 300 ms, 200 ms padding | Silero's standard values; tune on the dev set |
| 4 | Chunk packing | Contiguous original audio ≤ 30 s, cut in the middle of a silence; silences > 2 s are skipped | Simple time mapping, no words cut in half |
| 5 | whisper.cpp integration | `whisper-server`, pinned to cores 1-3 | Model stays loaded and runs as an isolated process |
| 6 | Model / quantization | turbo, q5_0 vs q8_0 → decided by benchmark | Quality vs. speed on Turkish |
| 7 | Decoding | `-l tr`, beam 5 vs greedy → benchmark; prompt = last ~200 characters | Accuracy and continuity between chunks |
| 8 | Word timestamps | Token timestamps + DTW, merged into words (UTF-8-safe) | The ç/ğ/ş issue and the merge step |
| 9 | Core split | 3 (STT) + 1 (everything else) → confirmed by thread benchmark | STT is the long pole |
| 10 | Embedding model | ~~WeSpeaker ResNet34~~ → **CAM++** ONNX (decided 2026-10-19, DOCUMENTATION.md §10) | Same speakers as ResNet34, 3.2× faster |
| 11 | Clustering | ~~Agglomerative, threshold~~ → **spectral (NME-SC)**, p_ratio 0.25; merge clusters < 5 s (decided 2026-10-19, DOCUMENTATION.md §14) | Finds the speaker count itself; AHC's threshold had no stable range on long recordings |
| 12 | Job/progress | SQLite job table; workers write progress, WebSocket pushes it | No Redis needed; survives restarts |

## Build order

1. **Environment:** build whisper.cpp (NEON/OpenMP), quantize turbo to q5_0/q8_0, Python venv, ONNX models.
2. **Thread + quantization benchmark** on a Turkish clip → decides #6, #7 and #9.
3. Recorder + live VAD + sessions + API + serving your `index.html`.
4. Chunk packer → STT worker → `words.json`.
5. DIY diarization Stage 1 → `turns.json`.
6. Orchestrator: run both in parallel, core pinning, progress over WebSocket.
7. Merge → `transcript.json` → display in the UI.
8. Evaluation harness (WER/CER, DER) on your labelled dev set.

## Uploaded audio files

For uploaded files, the same VAD runs as one pass over the whole file instead of live. The upload just enters the pipeline at a different point:

```
Recording: mic ─► recorder + live VAD ─► audio.wav + vad.json ─┐
                                                               ├─► chunk packer ─► STT ∥ Diar ─► merge
Upload:    file ─► ffmpeg (16k mono) ─► offline VAD ─► vad.json ┘
```

- **Same VAD code, same settings.** Feed the file through Silero in 32 ms frames, as fast as the CPU allows. It's quick: roughly a minute or less per hour of audio on one core.
- **Convert first.** ffmpeg turns any format (mp3, m4a, wav, video) into 16 kHz mono, so everything after this step is identical.
- **No need to wait for VAD to finish.** VAD runs well ahead of STT, so the chunk packer can start sending chunks as soon as the first speech regions are found.

One change to the project: **write `vad.py` as a frame-by-frame processor.** Live recording feeds it from the mic, and an upload feeds it from the file. Both use the same code and give the same results.
