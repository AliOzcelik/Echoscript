# Local Audio → Speaker-Labeled Notes + Summary

**Project goal:** Take audio in → identify speakers → transcribe → summarize with a local
LLM → output a speaker-labeled transcript ("notes") plus a structured summary.

## Locked decisions (from planning Q&A)

| Decision | Choice | Consequence |
|---|---|---|
| Privacy | **Fully local / offline only** | No audio or text ever leaves the device. Rules out cloud as a *shippable* path. |
| Usage mode | **Batch** (record/upload, then process) | No streaming diarization needed → far simpler + lighter. |
| Scenario 1 ("no hardware") | **Existing computer, CPU-only**, compared against a cloud baseline | Cloud is benchmarked for accuracy reference, **not** shipped (violates privacy). |
| Scenario 2 ("minimal hardware") | **Tiny SBC** (Raspberry Pi 5 or Jetson Orin Nano) | Fully local, low power. Jetson recommended; Pi is the floor. |

## The pipeline (same for both scenarios)

```
[1] Ingest      load wav/mp3/m4a (batch)
[2] Preprocess  resample 16kHz mono, VAD trim, optional denoise
[3] Diarize     "who spoke when"  → speaker turns + timestamps
[4] Transcribe  Whisper → text + word timestamps
[5] Merge       align [3]+[4] → "Speaker 1: ...", "Speaker 2: ..."
[6] Summarize   local LLM → summary, action items, decisions
                → OUTPUT: transcript.md (notes) + summary.md
```

Hardest stages: **[3] diarization** (heaviest, accuracy-sensitive) and **[6] LLM**
(most RAM-hungry). [4] ASR is cheap and solved.

## Chosen stack (tuned for ARM/SBC survival, fully local)

| Stage | Primary choice | Why | Fallback |
|---|---|---|---|
| ASR | **whisper.cpp** (`base`/`small`, multilingual incl. Turkish) | Best ARM/NEON + Jetson CUDA support, tiny footprint | `faster-whisper` (int8) |
| Diarization | **sherpa-onnx** offline speaker diarization (ONNX segmentation + embeddings) | Runs well on CPU/ARM, no PyTorch needed | `pyannote.audio 3.1` (heavier, GPU/desktop only) |
| LLM | **llama.cpp** + Qwen2.5-3B-Instruct or Llama-3.2-3B-Instruct, **Q4_K_M** | ~2.3GB, fits 8GB SBC, strong summarizer | Phi-3.5-mini; 1.5B if RAM-tight |
| Orchestration | Python, run stages **sequentially** (free RAM between) | 8GB can't hold all models at once | — |

> Note: WhisperX bundles ASR+diarization+alignment but pulls in PyTorch/pyannote — great on
> a desktop (Scenario 1), heavy for a Pi. So: WhisperX on desktop, sherpa-onnx on the SBC.

## Scenario 1 — "No hardware" (existing computer, CPU-only)

- **Hardware:** whatever laptop/desktop you already have. No purchase.
- **Stack:** WhisperX (`small`, int8) for ASR+diarization+alignment + llama.cpp 3B Q4.
- **Performance:** batch ~0.5–2× audio length on a modern CPU (10-min meeting → a few min).
- **Cloud baseline (reference only, NOT shipped):** Deepgram/AssemblyAI (ASR+diarization in one
  call) + a hosted LLM. Use it once to measure "ground-truth" accuracy so you know how much the
  local models give up. Then discard — it breaks the fully-local rule.

## Scenario 2 — "Minimal hardware" (tiny SBC, fully local)

Two tiers, honestly rated:

| Tier | Board | ~Cost | Reality |
|---|---|---|---|
| **Floor** | Raspberry Pi 5, 8GB + NVMe + active cooling | ~$90–130 | CPU-only. whisper.cpp `base` + sherpa-onnx diarization + 3B LLM Q4. Works, but **minutes** to process a short clip; LLM ~2–4 tok/s. |
| **Recommended** | **Jetson Orin Nano Super, 8GB** | ~$249 | CUDA GPU → faster-whisper + pyannote on GPU + 3–7B LLM. Several× faster, still tiny + low-power. Best "minimal but real" pick. |

**8GB RAM budget (run sequentially):** Whisper `small` ~0.5GB → diarization ONNX ~0.2GB →
3B LLM Q4 ~2.3GB. Never co-resident; load → run → free → next stage.

## Repo structure (proposed)

```
note_taking/
  pipeline/
    ingest.py        # [1] load + validate audio
    preprocess.py    # [2] resample, VAD, denoise
    diarize.py       # [3] sherpa-onnx (SBC) / pyannote (desktop)
    transcribe.py    # [4] whisper.cpp / faster-whisper
    merge.py         # [5] align diarization + transcript
    summarize.py     # [6] llama.cpp local LLM + prompt templates
    run.py           # orchestrates 1→6 sequentially
  models/            # downloaded model weights (gitignored)
  config.yaml        # model paths, sizes, device (cpu/cuda), language
  out/               # transcript.md + summary.md per run
  tests/
```

## Phased plan

- **Phase 0 — Setup & smoke test (½ day).** Python env; download whisper `small` + a 3B GGUF;
  transcribe one clip; summarize a pasted transcript. Proves [4] and [6] end-to-end on your dev machine.
- **Phase 1 — Scenario 1 vertical slice (2–3 days).** Wire [1]→[6] with WhisperX + llama.cpp.
  Output `transcript.md` + `summary.md`. This is a working app on your existing computer.
- **Phase 2 — Quality pass (2–3 days).** Diarization accuracy (speaker count, boundaries);
  summary prompt engineering (summary / action items / decisions); handle Turkish + long audio (chunking).
- **Phase 3 — Cloud baseline benchmark (½ day).** Run one cloud pass for accuracy reference;
  record the gap; then remove it. Documents what "fully local" costs you.
- **Phase 4 — SBC port (3–5 days).** Swap diarization → sherpa-onnx; whisper.cpp build; sequential
  RAM management; measure on Pi 5 and/or Jetson. Tune model sizes to hit acceptable time.
- **Phase 5 — UX + packaging (2–3 days).** CLI (and optional minimal web UI) to drop a file and
  get notes; `config.yaml` profiles for `desktop` vs `pi` vs `jetson`.

## Top risks

1. **Diarization on Pi** is the #1 risk — slow and accuracy-sensitive. Mitigation: sherpa-onnx +
   known speaker count when possible; fall back to Jetson if quality is unacceptable.
2. **RAM ceiling (8GB).** Mitigation: strict sequential execution, quantized models, swap on NVMe.
3. **Turkish/multilingual accuracy** needs at least Whisper `small` (`base` may be weak). Test early.
4. **Long meetings** → chunk audio + map-reduce summarization to fit the LLM context window.

## Recommended first step

Phase 0 on your current Mac/PC: prove transcription + local summarization work before touching
diarization or any SBC. Smallest loop that de-risks the whole project.
