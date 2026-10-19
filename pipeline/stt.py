"""Stt: pack VAD regions into <= 30s chunks, transcribe each on whisper-server, return timed words."""

import io
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

from pipeline.config import SAMPLE_RATE, SttConfig
from pipeline.vad import SpeechRegion


SAMPLES_PER_MS = SAMPLE_RATE // 1000
QUIET_SEARCH_MS = 5000  # speech without a pause is cut at the quietest point of a chunk's last 5 s


@dataclass(frozen=True)
class Chunk:
    index: int
    start_ms: int
    end_ms: int
    reason: str = ""    # why the chunk ends: "gap" (long silence), "full" (cut at a short silence),
                        # "forced" (30 s of speech without a pause: cut mid-speech), "end"

    @property
    def duration_ms(self):
        return self.end_ms - self.start_ms

    def to_dict(self):
        return asdict(self)



@dataclass(frozen=True)
class Word:
    start_ms: int
    end_ms: int
    text: str
    prob: float # mean token probability: low = Whisper was unsure

    def to_dict(self):
        return asdict(self)




# Packing

def quiet_point(wav, lo_ms, hi_ms) -> int:
    """The middle of the quietest 100 ms in [lo_ms, hi_ms]: a cut there splits no word."""
    audio, _ = sf.read(wav, start=lo_ms * SAMPLES_PER_MS, stop=hi_ms * SAMPLES_PER_MS, dtype="float32")
    n = len(audio) // (10 * SAMPLES_PER_MS)             # 10 ms frames
    if n < 10:
        return hi_ms
    energy = (audio[:n * 10 * SAMPLES_PER_MS].reshape(n, -1) ** 2).mean(axis=1)
    smooth = np.convolve(energy, np.ones(10) / 10, mode="valid")    # 100 ms windows
    return lo_ms + (int(np.argmin(smooth)) + 5) * 10


def split_long(region, max_ms, wav=None):
    """A region longer than one chunk (30s without a pause) is cut into parts <= max_ms.
    With the wav: each cut at the quietest point of the last 5 s, a breath between words
    (whisper drops the last seconds of a chunk cut mid-word). Without: equal parts."""
    if wav is None:
        n = -(-region.duration_ms // max_ms) # ceiling division
        step = region.duration_ms / n
        return [(region.start_ms + round(i * step), region.start_ms + round((i+1) * step)) for i in range(n)]
    parts = []
    start = region.start_ms
    while region.end_ms - start > max_ms:
        cut = quiet_point(wav, start + max_ms - QUIET_SEARCH_MS, start + max_ms)
        parts.append((start, cut))
        start = cut
    parts.append((start, region.end_ms))
    return parts


def pack(regions, config, wav=None):
    """Fill each 30s encoder window as much as possible; cut only at silences.
    wav: cut speech without a pause at quiet points (split_long), else into equal parts"""

    spans = []
    start = None
    end = None

    for region in regions:
        for k, (r_start, r_end) in enumerate(split_long(region, config.max_chunk_ms, wav)):

            if start is None:
                start = r_start
                end = r_end

            elif (r_start - end) > config.max_gap_ms or (r_end - start) > config.max_chunk_ms:
                # long silence, or the chunk would get too long (k > 0: inside one region, no pause)
                reason = "gap" if (r_start - end) > config.max_gap_ms else "forced" if k else "full"
                spans.append((start, end, reason))
                start = r_start
                end = r_end

            else:
                end = r_end                 # short silence: stays inside the chunk

    if start is not None:
        spans.append((start, end, "end"))

    return [Chunk(i, s, e, reason) for i, (s, e, reason) in enumerate(spans)]



# Transcription

def inference(client, audio, prompt, config, audio_ctx=0):
    """audio_ctx > 0: encode only that many encoder frames (50 per second) instead of the whole
    30 s window. Faster for short chunks, but the model was trained on full windows."""
    wav = io.BytesIO()
    sf.write(wav, audio,  SAMPLE_RATE, format="WAV", subtype="PCM_16")

    response = client.post(
        "/inference",
        files = {"file": ("chunk.wav", wav.getvalue(), "audio/wav")},
        data = {
            "language": config.language,
            "response_format": "verbose_json",
            "beam_size": str(config.beam_size),
            "temperature": "0.0",
            "max_len": "1",              # one segment per word
            "split_on_word": "true",     # ...split at word boundaries (UTF-8 safe text)
            # one encoder pass per chunk (~25 s on the Pi): otherwise whisper encodes a whole new
            # window for any tail after its last timestamp. Needs the patched whisper-server.
            "single_segment": "true",
            # verbose_json otherwise runs language detection after transcribing: one more encoder
            # pass (~25 s) per chunk, only to fill language_probabilities, which nothing reads
            "no_language_probabilities": "true",
            "prompt": prompt,
            "audio_ctx": str(audio_ctx),
        }
    )

    response.raise_for_status()
    data = response.json()

    if "error" in data:
        raise RuntimeError(f"whisper-server: {data['error']}")

    return data


def parse(data, word_time, chunk_ms):
    """Server response -> words with chunk-local times. Text from segments, time from tokens"""
    words = []

    for seg in data.get("segments", []):
        text = seg["text"].strip()
        if not text:
            continue

        start = round(seg["start"]  * 1000)
        end = round(seg["end"] * 1000)
        tokens = seg.get("words", []) # these are tokens, not words

        if word_time == "dtw":
            dtw = [t["t_dtw"] for t in tokens if t.get("t_dtw", -1) >= 0]
            if dtw:
                # DTW places the word, the segment gives its length. The two clocks are 0.2-0.5 s
                # apart, so a DTW start with a segment end often ended a word before it started.
                end = dtw[0] * 10 + (end - start)       # t_dtw is in centiseconds
                start = dtw[0] * 10

        if start >= chunk_ms:
            continue                        # timestamp past the audio: a hallucination

        probs = [t["probability"] for t in tokens]
        prob = sum(probs) / len(probs) if probs else 0.0
        words.append(Word(start, min(end, chunk_ms), text, round(prob, 3)))

    return tidy(words)


def tidy(words):
    """Times in order, no overlaps: each word ends at the latest when the next one starts."""
    out = []

    for i, w in enumerate(words):
        start = max(w.start_ms, out[-1].start_ms if out else 0)
        end = w.end_ms

        if i + 1 < len(words):
            end = min(end, words[i + 1].start_ms)

        out.append(Word(start, max(end, start), w.text, w.prob))

    return out


def tail(text, n):
    """Last ~n characters of the text, starting at a word boundary."""
    text = text.strip()
    if len(text) <= n:
        return text

    cut = text[-n:]
    return cut[cut.find(" ") + 1:]


def transcribe_chunk(client, wav, chunk, prompt, config):
    """One chunk on whisper-server -> its words in recording time, and the seconds it took."""
    # read only this chunk from disk: memory stays flat even for hours of audio
    audio, _ = sf.read(wav, start = chunk.start_ms*SAMPLES_PER_MS, stop = chunk.end_ms*SAMPLES_PER_MS, dtype="float32")

    t0 = time.perf_counter()
    data = inference(client, audio, prompt, config)
    took = time.perf_counter() - t0

    chunk_ms = len(audio) * 1000 // SAMPLE_RATE
    words = [Word(w.start_ms + chunk.start_ms, w.end_ms + chunk.start_ms, w.text, w.prob) for w in parse(data, config.word_time, chunk_ms)]
    return words, took


def transcribe(wav, chunks, config):
    words = []
    prompt = ""

    with httpx.Client(base_url = config.server_url, timeout = config.timeout_s) as client:
        for c in chunks:
            new, took = transcribe_chunk(client, wav, c, prompt, config)
            words += new
            prompt = tail(f"{prompt} {' '.join(w.text for w in new)}", config.prompt_chars)

            print(f"chunk {c.index + 1}/{len(chunks)} | {c.duration_ms / 1000:4.1f} s | {len(new):3d} words | {took:5.1f} s | RTF {took * 1000 / c.duration_ms:.2f}")

    return words
