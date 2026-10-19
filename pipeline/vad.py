from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

from pipeline.config import VadConfig, SAMPLE_RATE



FRAME = 512 # samples per model call (32ms at 16 kHz)
CONTEXT = 64 # Silero v5+ expects the last 64 samples of the previous call prepended


@dataclass(frozen=True)
class SpeechRegion:
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self):
        return self.end_ms - self.start_ms

    def to_dict(self):
        return asdict(self)



class SileroModel:
    """One ONNX call per 512-sample frame; carries the recurrent state between calls."""

    def __init__(self, model_path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1 # VAD is tiny; one thread, no contention with STT
        opts.inter_op_num_threads = 1

        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.sample_rate = np.array(SAMPLE_RATE, dtype=np.int64)
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, CONTEXT), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self.context, frame[np.newaxis, :]], axis=1) # (1, 576)
        prob, self.state = self.session.run(None, {"input": x, "state": self.state, "sr": self.sample_rate})
        self.context = x[:, -CONTEXT:]
        return float(prob[0, 0])


class StreamingVAD:

    def __init__(self, config: VadConfig):
        self.model = SileroModel(config.model)

        per_ms = SAMPLE_RATE // 1000
        self.threshold = config.threshold
        self.neg_threshold = config.neg_threshold

        self.min_speech = config.min_speech_ms * per_ms
        self.min_silence = config.min_silence_ms * per_ms
        self.pad = config.pad_ms * per_ms

        self.buffer = np.zeros(0, dtype=np.float32)     # leftover samples (<1 frame)
        self.pos = 0                                 # first sample of the next frame
        self.start = None                           # speech start while inside speech
        self.temp_end = None                        # where speech 'may' have ended
        self.last_end = 0                           # end of the last emitted region (padded)


    @property
    def open_start_ms(self) -> int | None:
        """Start (padded) of the speech going on now, not closed yet; None in silence."""
        if self.start is None:
            return None
        return max(self.start - self.pad, self.last_end) * 1000 // SAMPLE_RATE


    def close(self, end, limit=None) -> SpeechRegion | None:
        start = self.start
        self.start = None
        self.temp_end = None

        if (end - start) < self.min_speech:
            return None                 # too short (click, cough, breath)

        start = max(start - self.pad, self.last_end)            # padding never overlaps the previous region
        end = end + self.pad

        if limit is not None:
            end = min(end, limit)

        self.last_end = end

        return SpeechRegion(start * 1000 // SAMPLE_RATE, end * 1000 // SAMPLE_RATE)


    def step(self, prob, t) -> SpeechRegion | None:

        if prob >= self.threshold:
            self.temp_end = None     # speech (again): cancel a pending end
            if self.start is None:
                self.start = t
            return None

        if self.start is None or prob >= self.neg_threshold:
            return None                 # silence while idle, or unsure zone while speaking

        if self.temp_end is None:
            self.temp_end = t               # speech may have ended here

        if (t + FRAME - self.temp_end) < self.min_silence:
            return None                 # ... but the silence is not long enough yet

        return self.close(self.temp_end)



    def feed(self, samples: np.ndarray) -> list[SpeechRegion]:
        """Samples: float32, 16kHz, mono, range [-1, 1], any length"""

        self.buffer = np.concatenate([self.buffer, samples])
        n_frames = len(self.buffer) // FRAME
        regions = []

        for i in range(n_frames):
            prob = self.model(self.buffer[i * FRAME:(i+1) * FRAME])
            region = self.step(prob, self.pos)

            if region is not None:
                regions.append(region)

            self.pos += FRAME

        self.buffer = self.buffer[n_frames * FRAME:]
        return regions


    def flush(self) -> list[SpeechRegion]:
        """Call once at the end of the audio: closes a region that is still open"""
        if self.start is None:
            return []
        total = self.pos + len(self.buffer)
        end = self.temp_end if self.temp_end is not None else total
        region = self.close(end, limit=total)
        return [region] if region is not None else []



def vad_file(path, config, block_s=10.0) -> list[SpeechRegion]:
    vad = StreamingVAD(config)
    regions = [] # list[SpeechRegion]

    with sf.SoundFile(str(path)) as f:
        if f.samplerate != SAMPLE_RATE or f.channels != 1:
            raise ValueError(f"{path}: expected 16 kHz mono, got {f.samplerate} Hz x{f.channels}")

        for block in f.blocks(blocksize = int(block_s * SAMPLE_RATE), dtype="float32"):
            regions += vad.feed(block)

    return regions + vad.flush()
