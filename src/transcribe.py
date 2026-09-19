from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from hailo_platform.genai import Speech2Text, Speech2TextTask

from src.ingest import AudioData


whisper_path = Path.home() / "hailo-rpi5-examples/llm_models/Whisper-Small.hef"


class TranscribeError(Exception):
    """Raised when the speech-to-text model can't load or can't run"""



# One word and exactly when it si spoken
# Timestamps are what lets merge.py decide which speaker said which word
@dataclass
class Word:
    start: float
    end: float
    text: str


# A chunk of speech (sentence) with its words in it
@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory = list) # empty if word transcipt is disabled


# The whole transcription result
@dataclass
class Transcript:
    segments: list[TranscriptSegment]
    language: str

    @property
    def text(self):
        return " ".join(seg.text for seg in self.segments)
        # full transcript as one string






class Transcribe:

    # vdevice is shared with the LLM and created once in main.py
    # timeout_ms is raised because VAD segments can be up to 90 s long
    def __init__(self, vdevice, model_path=whisper_path, language=None, timeout_ms=120_000):

        self.language = language
        self.timeout_ms = timeout_ms

        try:
            self.model = Speech2Text(vdevice, str(model_path))
        except Exception as exc:
            raise TranscribeError(f"Could not load whisper model: '{model_path}': '{exc}'")

    # language overrides the constructor default for this call, None means auto-detect
    def transcribe(self, audio, language=None):
        language = language or self.language
        samples = np.ascontiguousarray(audio.samples[0], dtype=np.float32)

        try:
            hailo_segments = self.model.generate_all_segments(samples, task=Speech2TextTask.TRANSCRIBE, language=language, timeout_ms=self.timeout_ms)
        except Exception as exc:
            raise TranscribeError(f"Transcription failed: {exc}")

        # Hailo gives segment timestamps only, so words stays empty
        # merge.py then assigns one speaker per whole segment
        segments = [TranscriptSegment(start=s.start_sec, end=s.end_sec, text=s.text.strip()) for s in hailo_segments]

        # Hailo doesn't report the detected language, so return the requested one
        return Transcript(segments=segments, language=language or "unknown")

    # must be called before the VDevice closes
    def close(self):
        self.model.release()
