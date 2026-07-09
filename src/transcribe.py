from dataclasses import dataclass, field
import numpy as np
from faster_whisper import WhisperModel

from src.ingest import AudioData



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

    def __init__(self, model_size="small", device="cpu", compute_type="int8", cpu_threads=4, language=None, word_timestamps=True, vad_filter=True):

        self.language = language
        self.word_timestamps = word_timestamps
        self.vad_filter = vad_filter

        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type, cpu_threads=cpu_threads)
        except Exception as exc:
            raise TranscribeError(f"Could not load whisper model: '{model_size}': '{exc}'")

    def transcribe(self, audio):
        samples = np.ascontiguousarray(audio.samples[0], dtype=np.float32)

        try:
            segment_iter, info = self.model.transcribe(samples, language=self.language, word_timestamps=self.word_timestamps, vad_filter=self.vad_filter)
            segments = []
            for seg in segment_iter:
                words = []
                for w in seg.words:
                    word = Word(start=w.start, end=w.end, text=w.word.strip())
                    words.append(word)
                segment = TranscriptSegment(start=seg.start, end=seg.end, text=seg.text.strip(), words=words)
                segments.append(segment)

        except Exception as exc:
                raise TranscribeError(f"Transcription failed: {exc}")

        transcript = Transcript(segments=segments, language=info.language)

        return transcript
