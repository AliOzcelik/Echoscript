from pathlib import Path
import librosa
import numpy as np
from src.ingest import AudioData


# whisper and the diarizer model are both trained on 16 kHz mono audio
target_sample_rate = 16000


class PreprocessError(Exception):
    """Raised when audio can't be converted into the model-ready format"""


# Takes the AudioData and standardizes it
# raw audio data turns into 1 channel (mono), 16kHz, float32, peak-normalized, optionally trimmed
class Preprocess:

    def __init__(self, audio, target_sample_rate=target_sample_rate, normalize=True, trim_silence=False, trim_db=30):
        self.audio = audio
        self.target_sample_rate = target_sample_rate
        self.normalize = normalize
        self.trim_silence = trim_silence
        self.trim_db = trim_db

    # (channels, n) -> (1, n)
    # average the channels into one mono row
    def to_mono(self, samples):
        if samples.shape[0] == 1:
            return samples # already mono

        return np.mean(samples, axis=0, keepdims=True)
        # keepdims=True keeps the result2D as (1, n) instead of collapsing to (n,)


    # Change the sample rate to self.target_sample_rate
    # 44100 -> 16000
    def resample(self, samples, original_sample_rate):

        if original_sample_rate == self.target_sample_rate:
            return samples, original_sample_rate

        resampled = librosa.resample(samples, orig_sr=original_sample_rate, target_sr=self.target_sample_rate)

        return resampled, self.target_sample_rate


    # Scale the whole signal so its loudest point sits   just under clipping
    def peak_normalize(self, samples, target_peak=0.97):
        peak = np.max(np.abs(samples))
        if peak == 0:
            return samples # fully silent, avoid divide-by-zero

        return samples * (target_peak / peak)


    def trim(self, samples):
        # librosa.effects.trim wants a 1D-signal
        mono_row = samples[0]
        trimmed, _ = librosa.effects.trim(mono_row, top_db=self.trim_db)
        return trimmed[np.newaxis, :] # back to (1, n)

    def preprocess(self):
        samples = self.audio.samples
        sample_rate = self.audio.sample_rate

        try:
            samples = self.to_mono(samples)
            samples, sample_rate = self.resample(samples, sample_rate)
            if self.normalize:
                samples = self.peak_normalize(samples)
            if self.trim_silence:
                samples = self.trim(samples)

        except Exception as exc:
            raise PreprocessError(f"Failed to preprocess: {self.audio.source_path}")

        samples = samples.astype(np.float32, copy=False)

        if samples.shape[1] == 0:
            raise PreprocessError(f"Nothing left after preprocessing: {self.audio.source_path}")

        audio_data = AudioData(samples=samples, sample_rate=sample_rate, source_path=self.audio.source_path)

        return audio_data
