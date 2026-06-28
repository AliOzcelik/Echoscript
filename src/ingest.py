# this file is for ingesting audio files

from __future__ import annotiations
from dataclass import dataclass
from pathlib import Path
import librosa
import numpy as np


# Lookup speed on a set is O(1) meanwhile O(n) for list and tuples. Irrelevant for 5 items but it is correct default for membership tests.
supported_extensions = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}



# a class for holding data. Every later file receives an AudioData knowing that it has .samples, .sample_rate, .source_path
@dataclass
class AudioData:
    samples: np.ndarray
    sample_rate: int
    source_path: Path

    @property
    def num_channels(self):
	return self.samples.shape[0]

    @property
    def num_samples(self):
        return self.samples.shape[1]

    @property
    def duration_seconds(self):
        return self.num_samples / self.sample_rate

# self.samples: [num_channels, num_samples]
# num_channels: 1 for mono, 2 for stereo (one row per audio channel)
# num_samples: the amplitude values over time (sample_rate x duration of them)


class IngestionError(Exception):
"""Raised when an audio file can't be found, read, or decoded"""

class Audio:

    def __init__(self, path, supported_extensions=supported_extensions):
	self.path = path # either str or Path
	self.supported_extensions = supported_extensions

    def validate_path(self):
	if not self.path.exists():
	    raise IngestionError(f"File not found: {self.path}")
	if not self.path.is_file():
	    raise IngestionError(f"Not a file: {self.path}")
	if self.path.suffix.lower() not in self.supported_extensions:
	    raise IngestionError(f"Unsupported format: '{self.path.suffix}'\nSupported formats: {', '.join(sorted(self.supported_extensions))}")

    def load_audio(self):
	path = Path(self.path).expanduser().resolve()
	self.validate_path()

	try:
	    samples, sample_rate = librosa.load(self.path, sr=None, mono=False)
	except Exception as exc:
	    raise IngestionError(f"Could not decode {self.path}: {exc}") from exc

	# sr=None means don'r resample, keep the file's native sample rate
	# mono=False keeps all original channels (stereo stays stereo)

	if samples.ndim == 1:
	    samples = samples[np.newaxis, :]

	samples = samples.astype(np.float32, copy=False)

	if samples.shape[1] == 0:
	    raise IngestionError(f"File contains no audio samples: {self.path}")

	audio_data = AudioData(samples = samples, sample_rate = int(sample_rate), source_path = self.path)

	return audio_data



# audio = load_date()
# loaded: audio.source_path.name
# sample_rate = audio.sample_rate (Hz)
# channels = audio.num_channels
# samples = audio.num_samples
# duration = {audio.duration_seconds:.2f}
# dtype = audio.samples.dtype
# range = audio.samples.min() and audio.samples.max()
