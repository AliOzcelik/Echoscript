from dataclasses import dataclass
from pathlib import Path
import numpy as np
import sherpa_onnx
from src.ingest import AudioData


project_root = Path(__file__).resolve().parent.parent
# print(project_root)

segmentation_model = f"{project_root}/models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
embedding_model = f"{project_root}/models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"

class DiarizationError(Exception):
    """Raised when the diarization models can't load or can't run"""


# one labeled stretch of speech
@dataclass
class SpeakerTurn:
    start: float # seconds
    end: float
    speaker: str

    @property
    def duration(self):
        return self.end - self.start



class Diarizer:

    def __init__(self, segmentation_model=segmentation_model, embedding_model=embedding_model, num_speakers=None, clustering_treshold=0.5, num_threads=4):

        # raspberry pi 5 has 4 cores, use all of them in here
        # clustering treshold is used when num_speakers = None

        self.segmentation_model = segmentation_model
        self.embedding_model = embedding_model
        self.check_models()

        if num_speakers is None:
            self.num_clusters = -1
        else:
            self.num_clusters = num_speakers

        pyannote_config = sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model = self.segmentation_model)
        segmentation_config = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(pyannote_config, num_threads=num_threads, provider="cpu")
        embedding_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=self.embedding_model, num_threads=num_threads, provider="cpu")
        clustering_config = sherpa_onnx.FastClusteringConfig(num_clusters=self.num_clusters, threshold=clustering_treshold)

        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(segmentation=segmentation_config, embedding=embedding_config, clustering=clustering_config, min_duration_on=0.3, min_duration_off=0.5)

        # min_duration_on: ignores speech bursts shorter than this (s)
        # min_duration_off: ignores silences shorter than this (s)

        if not config.validate():
            raise DiarizationError("Invalid diarization config (check model paths).")

        self.speech_diarization_model = sherpa_onnx.OfflineSpeakerDiarization(config)

    def check_models(self):
        for path in (self.segmentation_model, self.embedding_model):
            if not Path(path).is_file():
                raise DiarizationError(f"Model not found: {path}")


    def diarize(self, audio):

        if audio.sample_rate != self.speech_diarization_model.sample_rate:
            raise DiarizationError(f"Expected {self.speech_diarization_model.sample_rate} Hz audio but got {audio.sample_rate} Hz. Run process.py first")

        # (1, n) -> (n,) 1D, guaranteed contiguous float32 for the C backend
        samples = np.ascontiguousarray(audio.samples[0], dtype=np.float32)

        try:
            results = self.speech_diarization_model(samples).sort_by_start_time()
        except Exception as exc:
            raise DiarizationError(f"Diarization failed: {exc}")


        # sherpa_onnx gives integer speaker ids
        # format them like pyannote's labels so downstream output looks the same with either engine
        turns = []
        for seg in results:
            speaker_segment = SpeakerTurn(start=seg.start, end=seg.end, speaker=f"Speaker_{seg.speaker:02d}")
            turns.append(speaker_segment)

        return turns
