import tomllib
from dataclasses import dataclass
from pathlib import Path


SAMPLE_RATE = 16000 # 16 kHz mono

@dataclass(frozen=True)
class VadConfig:
    model: Path
    threshold: float
    neg_threshold: float
    min_speech_ms: int
    min_silence_ms: int
    pad_ms: int


@dataclass(frozen=True)
class SttConfig:
    server_url: str
    language: str
    beam_size: int
    prompt_chars: int
    word_time: str
    max_chunk_ms: int
    max_gap_ms: int
    timeout_s: float

    def __post_init__(self):
        if self.word_time not in ("dtw", "segment"):
            raise ValueError(f"stt.word_time must be 'dtw' or 'segment', got {self.word_time!r}")


@dataclass(frozen=True)
class DiarConfig:
    model: Path
    method: str
    threshold: float
    min_cluster_ms: int
    num_speakers: int

    def __post_init__(self):
        if self.method not in ("ahc", "spectral"):
            raise ValueError(f"diar.method must be 'ahc' or 'spectral', got {self.method!r}")


@dataclass(frozen=True)
class SummaryConfig:
    backend: str
    hailo_model: Path
    llama_url: str
    max_tokens: int
    temperature: float
    min_words: int
    timeout_s: float

    def __post_init__(self):
        if self.backend not in ("hailo", "llama"):
            raise ValueError(f"summary.backend must be 'hailo' or 'llama', got {self.backend!r}")


@dataclass(frozen=True)
class RecordConfig:
    device: str


@dataclass(frozen=True)
class Config:
    data_dir: Path
    record: RecordConfig
    vad: VadConfig
    stt: SttConfig
    diar: DiarConfig
    summary: SummaryConfig



def load_config(path=Path("config.toml")):
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return Config(
                data_dir = Path(raw["paths"]["data_dir"]),
                record = RecordConfig(**raw["record"]),
                vad = VadConfig(**{**raw["vad"], "model": Path(raw["vad"]["model"])}),
                stt = SttConfig(**raw["stt"]),
                diar = DiarConfig(**{**raw["diar"], "model": Path(raw["diar"]["model"])}),
                summary = SummaryConfig(**{**raw["summary"], "hailo_model": Path(raw["summary"]["hailo_model"]).expanduser()})
                )
