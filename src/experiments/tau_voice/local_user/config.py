"""Small, explicit configuration for local-user trials; no credential fields."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tau2.config import (
    DEFAULT_LOCAL_USER_FRAME_MS,
    DEFAULT_LOCAL_USER_LLM_URL,
    DEFAULT_LOCAL_USER_MODEL,
    DEFAULT_LOCAL_USER_SPEAKER,
    DEFAULT_LOCAL_USER_TTS_MODEL,
    DEFAULT_LOCAL_USER_TTS_URL,
    DEFAULT_PCM_SAMPLE_RATE,
)


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AudioConfig(Strict):
    sample_rate: Literal[16000] = DEFAULT_PCM_SAMPLE_RATE
    frame_ms: Literal[20] = DEFAULT_LOCAL_USER_FRAME_MS
    effective_rms: float = Field(0.005, gt=0, lt=1)


class ASRConfig(Strict):
    model: str = "GilgameshWind/X-ASR-zh-en/160ms"
    model_path: Path
    execution_provider: Literal["cpu", "cuda"] = "cpu"
    num_threads: int = Field(2, ge=1)
    endpoint_padding_ms: int = Field(640, ge=0, le=3000)


class LLMConfig(Strict):
    model: str = DEFAULT_LOCAL_USER_MODEL
    base_url: str = DEFAULT_LOCAL_USER_LLM_URL
    max_output_tokens: int = 384
    synthesis_unit: Literal["clause"] = "clause"
    extra_body: dict = Field(
        default_factory=lambda: {"chat_template_kwargs": {"enable_thinking": False}}
    )


class TTSConfig(Strict):
    model: str = DEFAULT_LOCAL_USER_TTS_MODEL
    url: str = DEFAULT_LOCAL_USER_TTS_URL
    speaker: str = DEFAULT_LOCAL_USER_SPEAKER
    reference_sha256: str = (
        "7df4da8fc59d89e6f9a60f2b87218d02ff996d81d463a95e9d492d4a401889f8"
    )
    seed: int = 42
    stream_chunk_tokens: int = Field(2, ge=1, le=24)
    trim_leading_silence: bool = True
    normalize_date_whitespace: bool = True
    pause_warning_ms: int = 3000


class FeedbackConfig(Strict):
    enabled: bool = True
    excluded_clip_ids: list[str] = Field(default_factory=list)
    retrieval_limit: int = 4
    recent_avoid: int = 3
    max_audio_ms: int = 2200
    min_interval_ms: int = 7000
    min_played_ms: int = 1200
    max_silence_ms: int = 160
    max_per_segment: int = 1


class AgendaConfig(Strict):
    enabled: Literal[True] = True
    mode: Literal["single"] = "single"


class LocalUserConfig(Strict):
    """Runtime options; model assets may be external, runtime source may not."""

    audio: AudioConfig = Field(default_factory=AudioConfig)
    asr: ASRConfig
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    backchannel: FeedbackConfig = Field(default_factory=FeedbackConfig)
    agenda: AgendaConfig = Field(default_factory=AgendaConfig)
    api_timeout_seconds: float = Field(30, gt=0)
