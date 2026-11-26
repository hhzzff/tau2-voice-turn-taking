"""Audio activity for streaming messages and native-demo frame evidence."""

from binascii import Error as BinasciiError
from typing import Any

import numpy as np

from tau2.config import OPEN_MODEL_DEMO_ACTIVITY_FRAME_MS, OPEN_MODEL_DEMO_ACTIVITY_RMS
from tau2.data_model.audio import AudioData
from tau2.voice.utils.audio_preprocessing import convert_to_pcm16

AUDIO_RMS_THRESHOLD = 80.0


def active_pcm16_frames(samples, sample_rate=8000):
    """Return one energy decision per complete frame of mono PCM16 samples."""
    width = sample_rate * OPEN_MODEL_DEMO_ACTIVITY_FRAME_MS // 1000
    values = np.asarray(samples, dtype=float)
    frames = values[: len(values) // width * width].reshape(-1, width)
    return np.sqrt(np.mean(frames * frames, axis=1)) > OPEN_MODEL_DEMO_ACTIVITY_RMS


def audio_activity(chunk: Any, threshold: float = AUDIO_RMS_THRESHOLD) -> bool | None:
    """Return audio-RMS activity, or None when the chunk has no inspectable audio."""
    audio_content = getattr(chunk, "audio_content", None)
    audio_format = getattr(chunk, "audio_format", None)
    if not audio_content or audio_format is None:
        return None

    try:
        audio = AudioData(data=chunk.get_audio_bytes(), format=audio_format)
        pcm16 = convert_to_pcm16(audio).data
    except (AttributeError, BinasciiError, TypeError, ValueError):
        return None

    samples = np.frombuffer(pcm16, dtype="<i2").astype(np.float64)
    if samples.size == 0:
        return False
    rms = float(np.sqrt(np.mean(np.square(samples))))
    return rms >= threshold


def is_audible_chunk(chunk: Any, threshold: float = AUDIO_RMS_THRESHOLD) -> bool:
    """Prefer measured audio activity and fall back to the provider speech flag."""
    if chunk is None:
        return False
    measured = audio_activity(chunk, threshold)
    if measured is not None:
        return measured
    return bool(getattr(chunk, "contains_speech", False))
