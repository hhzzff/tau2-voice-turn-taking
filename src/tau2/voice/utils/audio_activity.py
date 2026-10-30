"""Audio-level speech activity checks for recorded and streaming messages."""

from binascii import Error as BinasciiError
from typing import Any

import numpy as np

from tau2.data_model.audio import AudioData
from tau2.voice.utils.audio_preprocessing import convert_to_pcm16

AUDIO_RMS_THRESHOLD = 80.0


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
