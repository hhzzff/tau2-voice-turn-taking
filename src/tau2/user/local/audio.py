"""PCM measurements and WAV output shared by the local user and driver."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def wav_writer(path: Path, rate: int = 16000):
    wav = wave.open(str(path), "wb")
    wav.setparams((1, 2, rate, 0, "NONE", "not compressed"))
    return wav


def rms(pcm: bytes) -> float:
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768
    return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0


def first_effective(pcm: bytes, threshold: float) -> int | None:
    indices = np.flatnonzero(
        np.abs(np.frombuffer(pcm, dtype="<i2").astype(np.int32)) >= threshold * 32768
    )
    return int(indices[0]) if len(indices) else None
