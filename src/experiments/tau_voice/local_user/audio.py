"""PCM measurements shared by the local user and driver."""

from __future__ import annotations

import numpy as np


def rms(pcm: bytes) -> float:
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768
    return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0


def first_effective(pcm: bytes, threshold: float) -> int | None:
    indices = np.flatnonzero(
        np.abs(np.frombuffer(pcm, dtype="<i2").astype(np.int32)) >= threshold * 32768
    )
    return int(indices[0]) if len(indices) else None
