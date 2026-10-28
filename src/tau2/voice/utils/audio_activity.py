"""Audio activity shared by native-demo triggers and exported evidence."""

import numpy as np

from tau2.config import OPEN_MODEL_DEMO_ACTIVITY_FRAME_MS, OPEN_MODEL_DEMO_ACTIVITY_RMS


def active_pcm16_frames(samples, sample_rate=8000):
    """Return one energy decision per complete frame of mono PCM16 samples."""
    width = sample_rate * OPEN_MODEL_DEMO_ACTIVITY_FRAME_MS // 1000
    values = np.asarray(samples, dtype=float)
    frames = values[: len(values) // width * width].reshape(-1, width)
    return np.sqrt(np.mean(frames * frames, axis=1)) > OPEN_MODEL_DEMO_ACTIVITY_RMS
