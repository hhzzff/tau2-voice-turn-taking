"""Locate a requested SSML break in the actual synthesized PCM waveform."""

import numpy as np

from tau2.voice.utils.audio_activity import AUDIO_RMS_THRESHOLD


def locate_synthesized_pause(
    samples: np.ndarray, sample_rate: int, requested_seconds: float
) -> dict:
    """Return a unique matching quiet gap, with 10 ms resolution.

    This measures an acoustic gap, not a forced alignment of words. Ambiguous
    or absent matches must not be used as a known event boundary.
    """
    width = max(1, round(sample_rate * 0.01))
    values = np.asarray(samples, dtype=np.float64)
    frames = np.pad(values, (0, (-len(values)) % width)).reshape(-1, width)
    quiet = np.sqrt(np.mean(frames**2, axis=1)) < AUDIO_RMS_THRESHOLD
    edges = np.diff(np.r_[False, quiet, False].astype(int))
    candidates = []
    resolution = width / sample_rate
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        duration = (end - start) * resolution
        # Exclude leading silence, accept an SSML break at the end for CTC.
        if start and abs(duration - requested_seconds) <= 0.15:
            candidates.append(
                {
                    "start_seconds": round(start * resolution, 6),
                    "end_seconds": round(
                        min(end * resolution, len(values) / sample_rate), 6
                    ),
                    "duration_seconds": round(duration, 6),
                }
            )
    return {
        "status": "measured"
        if len(candidates) == 1
        else "ambiguous"
        if candidates
        else "not_found",
        "method": "unique_quiet_gap_near_requested_duration",
        "requested_seconds": requested_seconds,
        "resolution_seconds": resolution,
        "rms_threshold": AUDIO_RMS_THRESHOLD,
        "candidates": candidates,
        **(candidates[0] if len(candidates) == 1 else {}),
    }
