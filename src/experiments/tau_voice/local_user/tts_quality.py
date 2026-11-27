"""TTS-only date formatting and bounded, passive PCM pause measurements."""

from __future__ import annotations

import re

from .audio import rms

_HSPACE = r"[^\S\r\n]"
_DATE = re.compile(
    rf"(?:(?<=[\u3400-\u9fff]){_HSPACE}+)?(?<![0-9])"
    rf"(?:[0-9]{{4}}{_HSPACE}*年{_HSPACE}*)?"
    rf"[0-9]{{1,2}}{_HSPACE}*月{_HSPACE}*[0-9]{{1,2}}{_HSPACE}*[日号]"
)


def normalize_date_whitespace(text: str) -> str:
    """Join Chinese calendar dates without changing digits, units, or punctuation.

    Also join an immediately preceding Chinese word to its date. English word
    spacing, unrelated numbers, and line breaks remain untouched.
    """
    return _DATE.sub(lambda match: re.sub(_HSPACE + "+", "", match.group()), text)


class PauseMonitor:
    """Measure 20 ms RMS frames without holding back or modifying speech PCM.

    A warning during generation cannot yet distinguish an internal pause from
    the final tail. The completed summary makes that distinction after EOS.
    """

    def __init__(self, sample_rate=16000, threshold=0.005, warning_ms=3000):
        self.rate, self.threshold, self.warning_ms = sample_rate, threshold, warning_ms
        self.frame_bytes = sample_rate // 50 * 2
        self.pending = b""
        self.samples = self.quiet = self.leading = self.internal = 0
        self.has_voice = self.warned = False

    def _frame(self, pcm):
        self.samples += len(pcm) // 2
        if rms(pcm) < self.threshold:
            self.quiet += len(pcm) // 2
            if not self.warned and self.quiet * 1000 >= self.warning_ms * self.rate:
                self.warned = True
                return {
                    "phase": "after_voice" if self.has_voice else "leading",
                    "quiet_ms": self.quiet * 1000 / self.rate,
                    "audio_at_ms": self.samples * 1000 / self.rate,
                    "warning_ms": self.warning_ms,
                }
        else:
            if self.has_voice:
                self.internal = max(self.internal, self.quiet)
            else:
                self.leading = self.quiet
            self.has_voice, self.warned, self.quiet = True, False, 0
        return None

    def feed(self, pcm):
        """Measure newly received PCM frames and return any newly reached pause warnings."""
        self.pending += pcm
        warnings = []
        while len(self.pending) >= self.frame_bytes:
            frame, self.pending = (
                self.pending[: self.frame_bytes],
                self.pending[self.frame_bytes :],
            )
            if warning := self._frame(frame):
                warnings.append(warning)
        return warnings

    def finish(self):
        """Flush the final partial frame and distinguish leading, internal and trailing silence."""
        if self.pending:
            self._frame(self.pending)
            self.pending = b""
        leading = self.leading if self.has_voice else self.quiet
        tail = self.quiet if self.has_voice else 0

        def ms(samples):
            return round(samples * 1000 / self.rate, 3)

        return {
            "frame_ms": 20,
            "rms_threshold": self.threshold,
            "warning_ms": self.warning_ms,
            "has_voice": self.has_voice,
            "raw_audio_ms": ms(self.samples),
            "leading_quiet_ms": ms(leading),
            "max_internal_pause_ms": ms(self.internal),
            "trailing_quiet_ms": ms(tail),
            "long_pause": max(leading, self.internal, tail) * 1000
            >= self.warning_ms * self.rate,
        }
