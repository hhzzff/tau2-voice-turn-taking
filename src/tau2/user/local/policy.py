"""Audio-only floor policy. Semantic task completion is deliberately not an input."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class FloorState:
    at_ns: int
    own_turn: int = 0
    own_active: bool = False
    reserved: bool = False
    own_end_ns: int = 0
    reply_seen: bool = False
    voice_active: bool = False
    response_open: bool = False
    buffered_samples: int = 0
    last_voice_ns: int = 0
    playback_end_ns: int = 0
    normal_end: bool = False
    ambiguous_audio: bool = False


@dataclass(frozen=True)
class FloorPermission:
    kind: str
    reason: str
    epoch: int


class FloorController:
    def __init__(self, end_guard_ms=200, pause_ms=1200, first_reply_ms=8000):
        self.end_guard_ns = end_guard_ms * 1_000_000
        self.pause_ns = pause_ms * 1_000_000
        self.first_reply_ns = first_reply_ms * 1_000_000
        self.epoch = 0
        self._key = None

    def evaluate(self, s: FloorState) -> FloorPermission:
        if s.own_active or s.reserved:
            kind, reason = "BUSY", "own_audio_or_reservation"
        elif s.ambiguous_audio:
            kind, reason = "WAIT", "unbound_audio"
        elif s.voice_active or s.buffered_samples:
            kind, reason = "LISTEN", "actual_voice_or_buffer"
        elif not s.own_turn:
            kind, reason = "RESPOND", "initial_opening"
        elif not s.reply_seen:
            recovering = s.own_end_ns and s.at_ns - s.own_end_ns >= self.first_reply_ns
            kind, reason = (
                ("RECOVER", "no_first_reply")
                if recovering
                else ("WAIT", "await_first_reply")
            )
        elif s.normal_end and not s.response_open and s.playback_end_ns:
            ready = (
                s.at_ns - max(s.playback_end_ns, s.last_voice_ns) >= self.end_guard_ns
            )
            kind, reason = (
                ("RESPOND", "playback_complete") if ready else ("WAIT", "end_guard")
            )
        elif s.last_voice_ns and s.at_ns - s.last_voice_ns >= self.pause_ns:
            kind, reason = "RESPOND", "pause_inferred_handoff"
        else:
            kind, reason = "WAIT", "short_pause"
        key = (s.own_turn, kind, reason)
        if key != self._key:
            self.epoch += 1
            self._key = key
        return FloorPermission(kind, reason, self.epoch)


def effective_action(
    proposed: str, permission: FloorPermission, *, observation=False
) -> str:
    if observation:
        return "WAIT"
    if permission.kind in {"RESPOND", "RECOVER"}:
        return "RESPOND"
    if permission.kind == "LISTEN" and proposed in {"INTERRUPT", "BACKCHANNEL"}:
        return proposed
    return "WAIT"


def evidence_still_valid(quote: str, previous: str, current: str) -> bool:
    """A quotation is not stable when a numeric/negation tail has been extended."""

    def clean(text):
        return re.sub(r"[\s，,。.!！?？；;：:]", "", text)

    quote, previous, current = map(clean, (quote, previous, current))
    if not quote or quote not in previous or quote not in current:
        return False
    end = current.rfind(quote) + len(quote)
    numeric = "0123456789零〇一二两三四五六七八九十百千万亿点"
    if end < len(current) and quote[-1] in numeric and current[end] in numeric:
        return False
    if quote.endswith(("不", "没", "不是", "没有")):
        return False
    return True
