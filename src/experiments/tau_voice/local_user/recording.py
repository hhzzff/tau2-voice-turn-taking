"""User observations and append-only experiment evidence."""

from __future__ import annotations

import asyncio
import hashlib
import json
import queue
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class Recorder:
    """Send journal writes to a worker; the audio loop never waits for disk IO."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.run_id = self.path.name
        self.events = []
        self.pending = queue.Queue(maxsize=10000)
        self.error = None
        self.closed = False
        self.worker = threading.Thread(target=self._write, daemon=True)
        self.worker.start()

    def _write(self):
        try:
            with (self.path / "events.jsonl").open("w", buffering=65536) as file:
                while (event := self.pending.get()) is not None:
                    if isinstance(event, tuple):
                        kind, name, data, rate = event
                        if kind == "audio":
                            with wave.open(str(self.path / name), "wb") as audio:
                                audio.setnchannels(1)
                                audio.setsampwidth(2)
                                audio.setframerate(rate)
                                audio.writeframes(data)
                        else:
                            (self.path / name).write_text(data)
                        continue
                    file.write(
                        json.dumps(event, ensure_ascii=False, default=str) + "\n"
                    )
        except Exception as exc:
            self.error = exc

    def emit(self, kind, **values):
        if self.closed or self.error:
            raise RuntimeError("Experiment journal is unavailable") from self.error
        stamp = time.perf_counter_ns()
        # Capture mutable values now; later agenda changes cannot rewrite evidence.
        event = json.loads(
            json.dumps(
                {"mono_ns": stamp, "kind": kind, **values},
                ensure_ascii=False,
                default=str,
            )
        )
        self.pending.put_nowait(event)
        self.events.append(event)
        return stamp

    def save_audio(self, name, pcm, rate=16000):
        self.pending.put_nowait(("audio", name, bytes(pcm), rate))

    def save_json(self, name, value):
        self.pending.put_nowait(
            ("json", name, json.dumps(value, ensure_ascii=False, indent=2) + "\n", None)
        )

    async def close(self):
        if self.error:
            raise RuntimeError("Experiment journal writer failed") from self.error
        if not self.closed:
            self.closed = True
            await asyncio.to_thread(self.pending.put, None, True, 5)
            await asyncio.to_thread(self.worker.join, 10)
        if self.worker.is_alive() or self.error:
            raise RuntimeError("Experiment journal failed to drain") from self.error


def now() -> int:
    return time.perf_counter_ns()


@dataclass(frozen=True)
class Frame:
    pcm: bytes
    sample_index: int
    received_mono_ns: int


@dataclass(frozen=True)
class Transcript:
    session_id: str
    event_id: str
    revision: int
    text: str
    is_final: bool
    audio_end_ms: float | None
    received_mono_ns: int


class Track:
    """Anchor each speech segment once, then preserve its PCM sample clock.

    Packet timestamps include scheduler and resampler batching jitter. Placing
    every packet at its wall-clock time inserts silence inside continuous audio.
    Timing measurements remain in the event log; genuine PCM silence is retained.
    """

    def __init__(self, origin, rate):
        self.origin, self.rate, self.pieces, self.end = origin, rate, [], 0
        self.segment = object()
        self.segments = []
        self.timing_pieces, self.timing_end = [], 0

    def add(self, at, pcm, *, segment):
        if len(pcm) % 2:
            raise ValueError("track requires aligned PCM16")
        offset = self.end
        if segment != self.segment:
            offset = max(offset, round((at - self.origin) * self.rate / 1e9))
            self.segment = segment
            self.segments.append(
                {"id": segment, "first_piece": len(self.pieces), "start_sample": offset}
            )
        self.pieces.append((offset, pcm))
        self.end = offset + len(pcm) // 2
        timed_offset = max(self.timing_end, round((at - self.origin) * self.rate / 1e9))
        self.timing_pieces.append((timed_offset, pcm))
        self.timing_end = timed_offset + len(pcm) // 2

    def render(self, length, *, timing=False):
        pieces, end = (
            (self.timing_pieces, self.timing_end) if timing else (self.pieces, self.end)
        )
        out = np.zeros(max(length, end), dtype="<i2")
        for offset, pcm in pieces:
            out[offset : offset + len(pcm) // 2] = np.frombuffer(pcm, dtype="<i2")
        return out

    def manifest(self):
        return {
            "sample_rate": self.rate,
            "segments": self.segments,
            "pieces": [
                {
                    "sample": offset,
                    "timing_sample": timed[0],
                    "samples": len(pcm) // 2,
                    "sha256": hashlib.sha256(pcm).hexdigest(),
                }
                for (offset, pcm), timed in zip(self.pieces, self.timing_pieces)
            ],
        }


class TurnRecorder:
    def __init__(self, session, turn):
        self.session, self.turn = session, turn

    def emit(self, kind, **values):
        feedback = self.turn.get("kind") == "backchannel"
        event_kind = kind.replace("user_", "backchannel_", 1) if feedback else kind
        tags = {"feedback_id": self.turn["id"]} if feedback else {}
        at = self.session.rec.emit(
            event_kind, turn=self.turn["index"], **tags, **values
        )
        self.turn["marks"].setdefault(kind, at)
        if kind in {"user_first_voice_tx", "user_audio_complete"} and hasattr(
            self.session, "own_audio_event"
        ):
            self.session.own_audio_event(kind, self.turn, at)
        if kind == "llm_complete":
            self.turn["text"] = values["text"]
        if (
            kind in {"tts_segment_submit", "tts_segment_complete"}
            and "output_start_sample" in values
        ):
            segments = self.turn.setdefault("speech_segments", [])
            entry = next(
                (s for s in segments if s["segment"] == values["segment"]), None
            )
            if entry is None:
                entry = {"segment": values["segment"], "output_end_sample": None}
                segments.append(entry)
            entry.update(
                {
                    k: values[k]
                    for k in ("output_start_sample", "output_end_sample", "text")
                    if k in values
                }
            )
        return at
