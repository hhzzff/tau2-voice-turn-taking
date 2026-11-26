"""Interfaces between the scene executor and tau's streaming participant adapter.

These contracts contain audio, playback lifecycle and delivery receipts. Scene
stages, prompts and semantic judgments belong to the executor, not the clocks.
"""

import asyncio
from typing import Any, Protocol, TypedDict

from .observation import HeardItem, PlaybackObservation


class OutputPacket(TypedDict):
    """One offered frame; the mutable turn ledger is committed only by receipt."""

    sequence: int
    pcm: bytes
    source_samples: int
    voice: bool
    turn: dict[str, Any] | None
    ending: dict[str, Any] | None


class LocalRuntime(Protocol):
    """The experiment runtime owned by a streaming participant and its runner."""

    playback: PlaybackObservation
    tasks: list[asyncio.Task]
    control_task: asyncio.Task
    sent_turn: int
    stop: bool

    def poll_output(self) -> OutputPacket:
        """Offer one ready frame without awaiting inference or committing facts."""
        ...

    def acknowledge_delivery(self, sequence: int, sent_ns: int) -> None:
        """Commit the offered frame after successful client transport submission."""
        ...

    def begin_agent(self, item: HeardItem) -> None:
        """Start observation of an item at its first actual playback frame."""
        ...

    def agent_frame(self, item: HeardItem, pcm: bytes, at: int) -> None:
        """Observe PCM already played at monotonic timestamp at (nanoseconds)."""
        ...

    def end_agent(self, item: HeardItem, reason: str) -> None:
        """Queue final observation of the played prefix, including interruptions."""
        ...

    async def close(self) -> None:
        """Stop generation and join ASR workers before releasing native resources."""
        ...


class VoiceTarget(Protocol):
    """A target whose receive, playback and outgoing transport run independently."""

    async def receive(self) -> None:
        """Receive until cancelled; an unexpected connection end is an error."""
        ...

    def play_frame(self, at: int) -> None:
        """Release at most one frame to the participant at the playback clock."""
        ...

    async def send_packet(self, packet: OutputPacket) -> int:
        """Return the submission timestamp only after successful transport send."""
        ...
