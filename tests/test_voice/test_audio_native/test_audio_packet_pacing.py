"""Packet pacing must not accumulate timer rounding or network send time."""

import asyncio
import math
from types import SimpleNamespace

import pytest

from tau2.voice.audio_native.gemini.discrete_time_adapter import (
    DiscreteTimeGeminiAdapter,
)


@pytest.mark.parametrize("send_seconds", [0.0, 0.003, 0.035])
@pytest.mark.parametrize("tick_ms", [100, 200])
def test_voip_pacing_compensates_for_coarse_timer_and_send_time(
    monkeypatch, send_seconds, tick_ms
):
    """Model Windows' coarse timer without network calls or real waiting."""
    adapter = DiscreteTimeGeminiAdapter(tick_ms, send_audio_instant=False)
    audio = bytes(i % 256 for i in range(32 * tick_ms))
    packet_count = tick_ms // 20
    now, sent, waits = 0.0, [], []

    async def send(packet):
        nonlocal now
        sent.append(packet)
        now += send_seconds

    async def coarse_sleep(seconds):
        nonlocal now
        waits.append(seconds)
        now += math.ceil(seconds / 0.015625) * 0.015625

    async def run():
        with monkeypatch.context() as patch:
            patch.setattr(
                asyncio, "get_running_loop", lambda: SimpleNamespace(time=lambda: now)
            )
            patch.setattr(asyncio, "sleep", coarse_sleep)
            await adapter._send_audio_chunked(audio, send, adapter._chunk_size)

    asyncio.run(run())
    assert b"".join(sent) == audio
    assert len(sent) == packet_count
    if send_seconds <= 0.003:
        # Allow one timer quantum for the whole input, not for every packet.
        assert tick_ms / 1000 <= now < tick_ms / 1000 + 0.016
    else:
        assert now == pytest.approx(packet_count * send_seconds)
        assert waits == []


@pytest.mark.parametrize("audio", [b"", b"packet" * 200])
def test_instant_audio_does_not_add_packet_waits(monkeypatch, audio):
    adapter = DiscreteTimeGeminiAdapter(200, send_audio_instant=True)
    sent = []

    async def send(packet):
        sent.append(packet)

    async def unexpected_sleep(seconds):
        pytest.fail("Instant sending must not wait between packets")

    monkeypatch.setattr(asyncio, "sleep", unexpected_sleep)
    asyncio.run(adapter._send_audio_chunked(audio, send, adapter._chunk_size))
    assert sent == ([audio] if audio else [])
