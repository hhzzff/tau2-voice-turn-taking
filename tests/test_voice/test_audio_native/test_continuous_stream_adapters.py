"""Offline regressions for continuously streamed Live and Seeduplex audio."""

import asyncio
import base64
from contextlib import suppress
from functools import wraps
from unittest.mock import AsyncMock

import pytest

from tau2.voice.audio_native.doubao.discrete_time_adapter import (
    DiscreteTimeDoubaoAdapter,
)
from tau2.voice.audio_native.openai.live_discrete_time_adapter import (
    DiscreteTimeOpenAILiveAdapter,
)
from tau2.voice.audio_native.tick_result import TickResult


def run_async(function):
    """Run a coroutine test without requiring an optional pytest plugin."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return wrapped


@pytest.mark.parametrize(
    "adapter_class,audio_event,text_event",
    [
        (
            DiscreteTimeDoubaoAdapter,
            "response.output_audio.delta",
            "response.output_text.delta",
        ),
        (
            DiscreteTimeOpenAILiveAdapter,
            "session.output_audio.delta",
            "session.output_transcript.delta",
        ),
    ],
)
@run_async
async def test_paced_tick_drains_received_audio_after_deadline(
    adapter_class, audio_event, text_event
):
    """Audio received while sending must play even if pacing used the tick."""
    adapter = adapter_class(20)
    adapter._events = asyncio.Queue()
    audio = b"\x01\x02" * 2400
    await adapter._events.put(
        {"type": audio_event, "delta": base64.b64encode(audio).decode()}
    )
    await adapter._events.put({"type": text_event, "delta": "Hello there."})

    async def slow_send(*args):
        await asyncio.sleep(0.025)

    adapter._send_audio_chunked = slow_send
    result = await adapter._async_run_tick(b"\xff" * 160, 0)
    assert adapter._events.empty()
    assert result.agent_audio_bytes == adapter.bytes_per_tick
    assert adapter._buffered_agent_audio
    assert len(result.get_played_agent_audio()) == 160
    assert result.proportional_transcript


def test_doubao_barge_in_discards_old_playback():
    adapter = DiscreteTimeDoubaoAdapter(200)
    adapter._buffered_agent_audio = [(b"old", "previous")]
    result = TickResult(
        tick_number=0, audio_sent_bytes=1600, audio_sent_duration_ms=200
    )
    result.agent_audio_chunks = [(b"queued", "previous")]
    adapter._process_event(
        result, {"type": "conversation.item.input_audio_transcription.started"}
    )
    assert result.was_truncated
    assert result.truncated_audio_bytes == 9
    assert result.vad_events == ["speech_started"]
    assert not result.agent_audio_chunks
    assert not adapter._buffered_agent_audio


@run_async
async def test_live_continuous_silence_preserves_timing_without_holding_floor():
    adapter = DiscreteTimeOpenAILiveAdapter(20)
    adapter._events = asyncio.Queue()
    adapter._send_audio_chunked = AsyncMock()
    await adapter._events.put(
        {
            "type": "session.output_audio.delta",
            "delta": base64.b64encode(b"\xff" * 320).decode(),
        }
    )
    result = await adapter._async_run_tick(b"\xff" * 160, 0)
    assert result.get_played_agent_audio() == b"\xff" * 160
    assert not result.item_ids
    assert adapter._buffered_agent_audio == [(b"\xff" * 160, None)]


def test_doubao_credentials_select_matching_auth_scheme(monkeypatch):
    adapter = DiscreteTimeDoubaoAdapter(200)
    monkeypatch.setenv("DOUBAO_REALTIME_API_KEY", "unit-test-token")
    monkeypatch.delenv("DOUBAO_APP_ID", raising=False)
    assert adapter._headers() == {"X-Api-Key": "unit-test-token"}
    monkeypatch.setenv("DOUBAO_APP_ID", "unit-test-app")
    headers = adapter._headers()
    assert headers["X-Api-App-ID"] == "unit-test-app"
    assert headers["X-Api-Access-Key"] == "unit-test-token"
    assert "X-Api-Key" not in headers


@run_async
async def test_doubao_mutes_once_during_stall_then_unmutes(monkeypatch):
    import tau2.voice.audio_native.doubao.discrete_time_adapter as module

    monkeypatch.setattr(module, "DEFAULT_VOICE_STREAM_IDLE_SECONDS", 0.001)
    adapter = DiscreteTimeDoubaoAdapter(20)
    adapter._events = asyncio.Queue()
    adapter._send_event = AsyncMock()
    adapter._send_audio_chunked = AsyncMock()
    task = asyncio.create_task(adapter._idle_watchdog())
    try:
        await asyncio.sleep(0.02)
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    assert adapter.idle_mute_count == 1
    assert (
        adapter._send_event.await_args_list[0].args[0]["type"]
        == "input_audio_mute.commit"
    )
    result = TickResult(tick_number=0, audio_sent_bytes=160, audio_sent_duration_ms=20)
    await adapter._execute_tick(b"\xff" * 160, 0, result, 0)
    assert (
        adapter._send_event.await_args_list[1].args[0]["type"]
        == "input_audio_unmute.commit"
    )
    assert not adapter._muted


@run_async
async def test_live_keepalive_sends_silence_only_when_ticks_idle(monkeypatch):
    import tau2.voice.audio_native.openai.live_discrete_time_adapter as module

    monkeypatch.setattr(module, "DEFAULT_VOICE_STREAM_IDLE_SECONDS", 0.001)
    adapter = DiscreteTimeOpenAILiveAdapter(20)
    sent = asyncio.Event()
    adapter._send_audio = AsyncMock(side_effect=lambda audio: sent.set())
    task = asyncio.create_task(adapter._keep_input_running())
    try:
        await asyncio.wait_for(sent.wait(), 0.5)
        assert adapter._send_audio.await_args.args[0] == b"\xff" * 160
        adapter._sending_audio = True
        calls = adapter._send_audio.await_count
        await asyncio.sleep(0.045)
        assert adapter._send_audio.await_count == calls
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


def test_doubao_disconnect_resets_state_even_if_close_fails():
    from unittest.mock import Mock

    adapter = DiscreteTimeDoubaoAdapter(200)
    adapter._connected = True
    adapter._tick_count = 9
    adapter._current_output_item = "old"
    adapter._bg_loop = Mock(is_running=True)

    def fail_close(coro, timeout):
        coro.close()
        raise TimeoutError("unit-test close timeout")

    adapter._bg_loop.run_coroutine.side_effect = fail_close
    with pytest.raises(TimeoutError):
        adapter.disconnect()
    assert not adapter._connected
    assert adapter._tick_count == 0
    assert adapter._current_output_item is None
    adapter._bg_loop.stop.assert_called_once()
