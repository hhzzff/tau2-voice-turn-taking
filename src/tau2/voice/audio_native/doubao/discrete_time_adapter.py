"""Bridge Seeduplex's documented JSON protocol to telephony simulation ticks."""

import asyncio
import base64
import json
import os
import uuid
from contextlib import suppress
from types import SimpleNamespace

import websockets
from loguru import logger

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
    DEFAULT_DOUBAO_APP_KEY,
    DEFAULT_DOUBAO_INPUT_SAMPLE_RATE,
    DEFAULT_DOUBAO_OUTPUT_SAMPLE_RATE,
    DEFAULT_DOUBAO_REALTIME_BASE_URL,
    DEFAULT_DOUBAO_REALTIME_MODEL,
    DEFAULT_DOUBAO_REALTIME_VOICE,
    DEFAULT_SEND_AUDIO_INSTANT,
    DEFAULT_VOICE_STREAM_IDLE_SECONDS,
)
from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT
from tau2.voice.audio_native.adapter import DiscreteTimeAdapter
from tau2.voice.audio_native.async_loop import BackgroundAsyncLoop
from tau2.voice.audio_native.audio_converter import StreamingTelephonyConverter
from tau2.voice.audio_native.tick_result import UtteranceTranscript


class DiscreteTimeDoubaoAdapter(DiscreteTimeAdapter):
    """Use native audio only; simulator metadata is never sent to Seeduplex."""

    def __init__(
        self,
        tick_duration_ms,
        send_audio_instant=DEFAULT_SEND_AUDIO_INSTANT,
        model=None,
        audio_format=None,
    ):
        audio_format = audio_format or TELEPHONY_AUDIO_FORMAT
        if audio_format != TELEPHONY_AUDIO_FORMAT:
            raise ValueError("Doubao adapter requires the telephony audio format")
        if send_audio_instant:
            raise ValueError("Seeduplex requires realtime 20 ms audio pacing")
        super().__init__(tick_duration_ms, audio_format, send_audio_instant)
        self.model = model or DEFAULT_DOUBAO_REALTIME_MODEL
        self._converter = StreamingTelephonyConverter(
            DEFAULT_DOUBAO_INPUT_SAMPLE_RATE, DEFAULT_DOUBAO_OUTPUT_SAMPLE_RATE
        )
        self._bg_loop = BackgroundAsyncLoop()
        self._connected = False
        self._socket = None
        self._reader_task = None
        self._watchdog_task = None
        self._events = None
        self._muted = False
        self._sending = False
        self._last_audio_send = 0.0
        self._current_output_item = None
        self._next_item = 0
        self.session_id = None
        self.session_closed = None
        self.latest_usage = None
        self.idle_mute_count = 0

    @property
    def is_connected(self):
        return self._connected and self._bg_loop.is_running

    def connect(self, system_prompt, tools, vad_config=None, modality="audio"):
        """Start a fresh native speech session with environment-only credentials."""
        if tools or modality != "audio":
            raise ValueError("Doubao bench adapter supports audio without tools")
        if self.is_connected:
            self.disconnect()
        self._bg_loop.start()
        try:
            self._bg_loop.run_coroutine(
                self._async_connect(system_prompt),
                timeout=DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
            )
            self._connected = True
        except Exception:
            with suppress(Exception):
                self._bg_loop.run_coroutine(
                    self._async_disconnect(),
                    timeout=DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT + 5,
                )
            self._bg_loop.stop()
            raise

    def _headers(self):
        key = os.getenv("DOUBAO_REALTIME_API_KEY", "").strip()
        if not key:
            raise ValueError("DOUBAO_REALTIME_API_KEY is required")
        app_id = os.getenv("DOUBAO_APP_ID", "").strip()
        if app_id:
            return {
                "X-Api-App-ID": app_id,
                "X-Api-Access-Key": key,
                "X-Api-App-Key": DEFAULT_DOUBAO_APP_KEY,
                "X-Api-Resource-Id": "volc.speech.dialog",
                "X-Api-Connect-Id": str(uuid.uuid4()),
            }
        return {"X-Api-Key": key}

    async def _async_connect(self, prompt):
        self._socket = await websockets.connect(
            DEFAULT_DOUBAO_REALTIME_BASE_URL,
            additional_headers=self._headers(),
            proxy=None,
        )
        self._events = asyncio.Queue()
        self.session_closed = None
        self._reader_task = asyncio.create_task(self._read_events())
        await self._send_event(
            {
                "type": "session.create",
                "session": {
                    "model": self.model,
                    "instructions": prompt,
                    "audio": {
                        "input": {
                            "format": {
                                "type": "pcm",
                                "rate": DEFAULT_DOUBAO_INPUT_SAMPLE_RATE,
                            }
                        },
                        "output": {
                            "format": {
                                "type": "pcm_s16le",
                                "rate": DEFAULT_DOUBAO_OUTPUT_SAMPLE_RATE,
                            },
                            "voice": os.getenv(
                                "DOUBAO_REALTIME_VOICE", DEFAULT_DOUBAO_REALTIME_VOICE
                            ),
                        },
                    },
                    "tools": [],
                },
            }
        )
        event = await asyncio.wait_for(self._events.get(), 20)
        if event.get("type") != "session.created":
            raise RuntimeError(f"Doubao session rejected: {event}")
        self.session_id = event.get("session", {}).get("id")
        self._last_audio_send = asyncio.get_running_loop().time()
        self._muted = False
        self._watchdog_task = asyncio.create_task(self._idle_watchdog())
        logger.info(
            f"Connected to Doubao Seeduplex {self.model}; session={self.session_id}"
        )

    async def _read_events(self):
        try:
            async for message in self._socket:
                await self._events.put(json.loads(message))
        except Exception as exc:
            await self._events.put(
                {"type": "adapter.error", "message": type(exc).__name__}
            )

    async def _send_event(self, event):
        event.setdefault("event_id", f"event_{uuid.uuid4().hex}")
        await self._socket.send(json.dumps(event, ensure_ascii=False))

    async def _idle_watchdog(self):
        # LLM/TTS generation pauses simulation ticks. Explicitly mute the input
        # during these wall-clock stalls rather than letting the server time out.
        while True:
            await asyncio.sleep(DEFAULT_VOICE_STREAM_IDLE_SECONDS)
            if (
                not self._sending
                and not self._muted
                and (
                    asyncio.get_running_loop().time() - self._last_audio_send
                    >= DEFAULT_VOICE_STREAM_IDLE_SECONDS
                )
            ):
                await self._send_event({"type": "input_audio_mute.commit"})
                self._muted = True
                self.idle_mute_count += 1

    async def _flush_pending_tool_results(self):
        if self._pending_tool_results:
            raise RuntimeError("Doubao bench session has no tools")

    async def _execute_tick(self, user_audio, tick_number, result, tick_start):
        self._sending = True
        try:
            if self._muted:
                await self._send_event({"type": "input_audio_unmute.commit"})
                self._muted = False
            pcm = self._converter.convert_input(user_audio)
            width = round(
                DEFAULT_DOUBAO_INPUT_SAMPLE_RATE * 2 * self._voip_interval_ms / 1000
            )
            await self._send_audio_chunked(pcm, self._send_audio, width)
        finally:
            self._sending = False
            self._last_audio_send = asyncio.get_running_loop().time()
        # Sending consumes the tick, so drain already received events even when
        # its deadline has elapsed. Do not leave audio indefinitely queued.
        while not self._events.empty():
            self._process_event(result, self._events.get_nowait())

    async def _send_audio(self, pcm):
        await self._send_event(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode("ascii"),
            }
        )

    def _item(self, event):
        item_id = (
            event.get("response_id")
            or event.get("item_id")
            or self._current_output_item
        )
        if not item_id:
            self._next_item += 1
            item_id = f"doubao-output-{self._next_item}"
        self._current_output_item = item_id
        self._utterance_transcripts.setdefault(
            item_id, UtteranceTranscript(item_id=item_id)
        )
        return item_id

    def _process_event(self, result, event):
        kind = event.get("type", "unknown")
        result.events.append(SimpleNamespace(**event))
        self.record_control_signal("doubao_s2s", event, result)
        if kind == "response.output_audio.delta":
            pcm = base64.b64decode(event.get("delta", ""))
            audio = self._converter.convert_output(pcm)
            if audio:
                item = self._item(event)
                result.agent_audio_chunks.append((audio, item))
                self._utterance_transcripts[item].add_audio(len(audio))
        elif kind == "response.output_text.delta":
            self._utterance_transcripts[self._item(event)].add_transcript(
                event.get("delta", "")
            )
        elif kind == "conversation.item.input_audio_transcription.started":
            result.vad_events.append("speech_started")
            # The official client clears queued playback at this server event.
            self.cancel_pending_playback(result)
            self._current_output_item = None
        elif kind == "conversation.item.input_audio_transcription.completed":
            result.vad_events.append("speech_stopped")
        elif kind == "response.done":
            self.latest_usage = event.get("usage")
        elif kind == "session.closed":
            self.session_closed = event
        elif kind in {"error", "adapter.error"}:
            raise RuntimeError(
                f"Doubao {kind}: {event.get('error') or event.get('message')}"
            )

    def run_tick(self, user_audio, tick_number=None):
        """Run one simulation tick on the persistent websocket event loop."""
        if not self.is_connected:
            raise RuntimeError("Doubao session is not connected")
        if tick_number is None:
            tick_number = self._tick_count
        self._tick_count = tick_number + 1
        return self._bg_loop.run_coroutine(
            self._async_run_tick(user_audio, tick_number),
            timeout=self.tick_duration_ms / 1000
            + DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
        )

    def disconnect(self):
        """Wait for session.closed before closing the websocket."""
        try:
            if self._bg_loop.is_running:
                self._bg_loop.run_coroutine(
                    self._async_disconnect(),
                    timeout=DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT + 5,
                )
        finally:
            self._bg_loop.stop()
            self._connected = False
            self.clear_buffers()
            self._converter.reset()
            self._tick_count = 0
            self._cumulative_user_audio_ms = 0
            self._current_output_item = None
            self._next_item = 0
            self._muted = False

    async def _async_disconnect(self):
        if self._watchdog_task:
            self._watchdog_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._watchdog_task
        if self._socket:
            try:
                await self._send_event({"type": "session.close"})
                while not self.session_closed:
                    event = await asyncio.wait_for(self._events.get(), 5)
                    if event.get("type") == "session.closed":
                        self.session_closed = event
            finally:
                await self._socket.close()
                self._socket = None
        if self._reader_task:
            self._reader_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._reader_task
