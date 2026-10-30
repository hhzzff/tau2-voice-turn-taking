"""Discrete-time adapter for OpenAI's GPT-Live WebSocket API."""

import asyncio
import base64
import json
import os
import uuid
from types import SimpleNamespace
from typing import Any, List, Optional

import websockets
from loguru import logger

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
    DEFAULT_OPENAI_LIVE_BASE_URL,
    DEFAULT_OPENAI_LIVE_CLOSE_TIMEOUT,
    DEFAULT_OPENAI_LIVE_MODEL,
    DEFAULT_OPENAI_LIVE_VOICE,
    DEFAULT_SEND_AUDIO_INSTANT,
)
from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT, AudioFormat
from tau2.environment.tool import Tool
from tau2.voice.audio_native.adapter import DiscreteTimeAdapter
from tau2.voice.audio_native.async_loop import BackgroundAsyncLoop
from tau2.voice.audio_native.tick_result import TickResult, UtteranceTranscript


class DiscreteTimeOpenAILiveAdapter(DiscreteTimeAdapter):
    """Bridge GPT-Live's continuous WebSocket session to simulation ticks."""

    def __init__(
        self,
        tick_duration_ms: int,
        send_audio_instant: bool = DEFAULT_SEND_AUDIO_INSTANT,
        model: Optional[str] = None,
        backend_model: str = "gpt-6.1-terra",
        audio_format: Optional[AudioFormat] = None,
    ):
        audio_format = audio_format or TELEPHONY_AUDIO_FORMAT
        if audio_format.encoding.value != "ulaw" or audio_format.sample_rate != 8000:
            raise ValueError("GPT-Live adapter currently requires 8 kHz G.711 mu-law")
        super().__init__(
            tick_duration_ms,
            audio_format=audio_format,
            send_audio_instant=send_audio_instant,
        )
        self.model = model or DEFAULT_OPENAI_LIVE_MODEL
        self.backend_model = backend_model
        self._chunk_size = int(
            self.audio_format.bytes_per_second * self._voip_interval_ms / 1000
        )
        self._bg_loop = BackgroundAsyncLoop()
        self._connected = False
        self._socket = None
        self._reader_task = None
        self._events = None
        self.session_id: Optional[str] = None
        self.session_closed: Optional[dict] = None
        self.latest_usage: Optional[dict] = None
        self._current_output_item: Optional[str] = None
        self._last_transcript_end_ms: Optional[int] = None
        self._next_output_item = 0

    @property
    def is_connected(self) -> bool:
        return self._connected and self._bg_loop.is_running

    def connect(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: Any = None,
        modality: str = "audio",
    ) -> None:
        if tools:
            raise ValueError("GPT-Live experiment adapter does not expose tools")
        if modality != "audio":
            raise ValueError("GPT-Live adapter supports audio modality only")
        if self._connected:
            self.disconnect()
        self._bg_loop.start()
        try:
            self._bg_loop.run_coroutine(
                self._async_connect(system_prompt),
                timeout=DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
            )
            self._connected = True
            logger.info(
                "Connected to OpenAI GPT-Live API "
                f"(session_id={self.session_id}, tick={self.tick_duration_ms}ms)"
            )
        except Exception as exc:
            self._bg_loop.stop()
            raise RuntimeError(
                f"Failed to connect to OpenAI GPT-Live API: {exc}"
            ) from exc

    async def _async_connect(self, system_prompt: str) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        self._socket = await websockets.connect(
            DEFAULT_OPENAI_LIVE_BASE_URL,
            additional_headers={"Authorization": f"Bearer {api_key}"},
        )
        self._events = asyncio.Queue()
        self._reader_task = asyncio.create_task(self._read_events())
        await self._send_event(
            {
                "type": "session.start",
                "event_id": f"event_{uuid.uuid4().hex}",
                "session": self._session_config(system_prompt),
            }
        )
        event = await asyncio.wait_for(self._events.get(), timeout=25)
        if event.get("type") == "error":
            detail = event.get("error", {})
            raise RuntimeError(
                f"{detail.get('code')}: {detail.get('message', 'Live session rejected')}"
            )
        if event.get("type") != "session.started":
            raise RuntimeError(f"Expected session.started, got {event.get('type')}")
        self.session_id = event.get("session", {}).get("id")

    def _session_config(self, system_prompt: str) -> dict:
        """Build the immutable Live model, audio, and backend settings."""
        return {
            "model": self.model,
            "instructions": system_prompt,
            "audio": {
                "format": {"type": "audio/pcmu", "rate": 8000},
                "output": {"voice": DEFAULT_OPENAI_LIVE_VOICE},
            },
            "delegation": {
                "type": "responses",
                "responses": {
                    "model": self.backend_model,
                    "tools": [],
                    "tool_choice": "none",
                },
            },
        }

    async def _read_events(self) -> None:
        try:
            async for message in self._socket:
                try:
                    event = json.loads(message)
                except (TypeError, json.JSONDecodeError) as exc:
                    await self._events.put(
                        {"type": "adapter.error", "message": str(exc)}
                    )
                    continue
                await self._events.put(event)
        except Exception as exc:
            if self._connected:
                await self._events.put(
                    {"type": "adapter.error", "message": type(exc).__name__}
                )

    async def _send_event(self, event: dict) -> None:
        if self._socket is None:
            raise RuntimeError("GPT-Live WebSocket is not connected")
        await self._socket.send(json.dumps(event, ensure_ascii=False))

    async def _flush_pending_tool_results(self) -> None:
        if self._pending_tool_results:
            raise RuntimeError("GPT-Live session was configured without function tools")

    async def _execute_tick(
        self,
        user_audio: bytes,
        tick_number: int,
        result: TickResult,
        tick_start: float,
    ) -> None:
        if user_audio:
            await self._send_audio_chunked(
                user_audio,
                self._send_audio,
                self._chunk_size,
            )
        deadline = tick_start + self.tick_duration_ms / 1000
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            try:
                event = await asyncio.wait_for(self._events.get(), timeout=remaining)
            except TimeoutError:
                break
            self._process_event(result, event)

    async def _send_audio(self, audio: bytes) -> None:
        await self._send_event(
            {
                "type": "session.input_audio.append",
                "audio": base64.b64encode(audio).decode("ascii"),
            }
        )

    def _process_event(self, result: TickResult, event: dict) -> None:
        event_type = event.get("type", "unknown")
        result.events.append(SimpleNamespace(**event))

        if event_type == "session.output_audio.delta":
            decoded = base64.b64decode(event.get("delta", ""))
            if decoded:
                item_id = self._current_output_item or self._new_output_item()
                result.agent_audio_chunks.append((decoded, item_id))
                self._utterance_transcripts[item_id].add_audio(len(decoded))
        elif event_type == "session.output_transcript.delta":
            delta = event.get("delta", "")
            if delta:
                start_ms = event.get("start_ms")
                if self._current_output_item is None or (
                    start_ms is not None
                    and self._last_transcript_end_ms is not None
                    and start_ms - self._last_transcript_end_ms > 1200
                ):
                    self._new_output_item()
                self._utterance_transcripts[self._current_output_item].add_transcript(
                    delta
                )
                if event.get("end_ms") is not None:
                    self._last_transcript_end_ms = event["end_ms"]
        elif event_type == "session.usage.updated":
            self.latest_usage = event.get("usage")
        elif event_type == "session.closed":
            self.session_closed = event
        elif event_type in ("error", "adapter.error"):
            error = event.get("error", {})
            message = event.get("message") or error.get("message", "Live API error")
            raise RuntimeError(message)
        elif event_type == "response.event":
            logger.debug("GPT-Live delegated backend event received")

    def _new_output_item(self) -> str:
        self._next_output_item += 1
        item_id = f"live-output-{self._next_output_item}"
        self._current_output_item = item_id
        self._utterance_transcripts[item_id] = UtteranceTranscript(item_id=item_id)
        return item_id

    def run_tick(
        self,
        user_audio: bytes,
        tick_number: Optional[int] = None,
    ) -> TickResult:
        if not self.is_connected:
            raise RuntimeError("GPT-Live API is not connected")
        if tick_number is None:
            tick_number = self._tick_count
        self._tick_count = tick_number + 1
        return self._bg_loop.run_coroutine(
            self._async_run_tick(user_audio, tick_number),
            timeout=self.tick_duration_ms / 1000
            + DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
        )

    def disconnect(self) -> None:
        if not self._connected:
            if self._bg_loop.is_running:
                self._bg_loop.stop()
            return
        try:
            self._bg_loop.run_coroutine(
                self._async_disconnect(),
                timeout=DEFAULT_OPENAI_LIVE_CLOSE_TIMEOUT
                + DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
            )
        except Exception as exc:
            logger.warning(f"Error finalizing GPT-Live session: {type(exc).__name__}")
        finally:
            self._bg_loop.stop()
            self._connected = False
            self._tick_count = 0
            self._cumulative_user_audio_ms = 0
            self.clear_buffers()

    async def _async_disconnect(self) -> None:
        if self._socket is None:
            return
        try:
            await self._send_event({"type": "session.close"})
            deadline = (
                asyncio.get_running_loop().time() + DEFAULT_OPENAI_LIVE_CLOSE_TIMEOUT
            )
            while not self.session_closed:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError("No session.closed event before close timeout")
                event = await asyncio.wait_for(self._events.get(), timeout=remaining)
                if event.get("type") == "session.closed":
                    self.session_closed = event
                elif event.get("type") == "session.usage.updated":
                    self.latest_usage = event.get("usage")
                elif event.get("type") in ("error", "adapter.error"):
                    break
        finally:
            await self._socket.close()
            if self._reader_task:
                await asyncio.gather(self._reader_task, return_exceptions=True)
            self._socket = None
