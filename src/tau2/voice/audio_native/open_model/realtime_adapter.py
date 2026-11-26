"""Experimental independent transport clock around the existing native adapter.

Input and inference are independent: buffer exact native frames in one serial
worker, with no future samples; collect returned events at tick boundaries.
The same adapter serves production agents and controlled replay experiments.
"""

import asyncio
import base64
import queue
import threading
import time

from tau2.voice.audio_native.open_model.adapter import OpenModelAdapter

_SHUTDOWN_WAIT_SECONDS = 5


class RealtimeOpenModelAdapter(OpenModelAdapter):
    """Keep inference off the tick thread while reusing playback/control logic."""

    def __init__(self, config, runtime_config=None, transport=None, *, native_width_ms):
        super().__init__(config, runtime_config, transport)
        self.native_width_ms = native_width_ms
        self.draining = False
        self._thread = None

    def connect(self, system_prompt, tools, vad_config, modality="audio"):
        if self.is_connected:
            self.disconnect()
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Inference from the previous session is still stopping")
        super().connect(system_prompt, tools, vad_config, modality)
        self._inbound = queue.Queue()
        self._outbound = queue.Queue()
        self._origin = None
        self._windows = 0
        self._input = bytearray()
        self._submitted = self._completed = 0
        self.draining = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._infer, daemon=True)
        self._thread.start()

    @property
    def pending_native_steps(self):
        """Number of submitted requests whose events have not been consumed."""
        return self._submitted - self._completed

    def _infer(self):
        while not self._stop.is_set():
            task = self._inbound.get()
            if task is None:
                return
            index, raw, deadline = task
            if self._stop.wait(max(0, deadline - time.monotonic())):
                return
            try:
                pcm = self.converter.convert_input(raw)
                expected = self.native_width_ms * 32
                if len(pcm) == expected - 2:
                    pcm += pcm[-2:]
                if len(pcm) != expected:
                    raise RuntimeError("Native input sample count mismatch")
                event = self.transport.request(
                    "step", audio_pcm16_b64=base64.b64encode(pcm).decode()
                )
                if not self._stop.is_set():
                    self._outbound.put((index, event))
            except Exception as exc:
                if not self._stop.is_set():
                    self._outbound.put((index, exc))
                return

    async def _execute_tick(self, user_audio, tick_number, result, tick_start):
        if self._step_control:
            raise ValueError("Realtime adapter uses native autonomous control only")
        if self._origin is None:
            self._origin = time.monotonic()
        if not self.draining:
            self._input.extend(user_audio)
            size = self.native_width_ms * 8
            while len(self._input) >= size:
                raw = bytes(self._input[:size])
                del self._input[:size]
                self._submitted += 1
                self._inbound.put(
                    (
                        self._submitted,
                        raw,
                        self._origin + self._submitted * self.native_width_ms / 1000,
                    )
                )
        # Unlike frozen mode, expiry does not depend on any request completing.
        self._windows += 1
        deadline = self._origin + self._windows * self.tick_duration_ms / 1000
        await asyncio.sleep(max(0, deadline - time.monotonic()))
        while True:
            try:
                index, event = self._outbound.get_nowait()
            except queue.Empty:
                break
            self._completed += 1
            if isinstance(event, Exception):
                raise event
            self._process_event(event, result, f"native-{index}")

    async def _wait_for_tick_end(self, remaining: float) -> None:
        """The absolute deadline was already handled in _execute_tick()."""

    def _process_event(self, event, result, tick_number):
        if self.config.model == "freeze_omni" and any(
            item.get("type") == "controller_stop_tts"
            for item in event.get("events", [])
        ):
            # Match the direct reference controller, not a newly inferred
            # autonomous model interrupt or an unconditional listen-state cut.
            control = {
                "type": "reference_controller_playback_cancel",
                "source": "controller_stop_tts",
                "cancelled_audio_bytes": self.cancel_pending_playback(result),
            }
            result.events.append(control)
            result.serializable_events.append(control)
        super()._process_event(event, result, tick_number)

    def disconnect(self):
        if not self.is_connected:
            return
        if self._thread is not None:
            self._stop.set()
            self._inbound.put(None)
            self._thread.join(timeout=_SHUTDOWN_WAIT_SECONDS)
            if self._thread.is_alive():
                # Close aborts the transport without sending a second request.
                # Keep the thread handle so connect cannot reset its state early.
                self._connected = False
                try:
                    self.transport.close()
                    self._thread.join(timeout=_SHUTDOWN_WAIT_SECONDS)
                finally:
                    self.clear_buffers()
                return
        super().disconnect()
