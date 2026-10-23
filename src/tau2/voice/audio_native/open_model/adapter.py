"""Native open models behind the shared frozen-time adapter template."""

import asyncio
import base64
import json
from pathlib import Path

from tau2.config import OPEN_MODEL_RUNTIME_MANIFESTS, SALMONN_BEFORE_SFT_CHECKPOINT
from tau2.voice.audio_native.adapter import DiscreteTimeAdapter
from tau2.voice.audio_native.audio_converter import StreamingTelephonyConverter
from tau2.voice.audio_native.open_model.transport import SubprocessTransport
from tau2.voice.audio_native.tick_result import UtteranceTranscript


def load_runtime_config(model):
    """Resolve explicit model assets without importing its GPU dependencies."""
    config = json.loads(Path(OPEN_MODEL_RUNTIME_MANIFESTS[model]).read_text())
    if model == "salmonn":
        config["checkpoint"] = SALMONN_BEFORE_SFT_CHECKPOINT
    return config


class OpenModelAdapter(DiscreteTimeAdapter):
    """One persistent native conversation; telephony input and output."""

    def __init__(self, config, runtime_config=None, transport=None):
        super().__init__(config.tick_duration_ms, pacing=config.pacing)
        self.config = config
        self.runtime_config = (
            load_runtime_config(config.model)
            if runtime_config is None
            else runtime_config
        )
        worker = (
            Path(__file__).resolve().parents[5] / "scripts/open_model_workers/worker.py"
        )
        self.transport = transport or SubprocessTransport(
            [self.runtime_config["python"], str(worker), "--model", config.model]
        )
        self._connected = False
        self.provenance = {}

    @property
    def is_connected(self):
        return self._connected

    def connect(self, system_prompt, tools, vad_config, modality="audio"):
        """Start fresh model, conversion, playback and transcript state."""
        if tools or modality != "audio":
            raise ValueError(
                "Open-model MVP supports audio conversations without tools"
            )
        if self.is_connected:
            self.disconnect()
        self.clear_buffers()
        self._tick_count = self._cumulative_user_audio_ms = 0
        try:
            self.provenance = self.transport.request(
                "connect", config=self.runtime_config, system_prompt=system_prompt
            )
        except Exception:
            self.transport.close()
            self.clear_buffers()
            raise
        rate = self.provenance.get(
            "sample_rate", 24000 if self.config.model == "minicpm" else 16000
        )
        self.converter = StreamingTelephonyConverter(16000, rate)
        self._connected = True

    def disconnect(self):
        """Release native caches even when shutdown fails."""
        try:
            if self._connected:
                self.transport.request("disconnect")
        finally:
            self._connected = False
            self.transport.close()
            self.clear_buffers()

    def run_tick(self, user_audio, tick_number=None):
        """Consume exactly one input tick and play at most one output tick."""
        if not self.is_connected:
            raise RuntimeError("Open model is not connected")
        if len(user_audio) != self.bytes_per_tick:
            raise ValueError(
                "Input audio must contain exactly one native simulation tick"
            )
        self._tick_count += 1
        return asyncio.run(
            self._async_run_tick(
                user_audio, tick_number if tick_number is not None else self._tick_count
            )
        )

    async def _execute_tick(self, user_audio, tick_number, result, tick_start):
        pcm = self.converter.convert_input(user_audio)
        expected = self.config.tick_duration_ms * 32
        # audioop's causal interpolator emits one fewer sample on first input.
        # Extend that initial boundary sample; never read a future tick.
        if len(pcm) == expected - 2:
            pcm += pcm[-2:]
        if len(pcm) != expected:
            raise RuntimeError("Native input sample count mismatch")
        event = await asyncio.to_thread(
            self.transport.request,
            "step",
            audio_pcm16_b64=base64.b64encode(pcm).decode(),
        )
        self._process_event(event, result, tick_number)

    def _process_event(self, event, result, tick_number):
        if event["sample_rate"] != self.converter.output_sample_rate:
            raise RuntimeError("Runtime changed its output sample rate")
        pcm = base64.b64decode(event["audio_pcm16_b64"], validate=True)
        if len(pcm) % 2:
            raise RuntimeError("Native PCM16 output has an incomplete sample")
        audio = self.converter.convert_output(pcm)
        if pcm and not any(pcm):
            audio = b""
        item = f"{tick_number}"
        text = event.get("text", "")
        self._utterance_transcripts[item] = UtteranceTranscript(
            item_id=item, audio_bytes_received=len(audio), transcript_received=text
        )
        if audio:
            result.agent_audio_chunks.append((audio, item))
        result.events.append(
            {
                "type": "native_generation",
                "text": text,
                "native_audio_samples": len(pcm) // 2,
                "sample_rate": event["sample_rate"],
            }
        )
        result.events.extend(event.get("events", []))

    async def _flush_pending_tool_results(self):
        if self._pending_tool_results:
            raise ValueError("Open-model MVP does not support tools")
