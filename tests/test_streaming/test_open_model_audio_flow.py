"""Delivered-audio and clock contracts through the native agent and scheduler."""

import base64
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT
from tau2.data_model.message import UserMessage
from tau2.data_model.simulation import OpenModelConfig
from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator
from tau2.registry import registry


@pytest.mark.parametrize(
    "model,width,byte_count", [("salmonn", 0.08, 2560), ("minicpm", 1.0, 32000)]
)
def test_frozen_user_generation_waits_without_losing_previous_audio(
    model, width, byte_count, get_environment, base_task
):
    """Wall time can pause, but the previous audio still reaches the native step."""
    generation_started = threading.Event()
    finish_generation = threading.Event()
    native_inputs = []

    class NativeRuntime:
        def request(self, op, **payload):
            if op == "connect":
                return {"sample_rate": 16000}
            if op == "step":
                native_inputs.append(base64.b64decode(payload["audio_pcm16_b64"]))
                return {
                    "sample_rate": 16000,
                    "audio_pcm16_b64": "",
                    "text": "",
                    "events": [],
                }
            return {}

        def close(self):
            pass

    class User:
        def get_init_state(self, message_history=None):
            return 0

        def get_next_chunk(self, state, participant_chunk=None, tool_results=None):
            if state == 1:
                generation_started.set()
                assert finish_generation.wait(10), "test did not release generation"
            # Three known, distinguishable constant mu-law input intervals.
            value = [0x80, 0x90, 0xA0][state]
            return UserMessage(
                role="user",
                content="",
                is_audio=True,
                audio_format=TELEPHONY_AUDIO_FORMAT,
                audio_content=base64.b64encode(
                    bytes([value]) * round(width * 8000)
                ).decode(),
                contains_speech=True,
                chunk_id=state,
                is_final_chunk=True,
            ), state + 1

        @staticmethod
        def is_stop(message):
            return False

    agent = registry.get_agent_factory("open_model_agent")(
        [],
        "Listen to every input interval.",
        open_model_config=OpenModelConfig(pacing="frozen", model=model),
        runtime_config={},
        transport=NativeRuntime(),
    )
    scheduler = FullDuplexOrchestrator(
        domain="mock",
        agent=agent,
        user=User(),
        environment=get_environment(),
        task=base_task,
        tick_duration_seconds=width,
        pacing="frozen",
    )
    try:
        scheduler.initialize()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(scheduler.step)
            try:
                assert generation_started.wait(5)
                print(
                    f"{model}: native calls during blocked user generation = {len(native_inputs)}"
                )
                assert len(scheduler.get_ticks()) == 1
            finally:
                finish_generation.set()
            future.result(timeout=10)
        scheduler.step()
        assert len(native_inputs) == 2
        assert all(len(chunk) == byte_count for chunk in native_inputs)
        # Mu-law 0x80 and 0x90 decode to +32124 and +15996. Skip the single
        # resampler boundary sample; these literals come from the mu-law codec.
        import struct

        assert struct.unpack_from("<h", native_inputs[0], 4)[0] == 32124
        assert struct.unpack_from("<h", native_inputs[1], 4)[0] == 15996
    finally:
        finish_generation.set()
        agent.stop()
