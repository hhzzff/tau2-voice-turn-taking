"""Frozen-time behavior at the public configuration and scheduling boundaries."""

import pytest

from tau2.data_model.simulation import OpenModelConfig


@pytest.mark.parametrize(
    "model, milliseconds, identity",
    [
        ("salmonn", 80, "SALMONN-Omni-before-SFT-checkpoint-44000"),
        ("minicpm", 1000, "MiniCPM-o-4.5"),
    ],
)
def test_open_model_native_duration(model, milliseconds, identity):
    config = OpenModelConfig(model=model)
    assert config.tick_duration_ms == milliseconds
    assert config.model_identity == identity
    assert (
        OpenModelConfig(model=model, tick_duration_seconds=milliseconds / 1000)
        == config
    )
    with pytest.raises(ValueError, match="native tick"):
        OpenModelConfig(model=model, tick_duration_seconds=0.2)


def test_frozen_adapter_preserves_overflow_without_realtime_sleep(monkeypatch):
    import asyncio

    from tau2.voice.audio_native.adapter import DiscreteTimeAdapter

    class ControlledAdapter(DiscreteTimeAdapter):
        def connect(self, *args, **kwargs):
            pass

        def disconnect(self):
            pass

        @property
        def is_connected(self):
            return True

        def run_tick(self, user_audio, tick_number=0):
            return asyncio.run(self._async_run_tick(user_audio, tick_number))

        async def _execute_tick(self, user_audio, tick_number, result, tick_start):
            if tick_number == 0:
                result.agent_audio_chunks.append((b"\x01" * 1000, None))

        async def _flush_pending_tool_results(self):
            pass

    sleeps = []

    async def record_sleep(duration):
        sleeps.append(duration)

    monkeypatch.setattr(asyncio, "sleep", record_sleep)
    adapter = ControlledAdapter(tick_duration_ms=80, pacing="frozen")
    first = adapter.run_tick(b"\xff" * 640, 0)
    second = adapter.run_tick(b"\xff" * 640, 1)
    assert first.get_played_agent_audio() == b"\x01" * 640
    assert second.get_played_agent_audio() == b"\x01" * 360 + b"\x7f" * 280
    assert first.tick_sim_duration_ms == second.tick_sim_duration_ms == 80
    assert first.computation_duration_seconds >= 0
    assert sleeps == []
    realtime = ControlledAdapter(tick_duration_ms=80)
    realtime.run_tick(b"\xff" * 640)
    assert len(sleeps) == 1
    assert 0 < sleeps[0] <= 0.08


@pytest.mark.parametrize("duration", [0.08, 1.0])
def test_step_exchanges_previous_audio_and_separates_computation_time(
    duration, get_environment, base_task, monkeypatch
):
    import base64

    from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT
    from tau2.data_model.message import AssistantMessage, UserMessage
    from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator

    class Participant:
        def __init__(self, role):
            self.role = role
            self.received = []

        def get_init_state(self, message_history=None):
            return 0

        def get_next_chunk(self, state, participant_chunk=None, tool_results=None):
            self.received.append(participant_chunk)
            message = AssistantMessage if self.role == "assistant" else UserMessage
            return message(
                role=self.role,
                content=f"{self.role} {state}",
                audio_format=TELEPHONY_AUDIO_FORMAT,
                audio_content=base64.b64encode(
                    bytes([state + 1]) * int(duration * 8000)
                ).decode(),
            ), state + 1

        @staticmethod
        def is_stop(message):
            return False

    agent, user = Participant("assistant"), Participant("user")
    orchestrator = FullDuplexOrchestrator(
        domain="mock",
        agent=agent,
        user=user,
        environment=get_environment(),
        task=base_task,
        tick_duration_seconds=duration,
        pacing="frozen",
    )
    orchestrator.initialize()
    previous = orchestrator.get_ticks()[-1]
    clock = iter([10.0, 13.0, 20.0, 20.01])
    monkeypatch.setattr(
        "tau2.orchestrator.full_duplex_orchestrator.time.perf_counter",
        lambda: next(clock),
    )
    for _ in range(2):
        orchestrator.step()
        tick = orchestrator.get_ticks()[-1]
        assert agent.received[-1].audio_content == previous.user_chunk.audio_content
        assert user.received[-1].audio_content == previous.agent_chunk.audio_content
        assert tick.tick_duration_seconds == duration
        assert tick.agent_chunk.contains_speech and tick.user_chunk.contains_speech
        previous = tick
    assert orchestrator.get_ticks()[-2].wall_clock_duration_seconds == 3.0
    assert orchestrator.get_ticks()[-1].wall_clock_duration_seconds == pytest.approx(
        0.01
    )
    assert (
        sum(t.tick_duration_seconds for t in orchestrator.get_ticks()[-2:])
        == duration * 2
    )
