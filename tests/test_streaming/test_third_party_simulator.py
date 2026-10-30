"""Offline behavioral coverage for the two-speaker interference experiment."""

import json

import numpy as np
import pytest

from experiments.tau_voice.third_party_interference import build_user
from tau2.data_model.audio import (
    PCM_SAMPLE_RATE,
    AudioData,
    AudioEncoding,
    AudioFormat,
    audio_bytes_to_string,
)
from tau2.data_model.message import AssistantMessage, TurnTakingAction
from tau2.data_model.simulation import AudioNativeConfig
from tau2.user.third_party_simulator import (
    InterferenceController,
    ReplyJudgment,
    ThirdPartyScenario,
    mix_pcm16,
    scenario_llm_args,
)
from tau2.voice.audio_native.adapter import create_adapter
from tau2.voice.audio_native.openai.live_discrete_time_adapter import (
    DiscreteTimeOpenAILiveAdapter,
)
from tau2.voice.audio_native.tick_result import TickResult


def test_ignored_first_still_allows_second_after_new_normal_turn():
    controller = InterferenceController(ThirdPartyScenario())
    assert not controller.can_start(10, 0, 3, True)
    assert not controller.can_start(10, 1, 1, True)
    assert not controller.can_start(10, 1, 3, False)
    assert controller.can_start(10, 1, 3, True)
    first = controller.start(10, 4, 1, "travel advice")
    for i in range(120):
        controller.observe(10 + i * 0.2, True, "继续旅行话题", 0.2)
    assert not controller.correction_due(20, True)
    controller.finish_if_due(32, True)
    assert first.status == "observed"
    assert not controller.can_start(48, 1, 3, True)
    assert controller.can_start(48, 2, 3, True)
    second = controller.start(48, 4, 2, "more travel advice")
    controller.finish_if_due(70, True)
    assert second.index == 2
    assert controller.report()["status"] == "complete"


@pytest.mark.parametrize(
    "verdict,expected", [("yes", True), ("no", False), ("unknown", False)]
)
def test_correction_requires_pause_and_semantic_reply(verdict, expected):
    c = InterferenceController(ThirdPartyScenario())
    event = c.start(10, 4, 1, "travel")
    c.record_judgment(11, ReplyJudgment(verdict=verdict, reason="evidence"))
    assert not c.correction_due(15, True)  # semantics alone is insufficient
    for now in (11, 11.2, 11.4):
        c.observe(now, False, "", 0.2)
    assert event.pause_at_seconds == pytest.approx(11)
    assert not c.correction_due(13, True)  # background clip not finished
    assert not c.correction_due(15, False)  # cannot interrupt a silent agent
    assert c.correction_due(15, True) is expected
    event.correction_at_seconds = 15
    assert not c.correction_due(15.2, True)


def test_second_event_never_triggers_another_correction():
    c = InterferenceController(ThirdPartyScenario())
    c.start(10, 4, 1, "travel").closed_at_seconds = 32
    e = c.start(50, 4, 2, "travel")
    e.pause_at_seconds = 51
    e.responded_to_third_party = True
    assert not c.correction_due(55, True)


def test_missing_events_judge_failure_and_missed_correction_are_incomplete():
    c = InterferenceController(ThirdPartyScenario())
    assert c.report()["status"] == "incomplete"
    e = c.start(10, 4, 1, "travel")
    e.pause_at_seconds = 12
    c.record_judgment(15, ReplyJudgment(verdict="unknown", reason="unavailable"))
    c.finish_if_due(32, True)
    assert e.status == "inconclusive"
    assert e.responded_to_third_party is None
    c = InterferenceController(ThirdPartyScenario())
    e = c.start(10, 4, 1, "travel")
    e.pause_at_seconds = 12
    e.responded_to_third_party = True
    c.finish_if_due(32, True)
    assert e.status == "correction_missed"
    assert c.report()["status"] == "incomplete"


def test_mixing_does_not_wrap_int16():
    actual = mix_pcm16(
        np.array([30000, -30000], dtype=np.int16), np.array([10000, -10000])
    )
    assert actual.tolist() == [32767, -32768]


@pytest.fixture
def prepared_user(tmp_path, monkeypatch):
    scenario = ThirdPartyScenario()
    user = build_user(scenario, tmp_path, AudioNativeConfig())
    voices = []

    def fake_tts(text, provider, config):
        voices.append(config.voice)
        assert provider == "openai"
        return AudioData(
            data=np.full(PCM_SAMPLE_RATE * 4, 1000, dtype="<i2").tobytes(),
            format=AudioFormat(
                encoding=AudioEncoding.PCM_S16LE, sample_rate=PCM_SAMPLE_RATE
            ),
        )

    monkeypatch.setattr("tau2.user.third_party_simulator.synthesize_voice", fake_tts)
    user.prepare_audio()
    user.prepare_audio()  # cached, including across orchestrator's double initialization
    assert voices == ["onyx", "onyx", "nova"]
    return user


def assistant(speaking, text=""):
    return AssistantMessage(role="assistant", content=text, contains_speech=speaking)


def assistant_with_pcm(samples, speaking, text=""):
    return AssistantMessage(
        role="assistant",
        content=text,
        contains_speech=speaking,
        audio_content=audio_bytes_to_string(samples.astype("<i2").tobytes()),
        audio_format=AudioFormat(
            encoding=AudioEncoding.PCM_S16LE, sample_rate=PCM_SAMPLE_RATE
        ),
    )


def test_audio_injection_keeps_third_party_out_of_primary_history(prepared_user):
    user = prepared_user
    user.get_init_state()
    state = user.get_init_state()
    user.normal_turns = 1
    # Feed speech ticks through the actual streaming mixin and telephony pipeline.
    for _ in range(11):
        chunk, state = user.get_next_chunk(state, assistant(True, "旅行"))
    assert len(user.controller.events) == 1
    assert chunk.content == ""
    assert chunk.audio_script_gold == ""
    assert chunk.contains_speech is False
    assert chunk.audio_format.encoding == AudioEncoding.ULAW
    assert len(chunk.get_audio_bytes()) == 1600
    assert any(value != 127 for value in chunk.get_audio_bytes())
    primary = b"".join(user._speaker_taps["primary_user"]._chunks)
    background = b"".join(user._speaker_taps["third_party"]._chunks)
    assert len(primary) == len(background) == state.elapsed_samples * 2
    assert not np.any(np.frombuffer(primary, dtype="<i2"))
    assert np.any(np.frombuffer(background, dtype="<i2"))
    user.save_report()
    report = json.loads((user.scenario_dir / "events.json").read_text(encoding="utf-8"))
    assert report["status"] == "incomplete"


def test_background_waits_for_audible_audio_when_provider_flag_leads(prepared_user):
    user = prepared_user
    state = user.get_init_state()
    user.normal_turns = 1
    silence = np.zeros(user.chunk_size, dtype="<i2")
    voice = np.full(user.chunk_size, 1000, dtype="<i2")

    for _ in range(12):
        chunk = assistant_with_pcm(silence, True, "premature transcript")
        assert not user.speech_detection(chunk)
        state.input_turn_taking_buffer = [chunk]
        user._next_turn_taking_action(state)
        state.elapsed_samples += user.chunk_size

    assert user.controller.events == []
    assert user.consecutive_audible_agent_seconds == 0

    for _ in range(10):
        chunk = assistant_with_pcm(voice, False, "audible transcript")
        assert user.speech_detection(chunk)
        state.input_turn_taking_buffer = [chunk]
        user._next_turn_taking_action(state)
        state.elapsed_samples += user.chunk_size

    assert user.consecutive_audible_agent_seconds == pytest.approx(2.0)
    assert len(user.controller.events) == 1
    assert "premature transcript" not in user.controller.events[0].before_transcript
    assert "audible transcript" in user.controller.events[0].before_transcript


def test_empty_user_llm_turn_emits_silence_without_aborting(prepared_user, monkeypatch):
    user = prepared_user
    state = user.get_init_state()
    _, state = user.get_next_chunk(state, assistant(True, "Hi"))
    monkeypatch.setattr(
        "tau2.user.user_simulator_streaming.generate",
        lambda **_: AssistantMessage(role="assistant", content=""),
    )

    chunk, state = user._perform_turn_taking_action(
        state, TurnTakingAction(action="generate_message")
    )

    assert chunk.contains_speech is False
    assert chunk.get_audio_bytes()
    assert user.normal_turns == 0


def test_live_reply_correction_overlaps_assistant_and_allows_second(
    prepared_user, monkeypatch
):
    user = prepared_user
    state = user.get_init_state()
    user.normal_turns = 1
    user.controller.start(0, 4, 1, "京都旅行")
    monkeypatch.setattr(
        user, "_judge", lambda _: ReplyJudgment(verdict="yes", reason="回答鸡蛋问题")
    )
    for _ in range(3):
        _, state = user.get_next_chunk(state, assistant(False))
    # Resume while the third-party clip still plays; correct after clip ends.
    for _ in range(22):
        chunk, state = user.get_next_chunk(
            state, assistant(True, "冰箱里还有三个鸡蛋，你可以做番茄炒蛋。")
        )
        if chunk.contains_speech:
            break
    event = user.controller.events[0]
    assert event.correction_at_seconds is not None
    assert event.correction_at_seconds >= event.end_seconds
    assert chunk.contains_speech
    assert chunk.turn_taking_action.interruption_type == "third_party_correction"
    assert state.time_since_last_other_talk == 0
    assert user.normal_turns == 1  # correction does not qualify as the next normal turn
    while state.output_streaming_queue:
        _, state = user.get_next_chunk(state, assistant(False))
    # Stop before normal generation would call a live LLM.
    user.controller.events[0].correction_end_seconds = (
        state.elapsed_samples / PCM_SAMPLE_RATE
    )
    user.controller.finish_if_due(state.elapsed_samples / PCM_SAMPLE_RATE, True)
    now = state.elapsed_samples / PCM_SAMPLE_RATE + 16
    assert not user.controller.can_start(now, 1, 3, True)
    assert user.controller.can_start(now, 2, 3, True)


def test_same_voice_rejected():
    with pytest.raises(ValueError, match="must differ"):
        ThirdPartyScenario(primary_voice="nova", third_party_voice="nova")


def test_terra_request_omits_unsupported_sampling_parameter():
    args = scenario_llm_args("gpt-6.1-terra", temperature=0.7)
    assert args == {
        "custom_llm_provider": "openai",
        "reasoning_effort": "none",
    }
    assert scenario_llm_args("gpt-4.1", temperature=0) == {"temperature": 0}


def test_gpt_live_model_uses_live_api_adapter_and_responses_backend():
    adapter, resolved_model = create_adapter(
        "openai", tick_duration_ms=200, model="gpt-live-1"
    )
    assert isinstance(adapter, DiscreteTimeOpenAILiveAdapter)
    assert resolved_model == "gpt-live-1"
    config = adapter._session_config("Speak English.")
    assert config["model"] == "gpt-live-1"
    assert config["audio"]["format"] == {"type": "audio/pcmu", "rate": 8000}
    assert config["delegation"]["responses"]["model"] == "gpt-6.1-terra"
    assert config["delegation"]["responses"]["tool_choice"] == "none"


def test_gpt_live_audio_and_transcripts_flow_through_tick_result():
    adapter = DiscreteTimeOpenAILiveAdapter(tick_duration_ms=200)
    result = TickResult(
        tick_number=1,
        audio_sent_bytes=1600,
        audio_sent_duration_ms=200,
        bytes_per_tick=1600,
        bytes_per_second=8000,
    )
    adapter._process_event(
        result,
        {
            "type": "session.output_transcript.delta",
            "delta": "Hello there.",
            "start_ms": 100,
            "end_ms": 400,
        },
    )
    adapter._process_event(
        result,
        {
            "type": "session.output_audio.delta",
            "delta": "AQID",
        },
    )
    assert result.agent_audio_data == b"\x01\x02\x03"
    assert result.agent_audio_chunks[0][1] == "live-output-1"
    assert adapter._utterance_transcripts["live-output-1"].transcript_received == (
        "Hello there."
    )
    assert result.has_provider_activity


def test_provider_truncation_qualifies_even_without_long_pause():
    c = InterferenceController(ThirdPartyScenario())
    e = c.start(10, 4, 1, "travel")
    c.observe(11, True, "冰箱里还有三个鸡蛋", 0.2, was_truncated=True)
    assert e.pause_at_seconds is None
    assert e.provider_interrupted_at_seconds == 11
    c.record_judgment(
        15, ReplyJudgment(verdict="yes", reason="answers background question")
    )
    assert c.correction_due(15, True)


def test_failed_judge_does_not_reuse_earlier_negative_decision():
    c = InterferenceController(ThirdPartyScenario())
    e = c.start(10, 4, 1, "travel")
    c.record_judgment(12, ReplyJudgment(verdict="no", reason="still on topic"))
    c.record_judgment(14, ReplyJudgment(verdict="unknown", reason="judge failed"))
    assert e.responded_to_third_party is None


def test_primary_generation_uses_primary_voice_and_normal_turn_counter(
    prepared_user, monkeypatch
):
    user = prepared_user
    state = user.get_init_state()
    monkeypatch.setattr(
        "tau2.user.user_simulator_streaming.generate",
        lambda **_: AssistantMessage(role="assistant", content="京都的交通怎么安排？"),
    )
    voices = []

    def fake_tts(text, provider, provider_config):
        voices.append(provider_config.voice)
        return AudioData(
            data=np.full(PCM_SAMPLE_RATE, 1000, dtype="<i2").tobytes(),
            format=AudioFormat(
                encoding=AudioEncoding.PCM_S16LE, sample_rate=PCM_SAMPLE_RATE
            ),
        )

    monkeypatch.setattr("tau2.agent.base.voice.synthesize_voice", fake_tts)
    # The dummy assistant tick gives the real generator a valid linearized context.
    for _ in range(30):
        chunk, state = user.get_next_chunk(state, assistant(False))
        if chunk.contains_speech:
            break
    assert chunk.contains_speech
    assert voices == ["nova"]
    assert user.normal_turns == 1
