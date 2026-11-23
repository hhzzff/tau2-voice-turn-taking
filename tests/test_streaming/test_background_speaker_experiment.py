"""Offline protocol tests for background speaker rejection and retention."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.tau_voice.background_speaker_experiment import (
    Decision,
    Observation,
    Scenario,
    Session,
    Speaker,
    load_bundle,
    prepare,
    run_scenario,
    save_wav,
    semantic_judge,
)
from tau2.data_model.audio import (
    PCM_SAMPLE_RATE,
    TELEPHONY_AUDIO_FORMAT,
    AudioData,
    AudioEncoding,
    AudioFormat,
)
from tau2.data_model.message import AssistantMessage


def test_semantic_judge_retries_claims_without_transcript_evidence(monkeypatch):
    from tau2.utils import llm_utils

    calls = []
    responses = iter(
        [
            {
                "background_addressed": True,
                "user_addressed": False,
                "explanation": "Answered the roommate",
                "background_evidence": "Quincy Market is a good option",
            },
            {
                "background_addressed": False,
                "user_addressed": True,
                "explanation": "The transcript answers the user's broth question",
                "user_evidence": "A little broth keeps the pasta moist.",
            },
        ]
    )

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return AssistantMessage(role="assistant", content=json.dumps(next(responses)))

    monkeypatch.setattr(llm_utils, "generate", fake_generate)
    decision = semantic_judge("test-model")(
        {"current": {"transcript": "A little broth keeps the pasta moist."}}
    )

    assert decision.background_addressed is False
    assert decision.user_addressed is True
    assert len(calls) == 2


def test_semantic_judge_fails_when_evidence_remains_unverifiable(monkeypatch):
    from tau2.utils import llm_utils

    monkeypatch.setattr(
        llm_utils,
        "generate",
        lambda **_: AssistantMessage(
            role="assistant",
            content=json.dumps(
                {
                    "background_addressed": True,
                    "user_addressed": False,
                    "explanation": "Unsupported claim",
                    "background_evidence": "Invented assistant output",
                }
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="verifiable transcript evidence"):
        semantic_judge("test-model")({"current": {"transcript": "Continue meal prep."}})


@pytest.fixture
def scenario():
    return Scenario(
        name="offline",
        topic="meal prep",
        user=Speaker(voice="nova"),
        background=Speaker(voice="onyx"),
        opening="Help me plan dinners.",
        background_first="Should we drive or take the train?",
        correction="That is someone else. Remember their voice; do not answer them.",
        continuation=["What can I cook in one pan?", "How do I finish in one hour?"],
        background_second="Should we visit the museum first?",
        closing="Thanks, goodbye.",
    )


class FakeAgent:
    def __init__(self, responses=()):
        self.responses = iter(responses)
        self.inputs = []
        self.cleaned_up = False
        self.connections = 0
        self.audio_format = AudioFormat(
            encoding=AudioEncoding.PCM_S16LE, sample_rate=PCM_SAMPLE_RATE
        )

    def get_init_state(self):
        self.connections += 1
        return object()

    def get_next_chunk(self, state, message):
        self.inputs.append(message)
        return next(
            self.responses,
            AssistantMessage(role="assistant", contains_speech=False),
        ), state

    @staticmethod
    def is_stop(message):
        return False

    def cleanup(self):
        self.cleaned_up = True


class ProtocolSession:
    """Use controlled observations to exercise conditional branches."""

    def __init__(self, output, observations=None):
        self.agent = FakeAgent()
        self.output = output
        self.records = []
        self.tick_ms = 200
        self.state = None
        self.calls = []
        self.observations = observations or {}
        self.saved = False

    def observe(self, stage, pcm, gain_db=0, min_seconds=0):
        self.calls.append((stage, gain_db, min_seconds))
        start = len(self.records)
        self.records.extend([{}] * 10)
        values = {"speech_ticks": 2, "transcript": "A meaningful answer."}
        values.update(self.observations.get(stage, {}))
        return Observation(
            stage=stage, start_tick=start, end_tick=len(self.records), **values
        )

    def save(self):
        self.saved = True


def execute(scenario, tmp_path, decisions, observations=None):
    session = ProtocolSession(tmp_path, observations)
    evidence = []

    def judge(item):
        evidence.append(item)
        is_background = item["current_speaker"] == "background"
        return decisions.get(
            item["current"]["stage"],
            Decision(
                background_addressed=False,
                user_addressed=not is_background,
                explanation="test",
            ),
        )

    clips = {stage: b"audio" for stage, _, _ in scenario.utterances()}
    result = run_scenario(scenario, clips, session, judge)
    return result, session, evidence


def test_initial_rejection_skips_correction_and_retest(scenario, tmp_path):
    result, session, evidence = execute(
        scenario,
        tmp_path,
        {},
        {"background_first": {"speech_ticks": 0, "transcript": ""}},
    )
    assert result["status"] == "initial_rejection_pass"
    assert result["initial_rejection"] is True
    assert result["instruction_retention"] is None
    assert [item[0] for item in session.calls] == [
        "opening",
        "background_first",
        "closing",
    ]
    assert len(evidence) == 2  # Silence requires no semantic judge call.
    assert session.saved and session.agent.cleaned_up
    assert session.agent.connections == 1


@pytest.mark.parametrize("second_addressed", [False, True])
def test_retention_requires_correction_and_intervening_conversation(
    scenario, tmp_path, second_addressed
):
    result, session, _ = execute(
        scenario,
        tmp_path,
        {
            "background_first": Decision(
                background_addressed=True,
                user_addressed=False,
                explanation="Answered travel question",
            ),
            "background_second": Decision(
                background_addressed=second_addressed,
                user_addressed=False,
                explanation="Retest",
            ),
        },
    )
    assert result["status"] == (
        "retention_fail" if second_addressed else "retention_pass"
    )
    assert result["initial_rejection"] is False
    assert result["instruction_retention"] is not second_addressed
    assert result["post_correction_user_responsiveness"] is True
    assert [item[0] for item in session.calls] == [
        "opening",
        "background_first",
        "correction",
        "continuation_0",
        "continuation_1",
        "background_second",
        "closing",
    ]
    background_calls = [
        item for item in session.calls if item[0].startswith("background_")
    ]
    assert all(item[1] == scenario.background_gain_db for item in background_calls)
    assert session.calls[4][2] == 13  # Two seconds already spent in the previous turn.
    assert session.agent.connections == 1


def test_continuing_original_topic_is_not_background_response(scenario, tmp_path):
    result, _, _ = execute(
        scenario,
        tmp_path,
        {
            "background_first": Decision(
                background_addressed=False,
                user_addressed=True,
                explanation="Continued explaining meal prep",
            ),
        },
    )
    assert result["status"] == "initial_rejection_pass"


@pytest.mark.parametrize(
    "stage,observation",
    [
        ("opening", {"speech_ticks": 0, "transcript": ""}),
        ("background_first", {"speech_ticks": 1, "transcript": ""}),
        ("background_first", {"speech_ticks": 0, "transcript": "Text without audio"}),
        ("background_first", {"timed_out": True}),
        ("closing", {"speech_ticks": 0, "transcript": ""}),
    ],
)
def test_silence_missing_transcript_or_timeout_is_not_false_success(
    scenario, tmp_path, stage, observation
):
    result, session, _ = execute(scenario, tmp_path, {}, {stage: observation})
    assert result["status"] == "inconclusive"
    assert session.saved and session.agent.cleaned_up
    assert (
        json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))["status"]
        == "inconclusive"
    )


def test_ambiguous_attribution_does_not_choose_a_branch(scenario, tmp_path):
    result, session, _ = execute(
        scenario,
        tmp_path,
        {
            "background_first": Decision(
                background_addressed=None,
                user_addressed=None,
                explanation="Cannot determine",
            ),
        },
    )
    assert result["status"] == "inconclusive"
    assert result["initial_rejection"] is None
    assert len(session.calls) == 2


def test_true_user_must_still_get_responses_after_correction(scenario, tmp_path):
    result, session, _ = execute(
        scenario,
        tmp_path,
        {
            "background_first": Decision(
                background_addressed=True,
                user_addressed=False,
                explanation="Wrong addressee",
            ),
        },
        {"continuation_0": {"speech_ticks": 0, "transcript": ""}},
    )
    assert result["status"] == "inconclusive"
    assert result["instruction_retention"] is None
    assert "background_second" not in [item[0] for item in session.calls]


def test_audio_only_input_window_and_aligned_recordings(tmp_path):
    agent = FakeAgent(
        [
            AssistantMessage(role="assistant", content="hello", contains_speech=True),
        ]
    )
    session = Session(
        agent,
        tmp_path,
        tick_ms=200,
        response_seconds=0.4,
        max_response_seconds=1,
        quiet_seconds=0.2,
    )
    pcm = np.full(PCM_SAMPLE_RATE // 10, 10000, dtype="<i2").tobytes()
    observation = session.observe("background_first", pcm, gain_db=-6)
    assert not observation.timed_out
    assert observation.transcript == "hello"
    assert (
        len(agent.inputs) == 3
    )  # Padded input tick plus two silent observation ticks.
    assert all(
        message.content is None
        and message.audio_script_gold is None
        and message.source is None
        for message in agent.inputs
    )
    samples = np.frombuffer(agent.inputs[0].get_audio_bytes(), dtype="<i2")
    assert 5000 < samples[0] < 5020
    assert not samples[-1]  # Partial audio ticks are padded with silence.
    session.save()
    import wave

    with wave.open(str(tmp_path / "conversation.wav"), "rb") as recording:
        assert recording.getnchannels() == 2
        assert recording.getnframes() == len(agent.inputs) * PCM_SAMPLE_RATE // 5


def test_telephony_interface_converts_audio_and_preserves_tick_duration(tmp_path):
    agent = FakeAgent()
    agent.audio_format = TELEPHONY_AUDIO_FORMAT
    session = Session(agent, tmp_path)
    pcm = np.full(session.tick_bytes // 2, 1000, dtype="<i2").tobytes()
    for _ in range(3):
        session.tick(pcm, "opening", active=True)
    assert all(
        message.audio_format == TELEPHONY_AUDIO_FORMAT for message in agent.inputs
    )
    assert all(len(message.get_audio_bytes()) == 1600 for message in agent.inputs)
    assert all(len(audio) == session.tick_bytes for audio in session.agent_audio)
    assert all(
        not np.any(np.frombuffer(audio, dtype="<i2")) for audio in session.agent_audio
    )
    session.save()


def test_bundle_verifies_voice_identity_and_audio_hash(scenario, tmp_path):
    clips = {}
    pcm = np.full(PCM_SAMPLE_RATE // 10, 1000, dtype="<i2").tobytes()
    for stage, text, speaker in scenario.utterances():
        path = tmp_path / f"{stage}.wav"
        save_wav(path, pcm)
        clips[stage] = {
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "speaker": speaker.model_dump(),
            "text": text,
        }
    manifest = tmp_path / "bundle.json"
    bundle = {"scenario": scenario.model_dump(), "clips": clips}
    manifest.write_text(json.dumps(bundle), encoding="utf-8")
    loaded, audio = load_bundle(manifest)
    assert loaded.background.voice == "onyx"
    assert len(audio) == len(scenario.utterances())
    clips["background_second"]["speaker"]["voice"] = "nova"
    manifest.write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(ValueError, match="identity/text mismatch"):
        load_bundle(manifest)
    clips["background_second"]["speaker"]["voice"] = "onyx"
    manifest.write_text(json.dumps(bundle), encoding="utf-8")
    (tmp_path / "background_second.wav").write_bytes(b"modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_bundle(manifest)


def test_example_scenario_has_two_distinct_fixed_voices():
    path = (
        Path(__file__).parents[2]
        / "src/experiments/tau_voice/scenarios/background_speaker_meal_prep.json"
    )
    scenario = Scenario.model_validate_json(path.read_text(encoding="utf-8"))
    background_speakers = [
        speaker
        for stage, _, speaker in scenario.utterances()
        if stage.startswith("background_")
    ]
    assert background_speakers == [scenario.background, scenario.background]
    assert scenario.user != scenario.background


@pytest.mark.parametrize("provider", ["openai", "elevenlabs", "volcengine"])
def test_prepare_reuses_background_voice_for_both_probes(
    scenario, tmp_path, monkeypatch, provider
):
    import importlib

    synthesis = importlib.import_module("tau2.voice.synthesis.synthesize")
    scenario.user.provider = provider
    scenario.background.provider = provider
    calls = []

    def synthesize(text, selected_provider, config):
        calls.append((text, selected_provider, config))
        return AudioData(
            data=np.full(PCM_SAMPLE_RATE // 10, 1000, dtype="<i2").tobytes(),
            format=AudioFormat(
                encoding=AudioEncoding.PCM_S16LE, sample_rate=PCM_SAMPLE_RATE
            ),
        )

    monkeypatch.setattr(synthesis, "synthesize_voice", synthesize)
    output = tmp_path / "audio"
    prepare(scenario, output)
    loaded, audio = load_bundle(output / "bundle.json")
    assert loaded == scenario
    assert len(audio) == len(scenario.utterances())
    voice_attribute = {
        "openai": "voice",
        "elevenlabs": "voice_id",
        "volcengine": "speaker",
    }[provider]
    background_configs = [
        config
        for text, _, config in calls
        if text in {scenario.background_first, scenario.background_second}
    ]
    assert len(background_configs) == 2
    assert all(
        getattr(config, voice_attribute) == scenario.background.voice
        for config in background_configs
    )
    assert all(selected_provider == provider for _, selected_provider, _ in calls)
