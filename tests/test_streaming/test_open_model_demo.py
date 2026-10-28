"""Demo behavior at configuration and existing scheduler boundaries."""

import pytest

from tau2.data_model.simulation import AudioNativeConfig


def test_positive_subsecond_user_thresholds_survive_one_second_ticks():
    config = AudioNativeConfig(
        tick_duration_seconds=1.0,
        round_thresholds_up=True,
        wait_to_respond_threshold_other_seconds=0.2,
        wait_to_respond_threshold_self_seconds=0.4,
        yield_threshold_when_interrupted_seconds=0.5,
        yield_threshold_when_interrupting_seconds=0.5,
        interruption_check_interval_seconds=0.3,
        backchannel_min_threshold_seconds=0.3,
        backchannel_max_threshold_seconds=1.1,
    )
    assert config.wait_to_respond_threshold_other_ticks == 1
    assert config.wait_to_respond_threshold_self_ticks == 1
    assert config.yield_threshold_when_interrupted_ticks == 1
    assert config.yield_threshold_when_interrupting_ticks == 1
    assert config.interruption_check_interval_ticks == 1
    assert config.backchannel_min_threshold_ticks == 1
    assert config.backchannel_max_threshold_ticks == 2
    legacy = AudioNativeConfig(
        tick_duration_seconds=0.2, yield_threshold_when_interrupted_seconds=0.5
    )
    assert legacy.yield_threshold_when_interrupted_ticks == 2


def test_user_generation_uses_explicit_luna_with_existing_login(monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from tau2.data_model.message import SystemMessage
    from tau2.utils.llm_utils import generate

    def run(command, **kwargs):
        assert command[command.index("--model") + 1] == "gpt-5.6-luna"
        assert command[command.index("--sandbox") + 1] == "read-only"
        assert "env" not in kwargs  # never copy/reinterpret OAuth as API key
        Path(command[command.index("--output-last-message") + 1]).write_text(
            json.dumps({"utterance": "Actually, I meant Tuesday."})
        )
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr("subprocess.run", run)
    message = generate(
        "codex/gpt-5.6-luna",
        [SystemMessage(role="system", content="Speak as the user.")],
    )
    assert message.content == "Actually, I meant Tuesday."
    assert message.tool_calls is None


def test_doubao_asr_rejects_provider_error_even_on_http_success(monkeypatch):
    from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat
    from tau2.data_model.voice import TranscriptionConfig
    from tau2.voice.transcription.transcribe import transcribe_audio

    class Response:
        headers = {"X-Api-Status-Code": "45000001"}

        def raise_for_status(self):
            pass

        def json(self):
            return {"result": {"text": "must not use this"}}

    monkeypatch.setenv("VOLCENGINE_ASR_API_KEY", "test-only")
    monkeypatch.setattr("requests.post", lambda *a, **kw: Response())
    result = transcribe_audio(
        AudioData(
            data=b"\0" * 3200,
            format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=16000),
        ),
        TranscriptionConfig(model="doubao-asr"),
    )
    assert result.error
    assert result.transcript == ""


def test_demo_exports_previous_handoff_stereo_and_honest_overlap(
    tmp_path, get_environment, base_task
):
    import base64
    import json
    import wave

    from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT
    from tau2.data_model.message import AssistantMessage, UserMessage
    from tau2.data_model.simulation import OpenModelConfig
    from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator
    from tau2.scripts.open_model_demo import run_demo

    class Participant:
        def __init__(self, role):
            self.role = role

        def get_init_state(self, message_history=None):
            return 0

        def get_next_chunk(self, state, participant_chunk=None, tool_results=None):
            message = AssistantMessage if self.role == "assistant" else UserMessage
            return message(
                role=self.role,
                content="hello",
                is_audio=True,
                audio_format=TELEPHONY_AUDIO_FORMAT,
                audio_content=base64.b64encode(
                    bytes([0x20 if self.role == "user" else 0x40]) * 8000
                ).decode(),
                contains_speech=True,
            ), state + 1

        @staticmethod
        def is_stop(message):
            return False

    config = OpenModelConfig(model="minicpm")
    orchestrator = FullDuplexOrchestrator(
        domain="mock",
        agent=Participant("assistant"),
        user=Participant("user"),
        environment=get_environment(),
        task=base_task,
        tick_duration_seconds=1,
        pacing="frozen",
    )
    result = run_demo(
        orchestrator,
        config,
        "user_backchannel",
        tmp_path,
        steps=2,
        provenance=lambda: {
            "runtime": "controlled-test",
            "completed_steps": orchestrator.step_count,
        },
    )
    events = [
        json.loads(line) for line in (tmp_path / "ticks.jsonl").read_text().splitlines()
    ]
    assert len(events) == 3  # initial interval is preserved
    assert events[1]["user_input_sha256"] == events[0]["agent_audio_sha256"]
    assert events[1]["agent_input_sha256"] == events[0]["user_audio_sha256"]
    assert events[-1]["sim_end_seconds"] == 3
    assert "native_events" in events[-1]
    assert "adapter_computation_seconds" in events[-1]
    with wave.open(str(tmp_path / "conversation.wav")) as wav:
        assert wav.getnchannels() == 2
        assert wav.getnframes() == 24000
        assert wav.getframerate() == 8000
    assert "conversation.wav" in (tmp_path / "index.html").read_text()
    assert result["observed_overlap_seconds"] == 2
    assert result["semantic_outcome"] == "not_assessed"
    assert result["scenario_triggered"] is False
    assert result["provenance"]["completed_steps"] == 2


@pytest.mark.parametrize("model", ["minicpm", "salmonn"])
@pytest.mark.parametrize(
    "scenario",
    ["qa", "correction", "user_backchannel", "model_backchannel", "long_conversation"],
)
def test_real_user_scenario_uses_only_delivered_asr_and_triggers_backchannel(
    tmp_path, monkeypatch, get_environment, base_task, model, scenario
):
    import base64
    import json
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np

    from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT
    from tau2.data_model.message import AssistantMessage
    from tau2.data_model.simulation import OpenModelConfig
    from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator
    from tau2.scripts.open_model_demo import build_demo_user, run_demo

    prompts = []

    def codex(command, **kwargs):
        prompts.append(kwargs["input"])
        Path(command[command.index("--output-last-message") + 1]).write_text(
            json.dumps(
                {
                    "utterance": "YES"
                    if "Respond with ONLY" in kwargs["input"]
                    and "NO" in kwargs["input"]
                    else "Please explain how to plan a trip."
                }
            )
        )
        return SimpleNamespace(returncode=0, stdout="")

    pcm = (np.sin(np.arange(32000) * 0.1) * 5000).astype("<i2").tobytes()

    class Response:
        headers = {"X-Api-Status-Code": "20000000"}

        def raise_for_status(self):
            pass

        def iter_lines(self, **kwargs):
            return [
                json.dumps({"code": 0, "data": base64.b64encode(pcm).decode()}),
                json.dumps({"code": 20000000}),
            ]

        def json(self):
            return {"result": {"text": "heard words only"}}

    monkeypatch.setattr("subprocess.run", codex)
    monkeypatch.setattr("requests.post", lambda *a, **kw: Response())
    monkeypatch.setenv("VOLCENGINE_TTS_API_KEY", "test-only")
    monkeypatch.setenv("VOLCENGINE_ASR_API_KEY", "test-only")

    class Model:
        def create_initial_message(self):
            return AssistantMessage(
                role="assistant",
                is_audio=False,
                contains_speech=False,
                audio_format=TELEPHONY_AUDIO_FORMAT,
                audio_content=base64.b64encode(
                    b"\x7f" * (config.tick_duration_ms * 8)
                ).decode(),
            )

        def get_init_state(self, message_history=None):
            return 0

        def get_next_chunk(self, state, participant_chunk=None, tool_results=None):
            return AssistantMessage(
                role="assistant",
                content="UNHEARD FUTURE WORDS",
                is_audio=False,
                audio_format=TELEPHONY_AUDIO_FORMAT,
                audio_content=base64.b64encode(
                    (
                        b"\x7f"
                        if scenario == "qa"
                        and state * config.tick_duration_seconds >= 3
                        else b"\x20"
                    )
                    * (config.tick_duration_ms * 8)
                ).decode(),
                contains_speech=not (
                    scenario == "qa" and state * config.tick_duration_seconds >= 3
                ),
            ), state + 1

        @staticmethod
        def is_stop(message):
            return False

    config = OpenModelConfig(model=model)
    user = build_demo_user(config, scenario)
    orchestrator = FullDuplexOrchestrator(
        domain="mock",
        agent=Model(),
        user=user,
        environment=get_environment(),
        task=base_task,
        tick_duration_seconds=config.tick_duration_seconds,
        pacing="frozen",
    )
    report = run_demo(
        orchestrator,
        config,
        scenario,
        tmp_path,
        steps=int(
            (30 if scenario == "long_conversation" else 6)
            / config.tick_duration_seconds
        ),
        provenance={"runtime": "controlled-test"},
    )
    assert report["scenario_triggered"]
    if scenario == "long_conversation":
        assert "original VoiceStreamingUserSimulator" in report["user_policy"]
        assert (
            len(
                [e for e in report["trigger_events"] if e["kind"] == "generate_message"]
            )
            >= 2
        )
        assert user.decision_model == "codex/gpt-5.6-luna"
        assert any("Respond with ONLY" in prompt for prompt in prompts)
        assert any(
            "Known user facts and listener instructions" in prompt
            and "600 yuan" in prompt
            for prompt in prompts
        )
        substantive = user._substantive_turns
        user._generate_backchannel_message(orchestrator.user_state)
        assert user._substantive_turns == substantive
    else:
        assert report["trigger_events"][0]["kind"] == (
            "model_backchannel_opportunity"
            if scenario == "model_backchannel"
            else scenario
        )
    assert report["observed_overlap_seconds"] > 0
    assert report["asr_events"]
    assert all("UNHEARD FUTURE WORDS" not in prompt for prompt in prompts)
    assert any(
        t.user_chunk.turn_taking_action.action
        == ("backchannel" if scenario == "user_backchannel" else "generate_message")
        for t in orchestrator.get_ticks()
        if t.user_chunk.turn_taking_action
    )


@pytest.mark.parametrize("scenario", ["correction", "model_backchannel"])
def test_observed_trigger_follows_utterance_through_silent_chunks(
    tmp_path, get_environment, base_task, scenario
):
    import base64

    from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT
    from tau2.data_model.message import AssistantMessage, UserMessage
    from tau2.data_model.simulation import OpenModelConfig
    from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator
    from tau2.scripts.open_model_demo import run_demo

    class Participant:
        events = [{"tick_index": 0, "kind": scenario}]

        def __init__(self, role):
            self.role = role

        def get_init_state(self, message_history=None):
            return 0

        def get_next_chunk(self, state, participant_chunk=None, tool_results=None):
            message = AssistantMessage if self.role == "assistant" else UserMessage
            active = self.role == "assistant" or state not in (0, 2)
            return message(
                role=self.role,
                is_audio=True,
                audio_format=TELEPHONY_AUDIO_FORMAT,
                audio_content=base64.b64encode(
                    (b"\x20" if active else b"\x7f") * 8000
                ).decode(),
                chunk_id=state % 4,
                is_final_chunk=(state % 4 == 3),
                contains_speech=active,
            ), state + 1

        @staticmethod
        def is_stop(message):
            return False

    config = OpenModelConfig(model="minicpm")
    orchestrator = FullDuplexOrchestrator(
        domain="mock",
        agent=Participant("assistant"),
        user=Participant("user"),
        environment=get_environment(),
        task=base_task,
        tick_duration_seconds=1,
        pacing="frozen",
    )
    report = run_demo(orchestrator, config, scenario, tmp_path, steps=5, provenance={})
    assert report["scenario_triggered"]
    event = report["trigger_events"][0]
    assert event["user_active_seconds"] == 2.0
    assert event["observed_overlap_seconds"] == 2.0


def test_doubao_no_speech_status_is_empty_success(monkeypatch):
    from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat
    from tau2.data_model.voice import TranscriptionConfig
    from tau2.voice.transcription.transcribe import transcribe_audio

    class Response:
        headers = {"X-Api-Status-Code": "20000003"}

        def raise_for_status(self):
            pass

        def json(self):
            raise AssertionError("No-speech status does not require a transcript body")

    monkeypatch.setenv("VOLCENGINE_ASR_API_KEY", "test-only")
    monkeypatch.setattr("requests.post", lambda *args, **kwargs: Response())
    result = transcribe_audio(
        AudioData(
            data=b"\0" * 3200,
            format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=16000),
        ),
        TranscriptionConfig(model="doubao-asr"),
    )
    assert result.error is None
    assert result.transcript == ""
    assert result.provider_status == "20000003"


@pytest.mark.parametrize("no_speech_status", ["20000003", "20000000"])
def test_qa_waits_through_asr_no_speech_then_responds_to_real_transcript(
    monkeypatch, get_environment, base_task, no_speech_status
):
    import base64
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT
    from tau2.data_model.message import AssistantMessage
    from tau2.data_model.simulation import OpenModelConfig
    from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator
    from tau2.scripts.open_model_demo import build_demo_user

    prompts = []
    asr_calls = []

    def codex(command, **kwargs):
        prompts.append(kwargs["input"])
        Path(command[command.index("--output-last-message") + 1]).write_text(
            json.dumps({"utterance": "Please tell me more."})
        )
        return SimpleNamespace(returncode=0, stdout="")

    class Response:
        def __init__(self, status, text="A real spoken answer."):
            self.text = text
            self.headers = {"X-Api-Status-Code": status}

        def raise_for_status(self):
            pass

        def iter_lines(self, **kwargs):
            yield json.dumps(
                {"code": 0, "data": base64.b64encode(b"\x10\x10" * 16000).decode()}
            )
            yield json.dumps({"code": 20000000})

        def json(self):
            return {"result": {"text": self.text}}

    def post(url, **kwargs):
        if "/recognize/" in url:
            asr_calls.append(url)
            return Response(
                no_speech_status if len(asr_calls) <= 2 else "20000000",
                text="   " if len(asr_calls) <= 2 else "A real spoken answer.",
            )
        return Response("20000000")

    monkeypatch.setattr("subprocess.run", codex)
    monkeypatch.setattr("requests.post", post)
    monkeypatch.setenv("VOLCENGINE_TTS_API_KEY", "test-only")
    monkeypatch.setenv("VOLCENGINE_ASR_API_KEY", "test-only")

    class Model:
        @staticmethod
        def message(speech):
            return AssistantMessage(
                role="assistant",
                is_audio=False,
                contains_speech=speech,
                audio_format=TELEPHONY_AUDIO_FORMAT,
                audio_content=base64.b64encode(
                    (b"\x20" if speech else b"\x7f") * 8000
                ).decode(),
            )

        def create_initial_message(self):
            return self.message(False)

        def get_init_state(self, message_history=None):
            return 0

        def get_next_chunk(self, state, participant_chunk=None, tool_results=None):
            return self.message(state in (0, 1, 4, 5)), state + 1

        @staticmethod
        def is_stop(message):
            return False

    config = OpenModelConfig(model="minicpm")
    user = build_demo_user(config, "qa")
    orchestrator = FullDuplexOrchestrator(
        domain="mock",
        agent=Model(),
        user=user,
        environment=get_environment(),
        task=base_task,
        tick_duration_seconds=1,
        pacing="frozen",
    )
    orchestrator.initialize()
    for _ in range(4):
        orchestrator.step()
    assert len(prompts) == 1
    assert not user.events
    assert len(user.asr_events) == 2
    assert all(event["outcome"] == "no_speech" for event in user.asr_events)
    assert all(
        event["provider_status"] == no_speech_status for event in user.asr_events
    )
    for _ in range(4):
        orchestrator.step()
    assert len(prompts) == 2
    assert user.events[0]["kind"] == "qa"
    assert "A real spoken answer." in prompts[-1]
    assert "[ASR heard 0.00–3.00s]" not in prompts[-1]
    assert "[ASR heard 0.00–4.00s]" not in prompts[-1]


def test_long_asr_commits_stable_words_without_replaying_revisions():
    from tau2.voice.utils.incremental_transcript import IncrementalTranscript

    stream = IncrementalTranscript()
    assert stream.update("Hello, my name is") == ""
    assert stream.update("hello my name is Alice") == "hello my name is"
    assert stream.update("MY name is Alice and I travel") == "Alice"
    assert (
        stream.update("name is Alice and I travel tomorrow", final=True)
        == "and I travel tomorrow"
    )
    assert stream.update("name is ALICE and I travel tomorrow.", final=True) == ""


@pytest.mark.parametrize("call_name", ["interruption_decision", "backchannel_decision"])
def test_listener_calls_constrain_luna_to_decisions(monkeypatch, call_name):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from tau2.data_model.message import UserMessage
    from tau2.utils.llm_utils import generate

    def run(command, **kwargs):
        schema = json.loads(
            Path(command[command.index("--output-schema") + 1]).read_text()
        )
        assert schema["properties"]["utterance"]["enum"] == ["YES", "NO"]
        Path(command[command.index("--output-last-message") + 1]).write_text(
            '{"utterance":"YES"}'
        )
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr("subprocess.run", run)
    result = generate(
        "codex/gpt-5.6-luna",
        [
            UserMessage(
                role="user", content="Decide whether to acknowledge this complete idea."
            )
        ],
        call_name=call_name,
    )
    assert result.content == "YES"
