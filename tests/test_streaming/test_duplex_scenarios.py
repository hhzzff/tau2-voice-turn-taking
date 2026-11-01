"""Offline tests for Terra's live event decisions and audio execution."""

import hashlib
import json
import os
import wave

import numpy as np
import pytest
from pydantic import ValidationError

from experiments.tau_voice.duplex_bench import (
    build_dynamic_user,
    build_parser,
    conversation_environment,
)
from tau2.config import DEFAULT_DUPLEX_USER_MODEL
from tau2.data_model.audio import PCM_SAMPLE_RATE, AudioData, AudioEncoding, AudioFormat
from tau2.data_model.message import AssistantMessage
from tau2.data_model.simulation import AudioNativeConfig
from tau2.user.duplex_evaluation import assess_report, attach_asr_evidence, judge_report
from tau2.user.duplex_events import (
    EVENT_CATALOG,
    DuplexDecision,
    DuplexScenario,
    DuplexTopic,
    generate_topic,
    select_events,
)
from tau2.user.duplex_scenario_user import RUNTIME_POLICY, DuplexController


def scenario(events):
    return DuplexScenario(
        selected_events=events,
        assistant_observation="proportional",  # Explicit offline text-only fixtures.
        topic="讨论一次野餐的安排",
        user_context="三个人，预算八百元，准备在周末出发。",
    )


def test_catalog_and_selection_do_not_create_a_script():
    families = [spec.family for spec in EVENT_CATALOG.values()]
    assert families.count("interruption") == 8
    assert families.count("rejection") == 6
    assert families.count("proactive") == 4
    assert families.count("turn_taking") == 1
    config = scenario(["follow_up", "rejection.noise"])
    assert config.selected_events == ["interruption.follow_up", "rejection.noise"]
    assert "events" not in config.model_dump()
    assert "opening" not in config.model_dump()
    assert "text" not in config.model_dump()
    assert config.user_model == DEFAULT_DUPLEX_USER_MODEL
    with pytest.raises(ValueError, match="ambiguous"):
        select_events(["clarification"])
    with pytest.raises(ValueError, match="Duplicate"):
        select_events(["follow_up", "interruption.follow_up"])


def test_prepare_saves_only_event_types_and_options(tmp_path):
    path = tmp_path / "scenario.json"
    args = build_parser().parse_args(
        ["prepare", "--events", "follow_up,rejection.noise", "--output", str(path)]
    )
    assert args.func(args) == 0
    saved = DuplexScenario.model_validate_json(path.read_text(encoding="utf-8"))
    assert saved.topic == ""
    assert saved.selected_events == ["interruption.follow_up", "rejection.noise"]
    assert (
        build_parser().parse_args(["run", "--events", "follow_up"]).user_model is None
    )


def test_run_keeps_all_call_logs_and_restores_logging_on_failure(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from experiments.tau_voice import duplex_bench
    from tau2.utils.llm_utils import _write_llm_log, llm_log_dir, llm_log_mode

    previous_directory = llm_log_dir.get()
    previous_mode = llm_log_mode.get()

    def failing_run(args):
        for content in ["first input", "second input"]:
            _write_llm_log(
                {"messages": [{"role": "user", "content": content}]},
                {"content": "saved output"},
                call_name="duplex_runtime_decision",
            )
        raise RuntimeError("trial failed after recorded calls")

    monkeypatch.setattr(duplex_bench, "_run", failing_run)
    with pytest.raises(RuntimeError, match="trial failed"):
        duplex_bench.run(SimpleNamespace(output=tmp_path))
    calls = [json.loads(p.read_text()) for p in (tmp_path / "llm_calls").glob("*.json")]
    assert len(calls) == 2
    assert {c["request"]["messages"][0]["content"] for c in calls} == {
        "first input",
        "second input",
    }
    assert all(c["response"]["content"] == "saved output" for c in calls)
    assert llm_log_dir.get() == previous_directory
    assert llm_log_mode.get() == previous_mode


def test_topic_generation_uses_terra_without_generating_event_lines(monkeypatch):
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return AssistantMessage(
            role="assistant",
            content=json.dumps(
                {"topic": "同学聚会", "user_context": "十个人，希望交通方便。"}
            ),
        )

    monkeypatch.setattr("tau2.utils.llm_utils.generate", fake_generate)
    topic = generate_topic(DuplexScenario(selected_events=["follow_up"]))
    assert topic.topic == "同学聚会"
    assert calls[0]["model"] == "gpt-6.1-terra"
    assert calls[0]["reasoning_effort"] == "none"
    assert set(topic.model_dump()) == {"topic", "user_context"}


def test_structured_topic_context_preserves_facts():
    context = {"goals": "交通方便", "budget": 800}
    topic = DuplexTopic(topic="同学聚会", user_context=context)
    assert json.loads(topic.user_context) == context


def test_controller_does_not_pick_events_or_consume_proposals():
    c = DuplexController(scenario(["follow_up", "rejection.noise"]), 0.2)
    decision = DuplexDecision(
        action="trigger_event",
        event_id="interruption.follow_up",
        text="等等，具体在哪集合？",
        reason="助手正在介绍地点。",
    )
    assert not c.allowed(decision, False)
    assert c.allowed(decision, True)
    assert len(c.pending) == 2
    assert not c.records
    assert not c.allowed(DuplexDecision(action="finish", reason="想结束。"), False)
    assert c.report()["status"] == "incomplete"


def test_contextual_event_requires_an_actual_spoken_prerequisite():
    c = DuplexController(scenario(["rejection.contextual"]), 0.2)
    trigger = DuplexDecision(
        action="trigger_event",
        event_id="rejection.contextual",
        text="助手，我是小林，可以帮我查一下天气吗？",
        reason="小林现在靠近了。",
    )
    assert not c.allowed(trigger, False)
    prepare = DuplexDecision(
        action="prepare_event",
        event_id="rejection.contextual",
        text="小林会在旁边说话，请不要回答他。",
        reason="先说明对象。",
    )
    c.begin(prepare, 0, 1, 1, False)
    assert c.allowed(trigger, False)
    assert c.pending == {"rejection.contextual"}


def test_memory_handoff_requires_prior_facts_and_keeps_evidence():
    c = DuplexController(scenario(["turn_taking.handoff"]), 0.2)
    trigger = DuplexDecision(
        action="trigger_event",
        event_id="turn_taking.handoff",
        text="What was the budget I gave you?",
        reason="Recall after another exchange.",
    )
    assert not c.allowed(trigger, False)
    c.begin(
        DuplexDecision(
            action="prepare_event",
            event_id="turn_taking.handoff",
            text="Our budget is eighty dollars.",
            reason="Establish a fact naturally.",
        ),
        0,
        2,
        2,
        False,
    )
    assert c.allowed(trigger, False)
    assert c.allowed(trigger, True)
    c.begin(trigger, 10, 2, 2, False)
    assert (
        c.records[0]["prior_user_turns"][0]["text"] == "Our budget is eighty dollars."
    )


def test_memory_answer_blocks_the_next_event_until_playback_is_quiet():
    """A long recall answer outlives six seconds and includes quiet audio frames."""
    c = DuplexController(scenario(["turn_taking.handoff", "topic_switch"]), 0.2)
    c.begin(
        DuplexDecision(
            action="trigger_event",
            event_id="turn_taking.handoff",
            text="Give me a menu using the earlier constraints.",
            reason="Recall facts established before an intervening exchange.",
        ),
        33,
        6.55,
        6.55,
        False,
    )
    next_event = DuplexDecision(
        action="trigger_event",
        event_id="interruption.topic_switch",
        text="Let's discuss transport instead.",
        reason="Another selected event.",
    )
    # Earlier answer leaking into the start of the request is not the recall answer.
    c.observe(33.2, True, "earlier reply", item_bearing_audio=True)
    c.advance_observation(39.6, False)
    assert c.active["memory_response_completion"]["answer_start_seconds"] is None
    for step in range(204, 301):
        now = step / 5
        c.observe(now, True, "recall answer", item_bearing_audio=True)
        c.advance_observation(now, False)
        assert c.active is not None
        assert not c.allowed(next_event, True)
    # Zero-energy frames still belong to the delivered utterance.
    for step in range(301, 311):
        now = step / 5
        c.observe(now, False, "", item_bearing_audio=True)
        c.advance_observation(now, False)
        assert c.active is not None
    for step in range(311, 315):
        now = step / 5
        c.observe(now, False, "")
        c.advance_observation(now, False)
        assert c.active is not None
    c.observe(63, False, "")
    c.advance_observation(63, False)
    assert c.active is None
    record = c.records[0]
    assert record["status"] == "observed"
    assert record["observation_end_seconds"] == 63
    assert record["memory_response_completion"]["status"] == "completed"
    assert c.allowed(next_event, True)


@pytest.mark.parametrize("continuous_reply", [False, True])
def test_memory_response_timeout_does_not_count_as_a_completed_observation(
    continuous_reply,
):
    config = scenario(["turn_taking.handoff", "rejection.noise"])
    config.memory_response_timeout_seconds = 20
    c = DuplexController(config, 0.2)
    c.begin(
        DuplexDecision(
            action="trigger_event",
            event_id="turn_taking.handoff",
            text="What constraints did I mention?",
            reason="Recall earlier facts.",
        ),
        10,
        2,
        2,
        False,
    )
    for step in range(61, 160):
        now = step / 5
        c.observe(now, continuous_reply, "answer" if continuous_reply else "")
        c.advance_observation(now, False)
        assert c.active is not None
    c.observe(32, continuous_reply, "answer" if continuous_reply else "")
    c.advance_observation(32, False)
    assert c.active is None
    c.pending.clear()
    assert c.report()["status"] == "incomplete"
    record = c.records[0]
    assert record["status"] == "observation_timeout"
    assert record["memory_response_completion"]["status"] == "timeout"
    assessed = assess_report(config, c.report(), 0.2)
    assert assessed["summary"]["observed"] == 0
    assert assessed["events"][0]["verdict"] == "not_evaluated"


def test_asr_review_keeps_original_window_and_memory_answer_separate():
    report = {
        "events": [
            {
                "event_id": key,
                "status": "observed",
                "start_seconds": 10,
                "observation_end_seconds": 20,
            }
            for key in ["interruption.follow_up", "turn_taking.handoff"]
        ],
        "decisions": [
            {"seconds": 30, "accepted": True, "decision": {"action": "finish"}}
        ],
    }
    snapshots = [
        {
            "segment_id": 1,
            "audio_end_seconds": 18,
            "decision_seconds": 18,
            "transcript": "Use",
            "error": None,
        },
        {
            "segment_id": 1,
            "audio_end_seconds": 20,
            "decision_seconds": 20,
            "transcript": "Use a bench",
            "error": None,
        },
        {
            "segment_id": 1,
            "audio_end_seconds": 30,
            "decision_seconds": 30,
            "transcript": "Use a bench for sixty dollars",
            "error": None,
        },
        {
            "segment_id": 2,
            "audio_end_seconds": 32,
            "decision_seconds": 32,
            "transcript": "Future unrelated speech",
            "error": None,
        },
    ]
    result = attach_asr_evidence(report, snapshots)
    ordinary, memory = result["events"]
    assert ordinary["asr_observations"] == [snapshots[1]]
    assert "memory_response_review" not in ordinary
    assert memory["observation_end_seconds"] == 20
    assert memory["memory_response_review"]["asr_observations"] == [snapshots[2]]
    assert "asr_observations" not in report["events"][0]


@pytest.mark.parametrize("speaker", ["primary", "background"])
def test_contextual_rejection_can_use_either_speaker(speaker):
    c = DuplexController(scenario(["rejection.contextual"]), 0.2)
    c.begin(
        DuplexDecision(
            action="prepare_event",
            event_id="rejection.contextual",
            text="Please ignore the conversation with Sam.",
            reason="Set a spoken condition.",
        ),
        0,
        2,
        2,
        False,
    )
    decision = DuplexDecision(
        action="trigger_event",
        event_id="rejection.contextual",
        contextual_speaker=speaker,
        text="Sam, when does the delivery arrive?",
        reason="Speak under the established condition.",
    )
    assert c.allowed(decision, False)
    assert c.begin(decision, 10, 2, 2, False) == speaker


def test_invalid_pause_and_user_stop_tokens_are_rejected():
    with pytest.raises(ValidationError, match="continuation"):
        DuplexDecision(
            action="trigger_event",
            event_id="rejection.pause",
            text="我想说的是……",
            reason="想一下。",
        )
    with pytest.raises(ValidationError, match="stop tokens"):
        DuplexDecision(action="speak", text="###STOP###", reason="结束。")


def test_private_events_never_reach_agent_policy(monkeypatch):
    monkeypatch.setenv("TAU2_CONTEXTUAL_TURN_TAKING_PLAN", "old private plan")
    with conversation_environment("Chinese"):
        assert "interruption" not in os.environ["TAU2_FREE_CONVERSATION_AGENT_POLICY"]
        assert os.environ["TAU2_CONTEXTUAL_TURN_TAKING_PLAN"] == ""
    assert os.environ["TAU2_CONTEXTUAL_TURN_TAKING_PLAN"] == "old private plan"


def _report(event_id, observations):
    return {
        "status": "complete",
        "events": [
            {
                "event_id": event_id,
                "status": "observed",
                "start_seconds": 1.0,
                "end_seconds": 3.0,
                "marker_seconds": 2.0,
                "pause_seconds": 1.2,
                "stimulus": {"text": "用户当时实际生成的话"},
                "observations": observations,
            }
        ],
    }


def test_response_activity_does_not_count_as_semantic_success():
    observations = [
        {"seconds": 1.0, "speaking": True, "text": "ongoing answer"},
        {"seconds": 3.2, "speaking": True, "text": "unrelated answer"},
    ]
    result = assess_report(
        scenario(["follow_up"]), _report("interruption.follow_up", observations), 0.2
    )
    assert result["events"][0]["verdict"] == "inconclusive"
    result = assess_report(
        scenario(["follow_up"]),
        _report("interruption.follow_up", observations[1:]),
        0.2,
    )
    assert result["events"][0]["verdict"] == "not_evaluated"


def test_background_overlapping_ongoing_answer_is_not_automatically_a_failure():
    report = _report(
        "rejection.third_party",
        [{"seconds": 1.2, "speaking": True, "text": "continuing original answer"}],
    )
    report["events"][0]["assistant_was_speaking"] = True
    result = assess_report(scenario(["rejection.third_party"]), report, 0.2)
    assert result["events"][0]["verdict"] == "inconclusive"


def test_failed_judge_and_missing_observations_never_pass(monkeypatch):
    monkeypatch.setattr(
        "tau2.utils.llm_utils.generate",
        lambda **kwargs: AssistantMessage(role="assistant", content="{}"),
    )
    config = scenario(["rejection.noise"])
    result = assess_report(config, _report("rejection.noise", []), 0.2)
    assert result["summary"]["pass"] == 0
    result = judge_report(config, result, "test-model")
    assert result["events"][0]["verdict"] == "inconclusive"


@pytest.mark.parametrize("overlap", [True, False])
def test_memory_verdict_is_independent_of_request_overlap(monkeypatch, overlap):
    config = scenario(["turn_taking.handoff"])
    report = _report(
        "turn_taking.handoff",
        [
            {"seconds": 1.2, "speaking": overlap, "text": "old reply"},
            {"seconds": 3.2, "speaking": True, "text": "Use tahini"},
        ],
    )
    row = report["events"][0]
    row["prior_user_turns"] = [
        {"source": "primary", "seconds": 0, "text": "My guest has a sesame allergy."},
        {"source": "background", "seconds": 0.5, "text": "I love sesame."},
    ]
    row["asr_observations"] = [
        {
            "segment_id": 1,
            "audio_start_seconds": 0,
            "audio_end_seconds": 2,
            "transcript": "Unrelated old reply",
        },
        {
            "segment_id": 2,
            "audio_start_seconds": 3.2,
            "audio_end_seconds": 4,
            "transcript": "Use",
        },
    ]
    row["memory_response_review"] = {
        "asr_observations": [
            {
                "segment_id": 2,
                "audio_start_seconds": 3.2,
                "audio_end_seconds": 6,
                "transcript": "Use tahini in the wrap.",
            },
        ]
    }

    def judge(**kwargs):
        assert kwargs["call_name"] == "duplex_memory_judge"
        prompt = kwargs["messages"][0].content
        assert "Do not replace memory evaluation with a timing explanation" in prompt
        payload = json.loads(prompt.split("\n")[-1])
        assert [s["transcript"] for s in payload["answer_asr"]] == [
            "Use tahini in the wrap."
        ]
        assert len(payload["prior_user_turns"]) == 1
        assert "user_context" not in payload
        assert "handoff_quality" not in payload
        return AssistantMessage(
            role="assistant",
            content=json.dumps(
                {
                    "verdict": "fail",
                    "reason": "Tahini violates the sesame allergy.",
                    "recall_test_validity": "valid",
                    "recall_test_reason": "Earlier allergy stated before a recall request.",
                    "facts": [
                        {
                            "fact": "Sesame allergy",
                            "prior_quote": "My guest has a sesame allergy.",
                            "prior_seconds": 0,
                            "result": "violated",
                            "answer_quote": "Use tahini in the wrap.",
                            "reason": "Tahini contains sesame.",
                        }
                    ],
                }
            ),
        )

    monkeypatch.setattr("tau2.utils.llm_utils.generate", judge)
    assessed = assess_report(config, report, 0.2)
    result = judge_report(config, assessed, "test-model")["events"][0]
    assert result["memory_assessment"]["verdict"] == "fail"
    assert result["memory_assessment"]["facts"][0]["result"] == "violated"
    assert "handoff_assessment" not in result
    assert result["verdict"] == "fail"


@pytest.mark.parametrize("assistant_speaking", [True, False])
def test_memory_request_can_overlap_assistant(assistant_speaking):
    c = DuplexController(scenario(["turn_taking.handoff"]), 0.2)
    c.prepared.add("turn_taking.handoff")
    request = DuplexDecision(
        action="trigger_event",
        event_id="turn_taking.handoff",
        text="Plan it using my earlier constraints.",
        reason="Recall.",
    )
    assert c.allowed(request, assistant_speaking)


@pytest.mark.parametrize("missing", ["facts", "answer"])
def test_memory_cannot_pass_without_content_evidence(monkeypatch, missing):
    config = scenario(["turn_taking.handoff"])
    report = _report(
        "turn_taking.handoff", [{"seconds": 3.2, "speaking": True, "text": "reply"}]
    )
    if missing != "answer":
        report["events"][0]["asr_observations"] = [
            {
                "segment_id": 1,
                "audio_start_seconds": 3.2,
                "audio_end_seconds": 5,
                "transcript": "Use lentils.",
            }
        ]
    content = {
        "verdict": "pass",
        "reason": "Looks good.",
        "recall_test_validity": "valid",
        "recall_test_reason": "Earlier constraints exist.",
        "facts": []
        if missing == "facts"
        else [
            {
                "fact": "Use lentils",
                "prior_quote": "I have lentils",
                "prior_seconds": 0,
                "result": "satisfied",
                "answer_quote": "Use lentils.",
                "reason": "Used the ingredient.",
            }
        ],
    }
    monkeypatch.setattr(
        "tau2.utils.llm_utils.generate",
        lambda **kwargs: AssistantMessage(
            role="assistant", content=json.dumps(content)
        ),
    )
    assessed = assess_report(config, report, 0.2)
    result = judge_report(config, assessed, "test-model")["events"][0]
    assert result["verdict"] == "inconclusive"
    assert "judge_error" in result
    assert "memory_assessment" not in result


@pytest.fixture
def fake_tts(monkeypatch):
    calls = []

    def synthesize(text, provider, config):
        calls.append((text, config.voice))
        return AudioData(
            data=np.full(16000, 1200, dtype="<i2").tobytes(),
            format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=16000),
        )

    monkeypatch.setattr("tau2.user.duplex_scenario_user.synthesize_voice", synthesize)
    return calls


def test_preserve_original_tts_before_resampling(monkeypatch, tmp_path):
    """Retain exact synthesis evidence when waveform and script may disagree."""
    raw = np.arange(2400, dtype="<i2").tobytes()
    monkeypatch.setattr(
        "tau2.user.duplex_scenario_user.synthesize_voice",
        lambda *args, **kwargs: AudioData(
            data=raw,
            format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=24000),
        ),
    )
    user = build_dynamic_user(
        scenario(["interruption.denial"]),
        tmp_path,
        AudioNativeConfig(),
        "nova",
        "onyx",
        "test-tts",
    )
    samples = user._speech("A complete sentence. Another sentence?")
    user._speech("Background speech.", background=True)
    evidence = json.loads((tmp_path / "synthesis.json").read_text(encoding="utf-8"))[
        "utterances"
    ]
    assert len(evidence) == 2
    first, second = evidence
    assert first["script_is_verified_transcript"] is False
    assert first["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    assert first["raw_duration_seconds"] == pytest.approx(0.1)
    assert first["resampled_duration_seconds"] == pytest.approx(0.1)
    assert len(samples) == PCM_SAMPLE_RATE // 10
    assert first["voice"] == "nova" and second["voice"] == "onyx"
    with wave.open(str(tmp_path / first["audio_file"]), "rb") as wav:
        assert wav.getframerate() == 24000
        assert wav.readframes(wav.getnframes()) == raw
    assert first["audio_file"] != second["audio_file"]


@pytest.mark.parametrize("assistant_text", ["周六九点在南门集合", "周日十点在北门集合"])
def test_runtime_model_controls_order_and_contextual_content(
    monkeypatch, tmp_path, fake_tts, assistant_text
):
    calls = []

    def decide(**kwargs):
        payload = json.loads(kwargs["messages"][0].content[len(RUNTIME_POLICY) + 1 :])
        calls.append(payload)
        pending = {item["id"] for item in payload["pending_events"]}
        if "rejection.noise" in pending:
            answer = DuplexDecision(
                action="trigger_event",
                event_id="rejection.noise",
                noise="phone_ring",
                reason="此时旁边的手机响了。",
            )
        elif "interruption.follow_up" in pending and payload["assistant_speaking"]:
            heard = payload["recent_conversation"][-1]["text"]
            answer = DuplexDecision(
                action="trigger_event",
                event_id="interruption.follow_up",
                text=f"等一下，关于你说的{heard}，能具体说明吗？",
                reason="刚才的实际回答产生了追问。",
            )
        elif not pending:
            answer = DuplexDecision(action="finish", reason="已经覆盖完事件。")
        else:
            answer = DuplexDecision(action="wait", reason="等待助手提供信息。")
        assert kwargs["model"] == "gpt-6.1-terra"
        return AssistantMessage(role="assistant", content=answer.model_dump_json())

    monkeypatch.setattr("tau2.user.duplex_scenario_user.generate", decide)
    user = build_dynamic_user(
        scenario(["follow_up", "rejection.noise"]),
        tmp_path,
        AudioNativeConfig(),
        "nova",
        "onyx",
        "test-tts",
    )
    assert not fake_tts
    state = user.get_init_state()
    for _ in range(500):
        speaking = (
            user.controller.pending == {"interruption.follow_up"}
            and user.controller.active is None
        )
        incoming = AssistantMessage(
            role="assistant",
            content=assistant_text if speaking else "",
            contains_speech=speaking,
        )
        chunk, state = user.get_next_chunk(state, incoming)
        if user.is_stop(chunk):
            break
    else:
        pytest.fail("Runtime did not finish")
    assert [record["event_id"] for record in user.controller.records] == [
        "rejection.noise",
        "interruption.follow_up",
    ]
    assert assistant_text in fake_tts[0][0]
    assert user.controller.finished
    assert user.controller.report()["status"] == "complete"
    assert len(calls) >= 3


def test_quiet_item_bearing_audio_cannot_end_the_recording(monkeypatch, tmp_path):
    calls = []

    def finish(**kwargs):
        calls.append(kwargs)
        return AssistantMessage(
            role="assistant",
            content=DuplexDecision(
                action="finish", reason="All coverage complete."
            ).model_dump_json(),
        )

    monkeypatch.setattr("tau2.user.duplex_scenario_user.generate", finish)
    user = build_dynamic_user(
        scenario(["rejection.noise"]),
        tmp_path,
        AudioNativeConfig(),
        "nova",
        "onyx",
        "test-tts",
    )
    state = user.get_init_state()
    user.controller.pending.clear()
    quiet_audio = AssistantMessage(
        role="assistant",
        content="",
        contains_speech=False,
        utterance_ids=["still-playing"],
    )
    for _ in range(20):
        chunk, state = user.get_next_chunk(state, quiet_audio)
        assert not user.is_stop(chunk)
    assert not calls
    idle = AssistantMessage(role="assistant", content="", contains_speech=False)
    for _ in range(10):
        chunk, state = user.get_next_chunk(state, idle)
        if user.is_stop(chunk):
            break
    else:
        pytest.fail("Recording never finished after actual audio delivery stopped")
    assert len(calls) == 1


def test_model_wait_is_respected_without_forcing_an_event(
    monkeypatch, tmp_path, fake_tts
):
    wait = DuplexDecision(action="wait", reason="当前没有自然的插话机会。")
    monkeypatch.setattr(
        "tau2.user.duplex_scenario_user.generate",
        lambda **kwargs: AssistantMessage(
            role="assistant", content=wait.model_dump_json()
        ),
    )
    user = build_dynamic_user(
        scenario(["follow_up"]),
        tmp_path,
        AudioNativeConfig(),
        "nova",
        "onyx",
        "test-tts",
    )
    state = user.get_init_state()
    for _ in range(180):
        _, state = user.get_next_chunk(
            state,
            AssistantMessage(
                role="assistant", content="继续解释。", contains_speech=True
            ),
        )
    assert user.controller.pending == {"interruption.follow_up"}
    assert not user.controller.records
    assert not fake_tts


def test_all_event_types_execute_from_runtime_model_actions(
    monkeypatch, tmp_path, fake_tts
):
    def decide(**kwargs):
        payload = json.loads(kwargs["messages"][0].content[len(RUNTIME_POLICY) + 1 :])
        pending = payload["pending_events"]
        if not pending:
            answer = DuplexDecision(action="finish", reason="事件已覆盖。")
        else:
            event_id = pending[-1]["id"]
            if (
                event_id
                in {
                    "rejection.contextual",
                    "proactive.contextual_interruption",
                    "turn_taking.handoff",
                }
                and event_id not in payload["prepared_events"]
            ):
                answer = DuplexDecision(
                    action="prepare_event",
                    event_id=event_id,
                    text="下一句话包含一个需要注意的条件，请先听清楚。",
                    reason="先建立条件。",
                )
            else:
                answer = DuplexDecision(
                    action="trigger_event",
                    event_id=event_id,
                    text="我想先补充一个想法。",
                    continuation=(
                        "我们等会儿再继续讨论这个具体安排。"
                        if event_id
                        in {"rejection.pause", "proactive.contextual_interruption"}
                        else ""
                    ),
                    pause_seconds=0.8 if event_id == "rejection.pause" else 0,
                    reason="此时对话提供了合适的机会。",
                )
        return AssistantMessage(role="assistant", content=answer.model_dump_json())

    monkeypatch.setattr("tau2.user.duplex_scenario_user.generate", decide)
    user = build_dynamic_user(
        scenario(list(EVENT_CATALOG)),
        tmp_path,
        AudioNativeConfig(),
        "nova",
        "onyx",
        "test-tts",
    )
    state = user.get_init_state()
    sources = set()
    for _ in range(4000):
        pending = sorted(user.controller.pending)
        next_event = pending[-1] if pending else None
        speaking = bool(
            next_event and EVENT_CATALOG[next_event].trigger == "assistant_speaking"
        )
        # Supply an actual answer to the protected memory request, then go quiet.
        active = user.controller.active
        if active and active["event_id"] == "turn_taking.handoff":
            now = state.elapsed_samples / PCM_SAMPLE_RATE
            speaking = active["end_seconds"] <= now < active["end_seconds"] + 2
        incoming = AssistantMessage(
            role="assistant",
            content="当前解释。" if speaking else "",
            contains_speech=speaking,
        )
        chunk, state = user.get_next_chunk(state, incoming)
        if chunk.source:
            sources.add(chunk.source)
        if chunk.source == "noise":
            assert not chunk.contains_speech
            assert not chunk.content
        if user.is_stop(chunk):
            break
    else:
        pytest.fail("Not all model-selected events finished")
    assert len(user.controller.records) == len(EVENT_CATALOG)
    assert all(record["status"] == "observed" for record in user.controller.records)
    assert {"background", "noise", "primary"} <= sources
    user.save_report()
    assert (tmp_path / "events.json").exists()
