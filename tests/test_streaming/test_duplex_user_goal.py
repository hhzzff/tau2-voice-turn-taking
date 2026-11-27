"""Current user goals survive test boundaries and constrain actual decisions."""

import json

import numpy as np
import pytest
from duplex_runtime_fixture import runtime_json
from pydantic import ValidationError

from experiments.tau_voice.duplex_bench import build_dynamic_user
from tau2.data_model.audio import PCM_SAMPLE_RATE, AudioData, AudioEncoding, AudioFormat
from tau2.data_model.message import AssistantMessage
from tau2.data_model.simulation import AudioNativeConfig
from tau2.user.duplex_events import (
    DuplexDecision,
    DuplexRuntimeDecision,
    DuplexScenario,
)
from tau2.user.duplex_scenario_user import RUNTIME_POLICY, DuplexController
from tau2.user.duplex_user_goal import CurrentUserGoal


@pytest.mark.parametrize("quiet_item", [None, "answer"])
@pytest.mark.parametrize("action", ["speak", "prepare_event"])
def test_ordinary_turn_waits_through_a_pause_in_the_assistant_answer(
    monkeypatch, tmp_path, quiet_item, action
):
    user = build_dynamic_user(
        plan(), tmp_path, AudioNativeConfig(), "nova", "onyx", "test"
    )
    state = user.get_init_state()
    decision = DuplexDecision(
        action=action,
        event_id="turn_taking.handoff" if action == "prepare_event" else None,
        text="How would you carry that on the bus?",
        reason="Ask for a practical comparison",
    )
    monkeypatch.setattr(
        "tau2.user.duplex_scenario_user.generate",
        lambda **kwargs: AssistantMessage(
            role="assistant", content=runtime_json(decision.model_dump_json())
        ),
    )

    def step(seconds, speaking=False, item=None):
        state.elapsed_samples = round(seconds * PCM_SAMPLE_RATE)
        state.input_turn_taking_buffer.clear()
        state.input_turn_taking_buffer.append(
            AssistantMessage(
                role="assistant",
                contains_speech=speaking,
                utterance_ids=[item] if item else None,
            )
        )
        return user._next_turn_taking_action(state).action

    user.next_check = 57.8
    assert step(57.4, speaking=True, item="answer") == "wait"
    assert step(57.8, item=quiet_item) == "wait"
    assert user.pending_decision is None
    assert step(58, speaking=True, item="answer") == "wait"
    assert step(59.1) == "generate_message"
    assert user.pending_decision == decision


def plan():
    return DuplexScenario(
        selected_events=["rejection.third_party", "turn_taking.handoff"],
        include_proactive_opportunities=False,
        event_order=["rejection.third_party", "turn_taking.handoff"],
        assistant_observation="proportional",
        topic="Plan food and cooling for a picnic on the bus",
    )


def repair_goal():
    return CurrentUserGoal(
        objective="Choose practical food and cooling for the bus ride",
        pending_request="Finish the cooling comparison and offer a simpler option",
        status="needs_repair",
        unresolved_issues=[
            "Assistant mistook background speech for mine and abandoned its answer"
        ],
        stance="frustrated",
        evidence=[
            {
                "speaker": "background",
                "seconds": 69.4,
                "quote": "Jamie, are the keys by the front door?",
            },
            {
                "speaker": "assistant",
                "seconds": 73.8,
                "quote": "It sounds like you're talking to someone else",
            },
        ],
    )


def history():
    return [
        {
            "role": "user",
            "source": "primary",
            "seconds": 57.5,
            "text": "Could we use something simpler?",
        },
        {
            "role": "user",
            "source": "background",
            "seconds": 69.4,
            "text": "Jamie, are the keys by the front door?",
        },
        {
            "role": "assistant",
            "seconds": 73.8,
            "text": "It sounds like you're talking to someone else. Did you want to keep chatting about the picnic?",
        },
    ]


def test_identified_repair_blocks_next_event_and_finish_but_allows_normal_repair():
    controller = DuplexController(plan(), 0.1)
    controller.current_goal = repair_goal()
    controller.pending.remove("rejection.third_party")
    controller.records.append({"event_id": "rejection.third_party"})
    controller.prepared.add("turn_taking.handoff")
    advance = DuplexDecision(
        action="trigger_event",
        event_id="turn_taking.handoff",
        text="Yes, please give me the full plan.",
        reason="Next event",
    )
    assert not controller.allowed(advance, False)
    assert "goal" in controller.feedback.lower()
    repair = DuplexDecision(
        action="speak",
        text="That was someone beside me. You hadn't finished explaining the cooling options—please continue.",
        reason="Correct the speaker and recover the unfinished answer",
    )
    assert controller.allowed(repair, False)
    before = controller.pending.copy()
    controller.begin(repair, 79, 4, 4, False)
    assert controller.pending == before
    assert controller.current_goal.status == "needs_repair"
    assert controller.next_required_event == "turn_taking.handoff"
    assert not controller.allowed(
        DuplexDecision(action="finish", reason="Finished tests"), False
    )


def test_runtime_requires_goal_assessment_but_historical_decisions_still_load():
    archived = {"action": "wait", "reason": "Listen"}
    assert DuplexDecision.model_validate(archived).action == "wait"
    with pytest.raises(ValidationError, match="current_goal"):
        DuplexRuntimeDecision.model_validate(archived)


def test_cannot_mark_unresolved_work_as_satisfied():
    value = repair_goal().model_dump()
    value["status"] = "satisfied"
    with pytest.raises(ValidationError, match="unresolved"):
        CurrentUserGoal.model_validate(value)


def test_goal_reaches_runtime_and_report_and_rejects_wrong_speaker_evidence(
    monkeypatch, tmp_path
):
    user = build_dynamic_user(
        plan(), tmp_path, AudioNativeConfig(), "nova", "onyx", "test"
    )
    user.controller.history = history()
    payloads = []
    goal = repair_goal()

    def generate(**kwargs):
        payloads.append(
            json.loads(kwargs["messages"][0].content[len(RUNTIME_POLICY) + 1 :])
        )
        return AssistantMessage(
            role="assistant",
            content=json.dumps(
                {
                    "action": "wait",
                    "reason": "Wait for a natural boundary to repair",
                    "current_goal": goal.model_dump(),
                }
            ),
        )

    monkeypatch.setattr("tau2.user.duplex_scenario_user.generate", generate)
    assert user._decide(78).action == "wait"
    assert payloads[0]["current_goal"]["objective"] == plan().topic
    assert user.controller.current_goal == goal
    assert user.controller.report()["current_goal"]["status"] == "needs_repair"
    assert user._decide(79).action == "wait"
    assert payloads[1]["current_goal"]["unresolved_issues"] == goal.unresolved_issues
    goal = goal.model_copy(deep=True)
    goal.evidence[0].speaker = "primary"
    with pytest.raises(ValueError, match="evidence"):
        user._decide(80)
    assert user.controller.current_goal.evidence[0].speaker == "background"


def test_goal_can_clear_only_after_observed_answer_and_fresh_episode_resets_it(
    monkeypatch, tmp_path
):
    user = build_dynamic_user(
        plan(), tmp_path, AudioNativeConfig(), "nova", "onyx", "test"
    )
    user.controller.current_goal = repair_goal()
    user.controller.history = history() + [
        {
            "role": "assistant",
            "seconds": 90,
            "text": "Sorry, that was the background speaker. The lighter bag and frozen bottles solve the bus carrying problem.",
        }
    ]
    resolved = CurrentUserGoal(
        objective=repair_goal().objective,
        pending_request="",
        status="satisfied",
        evidence=[
            {
                "speaker": "assistant",
                "seconds": 90,
                "quote": "The lighter bag and frozen bottles solve the bus carrying problem.",
            }
        ],
    )
    monkeypatch.setattr(
        "tau2.user.duplex_scenario_user.generate",
        lambda **kwargs: AssistantMessage(
            role="assistant",
            content=json.dumps(
                {
                    "action": "wait",
                    "reason": "Now the answer is restored",
                    "current_goal": resolved.model_dump(),
                }
            ),
        ),
    )
    user._decide(94)
    assert user.controller.current_goal.status == "satisfied"
    user.get_init_state()
    assert user.controller.current_goal.status == "not_started"
    assert user.controller.current_goal.unresolved_issues == []


def test_proposed_repair_or_same_old_reply_cannot_clear_unresolved_goal(
    monkeypatch, tmp_path
):
    user = build_dynamic_user(
        plan(), tmp_path, AudioNativeConfig(), "nova", "onyx", "test"
    )
    user.controller.current_goal = repair_goal()
    user.controller.goal_assessed_at = 79
    user.controller.history = history()
    resolved = CurrentUserGoal(
        objective=repair_goal().objective,
        status="satisfied",
        evidence=[repair_goal().evidence[-1]],
    )
    monkeypatch.setattr(
        "tau2.user.duplex_scenario_user.generate",
        lambda **kwargs: AssistantMessage(
            role="assistant",
            content=json.dumps(
                {
                    "action": "speak",
                    "text": "That was someone else; please continue.",
                    "reason": "Repair the exchange",
                    "current_goal": resolved.model_dump(),
                }
            ),
        ),
    )
    with pytest.raises(ValueError, match="newly heard"):
        user._decide(80)
    assert user.controller.current_goal.status == "needs_repair"


def test_repair_runs_through_audio_without_leaking_goal_or_consuming_event(
    monkeypatch, tmp_path
):
    user = build_dynamic_user(
        plan(), tmp_path, AudioNativeConfig(), "nova", "onyx", "test"
    )
    state = user.get_init_state()
    user.controller.history = history()
    repair_text = "That was someone else. Please finish the cooling advice."
    monkeypatch.setattr(
        "tau2.user.duplex_scenario_user.generate",
        lambda **kwargs: AssistantMessage(
            role="assistant",
            content=json.dumps(
                {
                    "action": "speak",
                    "text": repair_text,
                    "reason": "Recover the unfinished answer",
                    "current_goal": repair_goal().model_dump(),
                }
            ),
        ),
    )
    synthesis_text = []

    def synthesize(text, provider, config):
        synthesis_text.append(text)
        return AudioData(
            data=np.full(16000, 1200, dtype="<i2").tobytes(),
            format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=16000),
        )

    monkeypatch.setattr("tau2.user.duplex_scenario_user.synthesize_voice", synthesize)
    output, state = user.get_next_chunk(
        state, AssistantMessage(role="assistant", content="", contains_speech=False)
    )
    assert synthesis_text == [repair_text]
    assert "needs_repair" not in (output.content or "")
    assert user.controller.records == []
    assert user.controller.pending == set(plan().event_ids)
    user.save_report()
    saved = json.loads((tmp_path / "events.json").read_text(encoding="utf-8"))
    assert saved["current_goal"]["status"] == "needs_repair"
    assert saved["decisions"][0]["current_goal"]["unresolved_issues"]
