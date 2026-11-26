"""Controlled stimuli and disclosure cannot silently change during migration."""

from types import SimpleNamespace

import pytest

from experiments.tau_voice.local_user.agenda import ScenarioAgenda, settlement_reference
from experiments.tau_voice.local_user.policy import (
    FloorController,
    FloorState,
    effective_action,
)
from experiments.tau_voice.local_user.scenarios import (
    SCENE_EVENTS,
    load_clip,
    load_scenes,
)


def test_scene_catalog_matches_the_tau_event_ids():
    cases = load_scenes()
    assert len(cases) == 19
    assert SCENE_EVENTS["P03"] == "proactive.clarification"
    assert SCENE_EVENTS["M01"] == "turn_taking.handoff"
    clips = set()
    for case in cases:
        for stage in case["scenario_runtime"]["stages"]:
            clips.update(part["asset"] for part in stage.get("segments", []))
            if "asset" in stage:
                clips.add(stage["asset"])
    for ident in clips:
        pcm, row = load_clip(ident)
        assert pcm and row["sha256"]


def test_ctc_keeps_the_approved_continuous_retrace():
    case = next(c for c in load_scenes() if c["id"] == "P01")
    parts = case["scenario_runtime"]["stages"][0]["segments"]
    assert parts == [
        {"id": "continuous", "asset": "p01_v16_retrace", "gap_after_ms": 0}
    ]


def test_contextual_rejection_uses_the_same_speaker_before_and_after():
    case = next(c for c in load_scenes() if c["id"] == "R06")
    stimuli = [
        s for s in case["scenario_runtime"]["stages"] if s["kind"] == "environment"
    ]
    assert len(stimuli) == 2
    assert stimuli[0]["asset"] == stimuli[1]["asset"]
    assert stimuli[0].get("gain", 1) == stimuli[1].get("gain", 1)


@pytest.mark.parametrize("complete", [False, True])
def test_private_facts_require_actual_completed_transmission(tmp_path, complete):
    case = next(c for c in load_scenes() if c["id"] == "P03")
    rec = SimpleNamespace(path=tmp_path, emit=lambda *args, **kwargs: None)
    agenda = ScenarioAgenda(case, None, None, rec)
    stage = {
        "id": "answer1",
        "kind": "private_answer",
        "update": {"dinner_amount": 288, "taxi_amount": 78},
    }
    marks = {"user_first_voice_tx": 100}
    if complete:
        marks["user_audio_complete"] = 200
    agenda.user_finished(
        {"index": 1, "pending_intent": stage, "marks": marks, "text": "288，78"}
    )
    assert bool(agenda.profile) is complete
    assert agenda.task_states["answer1"]["delivered"] is complete
    # Information disclosure never declares the task semantically correct.
    assert agenda.task_states["answer1"]["met"] == {}
    assert agenda.manifest()["task_reference"]["ready"] is complete


def test_settlement_reference_requires_both_delivered_amounts():
    scene = next(c for c in load_scenes() if c["id"] == "P03")["scenario_runtime"]
    problem = scene["settlement_problem"]
    assert not settlement_reference(problem, {"dinner_amount": 288})["ready"]
    result = settlement_reference(problem, {"dinner_amount": 288, "taxi_amount": 78})
    assert [(r["from"], r["amount_cents"]) for r in result["transfers"]] == [
        ("小林", 5700),
        ("阿哲", 9600),
    ]


def test_floor_does_not_confuse_task_completion_with_speech_opportunity():
    floor = FloorController()
    state = FloorState(
        at_ns=2_000_000_000,
        own_turn=1,
        reply_seen=True,
        normal_end=True,
        playback_end_ns=1_000_000_000,
        last_voice_ns=1_000_000_000,
    )
    permission = floor.evaluate(state)
    assert effective_action("WAIT", permission) == "RESPOND"
    permission = floor.evaluate(state.model_copy(update={"own_active": True}))
    assert effective_action("INTERRUPT", permission) == "WAIT"
