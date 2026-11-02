"""Postconversation acoustic clarification assessments from saved delivery evidence."""

import pytest

from tau2.user.duplex_evaluation import assess_report
from tau2.user.duplex_events import AcousticClarificationConfig, DuplexScenario


def scenario():
    return DuplexScenario(
        selected_events=[],
        acoustic_clarification=AcousticClarificationConfig(mode="dropchunk"),
        topic="Travel",
        user_context="Private intent, not delivered context.",
    )


def recorded_report(mode="dropchunk"):
    target = {
        "id": "user-0001",
        "status": "incomplete",
        "mode": mode,
        "severity": "hard",
        "script_text": "Go to Paris then book a quiet hotel.",
        "script_is_verified_transcript": False,
        "keyword": {"text": "Paris", "start_char": 6, "end_char": 11},
        "alignment_response": {"characters": []},
        "target_samples": [100, 200],
        "sample_rate": 100,
        "sample_count": 500,
        "start_seconds": 0,
        "natural_end_seconds": 5,
        "actual_stop_seconds": 3,
        "modified_intervals_samples": [[120, 180]],
        "delivered_modified_samples": 60,
        "delivered_target_samples": 100,
        "cancelled_samples": [300, 500],
        "delivery": [
            {"start_seconds": 0, "end_seconds": 3, "sample_start": 0, "sample_end": 300}
        ],
        "delivered_audio_file": "target.wav",
    }
    return {
        "status": "incomplete",  # Truncated user tail is NOT interrupted conversation.
        "events": [],
        "decisions": [],
        "conversation": {"status": "complete", "termination_reason": "user_stop"},
        "acoustic_clarification": {
            "status": "incomplete",
            "mode": mode,
            "severity": "hard",
            "utterances": [target],
            "user_utterances": [target],
            "assistant_observations": [],
        },
    }


@pytest.mark.parametrize("mode", ["dropchunk", "dns"])
def test_delivered_damage_is_valid_even_when_interruption_cancels_clean_tail(mode):
    result = assess_report(scenario(), recorded_report(mode), 0.1)
    acoustic = result["acoustic_clarification"]["assessment"]
    assert acoustic["input_validity"] == "valid"
    assert acoustic["conversation_status"] == "complete"
    assert acoustic["verdict"] == "inconclusive"
    assert acoustic["target"]["cancelled_samples"] == [300, 500]
    assert acoustic["summary"]["valid_input_denominator"] == 1


def evidence_report(question_start=2.5, question_end=2.9):
    report = recorded_report()
    transcripts = [
        {
            "id": "u0",
            "role": "user",
            "start_seconds": 0,
            "end_seconds": 3,
            "transcript": "Go to [unclear] then book a quiet hotel.",
            "audio_file": "target.wav",
        },
        {
            "id": "a0",
            "role": "assistant",
            "start_seconds": question_start,
            "end_seconds": question_end,
            "transcript": "Which city?",
            "audio_file": "question.wav",
        },
        {
            "id": "u1",
            "role": "user",
            "start_seconds": 6,
            "end_seconds": 7,
            "transcript": "Paris.",
            "audio_file": "repair.wav",
        },
        {
            "id": "a1",
            "role": "assistant",
            "start_seconds": 8,
            "end_seconds": 9,
            "transcript": "I will look for a quiet hotel in Paris.",
            "audio_file": "followup.wav",
        },
    ]
    report["acoustic_clarification"]["evidence"] = {
        "transcripts": transcripts,
        "context_prefixes": {
            "a0": [
                {
                    "id": "u0-prefix-a0",
                    "role": "user",
                    "start_seconds": 0,
                    "end_seconds": question_start,
                    "transcript": "Go to [unclear]",
                    "audio_file": "target-prefix.wav",
                }
            ]
        },
    }
    return report


def fake_judge(
    monkeypatch, *, adequacy="appropriate", repair="completed", no_query=False
):
    import json

    from tau2.data_model.message import AssistantMessage

    calls = []

    def generate(**kwargs):
        stage = kwargs["call_name"]
        payload = json.loads(kwargs["messages"][0].content.split("\n")[-1])
        calls.append((stage, payload))
        if stage == "acoustic_question_discovery":
            content = {
                "request": None
                if no_query
                else {"segment_id": "a0", "quote": "Which city?"},
                "answer": {
                    "segment_id": "a1",
                    "quote": "I will look for a quiet hotel in Paris.",
                },
                "no_query_certain": True,
                "reason": "Recorded speech reviewed.",
            }
        elif stage == "acoustic_request_judge":
            assert "Paris." not in json.dumps(payload.get("context", []))
            assert "quiet hotel" not in json.dumps(payload.get("context", []))
            content = {
                "adequacy": adequacy,
                "reason": "Asks for the unclear city.",
                "evidence": [{"id": "u0-prefix-a0", "quote": "[unclear]"}],
            }
        elif stage == "acoustic_repair_judge":
            content = {
                "status": repair,
                "reason": "Clear repeat and correct subsequent usage.",
                "evidence": [{"id": "a0", "quote": "Which city?"}]
                if repair == "missing_user_reply"
                else [
                    {"id": "u1", "quote": "Paris."},
                    {"id": "a1", "quote": "quiet hotel in Paris"},
                ],
            }
        else:
            content = {
                "result": "supported_inference",
                "reason": "Earlier delivered context identifies city.",
                "evidence": [{"id": "u0-prefix-a0", "quote": "[unclear]"}],
            }
        return AssistantMessage(role="assistant", content=json.dumps(content))

    monkeypatch.setattr("tau2.utils.llm_utils.generate", generate)
    return calls


@pytest.mark.parametrize(
    "start,end,category",
    [(2.5, 2.9, "during_user"), (5.2, 5.6, "after_user"), (4.9, 5.2, "uncertain")],
)
def test_full_clarification_has_same_semantics_at_both_timings(
    monkeypatch, start, end, category
):
    from tau2.user.duplex_evaluation import judge_report

    calls = fake_judge(monkeypatch)
    report = evidence_report(start, end)
    result = judge_report(
        scenario(), assess_report(scenario(), report, 0.1), "offline-judge"
    )
    row = result["acoustic_clarification"]["assessment"]
    assert row["verdict"] == "pass"
    assert row["question"]["adequacy"] == "appropriate"
    assert row["repair"]["status"] == "completed"
    assert row["timing"]["category"] == category
    assert row["timing"]["question_onset_bounds_seconds"] == [start, end]
    assert row["summary"]["semantic_evaluated_denominator"] == 1
    request_payload = next(
        payload for stage, payload in calls if stage == "acoustic_request_judge"
    )
    assert "script_text" not in request_payload
    assert "Private intent" not in str(request_payload)
    assert "repair.wav" not in str(request_payload)
    assert "future" not in request_payload


@pytest.mark.parametrize(
    "case",
    [
        "alignment_failed",
        "unheard",
        "timeout",
        "no_reply",
        "wrong_question",
        "judge_failed",
        "before_target",
        "no_query",
    ],
)
def test_incomplete_and_nonclarification_outcomes_remain_explicit(monkeypatch, case):
    from tau2.user.duplex_evaluation import judge_report

    report = evidence_report()
    calls = fake_judge(
        monkeypatch,
        adequacy="misdirected" if case == "wrong_question" else "appropriate",
        repair="missing_user_reply" if case == "no_reply" else "completed",
        no_query=case == "no_query",
    )
    if case == "alignment_failed":
        report["acoustic_clarification"]["utterances"][0]["status"] = "failed"
    elif case == "unheard":
        report["acoustic_clarification"]["utterances"][0]["delivery"] = []
    elif case == "timeout":
        report["conversation"] = {
            "status": "incomplete",
            "termination_reason": "max_steps",
        }
    elif case == "no_reply":
        report["acoustic_clarification"]["evidence"]["transcripts"] = report[
            "acoustic_clarification"
        ]["evidence"]["transcripts"][:2]
    elif case == "judge_failed":

        def broken(**kwargs):
            raise RuntimeError("external model unavailable")

        monkeypatch.setattr("tau2.utils.llm_utils.generate", broken)
    elif case == "before_target":
        report["acoustic_clarification"]["evidence"]["transcripts"][1].update(
            start_seconds=0.1, end_seconds=0.2
        )
    elif case == "no_query":
        # No query is not automatically a success; require supported inference proof.
        report["acoustic_clarification"]["evidence"]["context_prefixes"]["a1"] = report[
            "acoustic_clarification"
        ]["evidence"]["context_prefixes"]["a0"]
    result = judge_report(
        scenario(), assess_report(scenario(), report, 0.1), "offline-judge"
    )
    row = result["acoustic_clarification"]["assessment"]
    if case in {"alignment_failed", "unheard"}:
        assert row["input_validity"] == "invalid" and row["verdict"] == "not_evaluated"
        assert calls == []
    elif case == "wrong_question":
        assert row["verdict"] == "fail"
    elif case == "no_query":
        assert row["question"]["presence"] == "absent"
        assert row["no_query"]["result"] == "supported_inference"
        assert row["clarification_completed"] is False
    else:
        assert row["verdict"] == "inconclusive"
    if case == "no_reply":
        assert row["question"]["adequacy"] == "appropriate"
        assert row["repair"]["status"] == "missing_user_reply"
    if case == "judge_failed":
        assert row["summary"]["judge_failure"] == 1
    if case == "before_target":
        assert row["question"]["delivered_modified_before_request_samples"] == 0
        assert len(calls) == 1


def test_final_audio_evidence_crops_received_prefix_and_offline_replay_never_transcribes(
    monkeypatch, tmp_path
):
    import json
    import wave

    import numpy as np

    from experiments.tau_voice.duplex_bench import build_parser
    from tau2.data_model.voice import TranscriptionResult
    from tau2.user.duplex_evaluation import finalize_acoustic_evidence

    def wav(relative, samples):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(100)
            stream.writeframes(np.asarray(samples, dtype="<i2").tobytes())

    report = recorded_report()
    target = report["acoustic_clarification"]["utterances"][0]
    # Last emitted second is never consumed; actual receiver input codec has different samples.
    target["delivery"] = [
        {
            "start_seconds": i,
            "end_seconds": i + 1,
            "sample_start": 100 * i,
            "sample_end": 100 * (i + 1),
            "padding_samples": 0,
        }
        for i in range(3)
    ]
    wav("target.wav", [1] * 100 + [2] * 100 + [99] * 100)
    wav("agent_audio/agent_input.wav", [11] * 100 + [22] * 100)
    wav("question.wav", [33] * 100)
    report["acoustic_clarification"]["assistant_observations"] = [
        {
            "start_seconds": 2,
            "end_seconds": 3,
            "audible": True,
            "utterance_ids": ["question"],
            "audio_file": "question.wav",
        }
    ]
    calls = []

    def asr(audio, config):
        values = np.frombuffer(audio.data, dtype="<i2")
        calls.append(values.copy())
        return TranscriptionResult(
            transcript="Which city?" if 33 in values else "heard prefix"
        )

    monkeypatch.setattr("tau2.voice.transcription.transcribe.transcribe_audio", asr)
    final = finalize_acoustic_evidence(
        scenario(),
        report,
        tmp_path,
        1.0,
        [{"tick_id": i, "agent_chunk": {}} for i in range(3)],
    )
    target = final["acoustic_clarification"]["utterances"][0]
    assert target["received_modified_samples"] == 60
    assert target["delivery"][-1]["received"] is False
    evidence = final["acoustic_clarification"]["evidence"]
    user = next(row for row in evidence["transcripts"] if row["role"] == "user")
    assert user["start_seconds"] == 1.0
    assert user["end_seconds"] == 3.0
    assert user["source"] == "decoded_provider_input"
    prefix = evidence["context_prefixes"]["assistant-0001"][0]
    assert prefix["end_seconds"] == 2.0
    assert any(np.array_equal(item, [11] * 100) for item in calls)
    assert all(99 not in item and 1 not in item for item in calls)

    (tmp_path / "scenario.json").write_text(scenario().model_dump_json())
    (tmp_path / "events.json").write_text(json.dumps(final))
    (tmp_path / "run_config.json").write_text(
        json.dumps({"audio_native_config": {"tick_duration_seconds": 1.0}})
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline assessment must not call ASR")

    monkeypatch.setattr(
        "tau2.voice.transcription.transcribe.transcribe_audio", forbidden
    )
    args = build_parser().parse_args(["assess", "--output", str(tmp_path)])
    assert args.func(args) == 0
    assessment = json.loads((tmp_path / "assessment.json").read_text())
    assert (
        assessment["acoustic_clarification"]["assessment"]["input_validity"] == "valid"
    )


@pytest.mark.parametrize(
    "problem", ["missing_asr", "fabricated_quote", "backchannel", "receiving_delay"]
)
def test_evidence_failures_and_semantic_onset_do_not_become_success(
    monkeypatch, problem
):
    from tau2.user.duplex_evaluation import judge_report

    report = evidence_report()
    fake_judge(monkeypatch)
    acoustic = report["acoustic_clarification"]
    if problem == "missing_asr":
        acoustic["evidence"]["errors"] = [
            {"stage": "assistant_audio", "segment_id": "missing-turn"}
        ]
    elif problem == "fabricated_quote":
        acoustic["evidence"]["context_prefixes"]["a0"][0]["transcript"] = (
            "Completely different content."
        )
    elif problem == "backchannel":
        acoustic["evidence"]["transcripts"].insert(
            0,
            {
                "id": "backchannel",
                "role": "assistant",
                "start_seconds": 0.2,
                "end_seconds": 0.3,
                "transcript": "Mm-hm.",
            },
        )
    elif problem == "receiving_delay":
        acoustic["utterances"][0]["delivery"][0]["received_start_seconds"] = 2.0
    row = judge_report(
        scenario(), assess_report(scenario(), report, 0.1), "offline-judge"
    )["acoustic_clarification"]["assessment"]
    if problem == "backchannel":
        assert row["verdict"] == "pass"
        assert row["timing"]["question_onset_bounds_seconds"] == [2.5, 2.9]
    else:
        assert row["verdict"] == "inconclusive"
    if problem == "missing_asr":
        assert row["evidence_status"] == "insufficient"
    if problem == "fabricated_quote":
        assert row["summary"]["judge_failure"] == 1
    if problem == "receiving_delay":
        assert row["question"]["delivered_modified_before_request_samples"] == 0


def test_mechanical_summary_keeps_missing_evidence_and_excluded_denominators_explicit():
    report = recorded_report()
    report["acoustic_clarification"]["utterances"][0]["delivery"] = []
    row = assess_report(scenario(), report, 0.1)["acoustic_clarification"]["assessment"]
    assert row["summary"]["valid_input_denominator"] == 0
    assert row["summary"]["missing_evidence"] == 1
    assert row["summary"]["unknown_question"] == 1
    assert row["exclusions"] == ["target_modification_not_consumed"]
    assert all(
        group["count"] == 0 for group in row["summary"]["timing_groups"].values()
    )
