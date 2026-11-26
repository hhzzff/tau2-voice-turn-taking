"""Cross-feature regressions for the local voice and remote acoustic paths."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.tau_voice import duplex_bench
from tau2.data_model.audio import (
    PCM_SAMPLE_RATE,
    AudioData,
    AudioEncoding,
    AudioFormat,
    audio_bytes_to_string,
)
from tau2.data_model.message import AssistantMessage
from tau2.data_model.simulation import AudioNativeConfig
from tau2.user.duplex_events import DuplexDecision, DuplexScenario


def assistant_chunk(user, speaking=False):
    """Supply actual PCM so the shared activity detector is exercised."""
    samples = np.full(user.chunk_size, 1600 if speaking else 0, dtype="<i2")
    return AssistantMessage(
        role="assistant",
        content="I see." if speaking else "",
        is_audio=True,
        audio_content=audio_bytes_to_string(samples.tobytes()),
        audio_format=AudioFormat(
            encoding=AudioEncoding.PCM_S16LE, sample_rate=PCM_SAMPLE_RATE
        ),
        contains_speech=speaking,
        utterance_ids=["reply"] if speaking else None,
    )


def test_standalone_acoustic_task_and_ordered_combination_keep_distinct_coverage():
    standalone = DuplexScenario(acoustic_clarification={"mode": "dropchunk"})
    assert standalone.event_ids == []
    combined = DuplexScenario(
        selected_events=["follow_up"],
        acoustic_clarification={"mode": "dropchunk"},
        event_order=[
            "interruption.follow_up",
            "proactive.backchannel",
            "proactive.completion",
        ],
        event_guidance={"proactive.completion": "Use shared destination facts."},
    )
    restored = DuplexScenario.model_validate_json(combined.model_dump_json())
    assert restored.event_ids == combined.event_order
    assert restored.acoustic_clarification.mode == "dropchunk"


def test_one_report_keeps_integrated_event_and_acoustic_judgments(monkeypatch):
    from tau2.user.duplex_evaluation import assess_report, judge_report

    prior = {
        "id": "u1",
        "role": "user",
        "transcript": "I enjoy birdwatching.",
        "start_seconds": 0,
        "end_seconds": 1,
    }
    answer = {
        "id": "a1",
        "role": "assistant",
        "transcript": "Try a pocket bird guide.",
        "start_seconds": 2,
        "end_seconds": 3,
    }
    plan = DuplexScenario(
        selected_events=["rejection.noise"],
        include_proactive_opportunities=False,
        acoustic_clarification={"mode": "dropchunk"},
    )
    report = {
        "status": "complete",
        "conversation": {"status": "complete"},
        "events": [
            {
                "event_id": "rejection.noise",
                "status": "observed",
                "start_seconds": 5,
                "end_seconds": 6,
                "marker_seconds": 6,
                "observation_end_seconds": 8,
                "assistant_was_speaking": False,
                "stimulus": {"text": "", "noise": "door_knock"},
                "observations": [{"seconds": 7, "speaking": False, "text": ""}],
            }
        ],
        "acoustic_clarification": {
            "utterances": [
                {
                    "status": "delivered",
                    "target_samples": [10, 20],
                    "alignment_response": {"words": [{}]},
                    "sample_rate": 100,
                    "modified_intervals_samples": [[11, 19]],
                    "delivery": [
                        {
                            "sample_start": 0,
                            "sample_end": 100,
                            "start_seconds": 0,
                            "end_seconds": 1,
                            "received": True,
                        }
                    ],
                }
            ],
            "evidence": {
                "transcripts": [prior, answer],
                "context_prefixes": {"a1": [prior]},
                "errors": [],
            },
        },
    }
    calls = []

    def generate(**kwargs):
        calls.append(kwargs["call_name"])
        if kwargs["call_name"] == "duplex_event_judge":
            result = {"verdict": "inconclusive", "reason": "Stimulus needs review."}
        else:
            payload = json.loads(kwargs["messages"][0].content.split("\n")[-1])
            if payload["response_schema"]["title"] == "Discovery":
                result = {
                    "request": None,
                    "answer": {"segment_id": "a1", "quote": answer["transcript"]},
                    "no_query_certain": True,
                    "reason": "No repair question.",
                }
            else:
                result = {
                    "response_relevance": "target_relevant",
                    "result": "supported_inference",
                    "reason": "The interest is present in delivered context.",
                    "evidence": [{"id": "u1", "quote": "birdwatching"}],
                }
        return AssistantMessage(role="assistant", content=json.dumps(result))

    monkeypatch.setattr("tau2.utils.llm_utils.generate", generate)
    assessment = judge_report(plan, assess_report(plan, report, 0.1), "test-judge")
    assert assessment["events"][0]["verdict"] == "inconclusive"
    assert assessment["summary"]["inconclusive"] == 1
    assert assessment["acoustic_clarification"]["assessment"]["verdict"] == "pass"
    assert "duplex_event_judge" in calls and len(calls) >= 3


@pytest.mark.parametrize("provider", ["volcengine", "openai", "elevenlabs"])
def test_explicit_tts_provider_preserves_pause_delivery_and_synthesis_evidence(
    monkeypatch, tmp_path, provider
):
    calls = []
    prefix = np.full(PCM_SAMPLE_RATE, 1800, dtype="<i2")
    pause = np.zeros(round(PCM_SAMPLE_RATE * 0.75), dtype="<i2")

    def synthesize(text, actual_provider, config):
        calls.append((text, actual_provider, getattr(config, "ssml", None)))
        samples = (
            np.concatenate([prefix, pause, prefix])
            if getattr(config, "ssml", None)
            else prefix
        )
        return AudioData(
            data=samples.tobytes(),
            format=AudioFormat(
                encoding=AudioEncoding.PCM_S16LE, sample_rate=PCM_SAMPLE_RATE
            ),
        )

    decision = DuplexDecision(
        action="trigger_event",
        event_id="rejection.pause",
        text="Let me think about",
        continuation="the packing order.",
        pause_seconds=0.75,
        reason="An ordinary thinking pause with several possible continuations.",
    )
    monkeypatch.setattr("tau2.user.duplex_scenario_user.synthesize_voice", synthesize)
    monkeypatch.setattr(
        "tau2.user.duplex_scenario_user.generate",
        lambda **kwargs: AssistantMessage(
            role="assistant", content=decision.model_dump_json()
        ),
    )
    user = duplex_bench.build_dynamic_user(
        DuplexScenario(
            selected_events=["rejection.pause"],
            include_proactive_opportunities=False,
            assistant_observation="proportional",
        ),
        tmp_path,
        AudioNativeConfig(),
        "primary-voice",
        "background-voice",
        "test-tts",
        tts_provider=provider,
    )
    state = user.get_init_state()
    for _ in range(28):
        user.get_next_chunk(state, assistant_chunk(user))
    event = user.save_report()["events"][0]
    assert event["event_id"] == "rejection.pause"
    assert event["pause_audio_evidence"]["duration_seconds"] == pytest.approx(0.75)
    assert event["marker_seconds"] == pytest.approx(1.0)
    assert event["end_seconds"] == pytest.approx(2.75)
    assert all(item[1] == provider for item in calls)
    assert len(calls) == (1 if provider == "volcengine" else 2)
    if provider == "volcengine":
        assert '<break time="750ms"/>' in calls[0][2]
    evidence = json.loads((tmp_path / "synthesis.json").read_text(encoding="utf-8"))
    assert all(row["provider"] == provider for row in evidence["utterances"])


def test_acoustic_memory_observation_keeps_the_complete_answer(monkeypatch, tmp_path):
    model_calls = []

    def generate(**kwargs):
        payload = json.loads(kwargs["messages"][0].content.split("\n")[-1])
        model_calls.append(payload["elapsed_seconds"])
        return AssistantMessage(
            role="assistant",
            content=json.dumps({"action": "finish", "reason": "The answer ended."}),
        )

    monkeypatch.setattr("tau2.user.duplex_scenario_user.generate", generate)
    user = duplex_bench.build_dynamic_user(
        DuplexScenario(
            selected_events=["turn_taking.handoff"],
            include_proactive_opportunities=False,
            acoustic_clarification={"mode": "dropchunk"},
            assistant_observation="proportional",
        ),
        tmp_path,
        AudioNativeConfig(),
        "nova",
        "onyx",
        "test-tts",
    )
    state = user.get_init_state()
    user.controller.begin(
        DuplexDecision(
            action="trigger_event",
            event_id="turn_taking.handoff",
            text="Please use the earlier constraints.",
            observation_seconds=2,
            reason="Recall the previously established facts.",
        ),
        now=0,
        duration=1,
        marker=1,
        assistant_speaking=False,
    )
    for tick in range(95):
        user.get_next_chunk(state, assistant_chunk(user, 10 <= tick < 75))
        if tick < 75:
            assert not model_calls
    assert model_calls and min(model_calls) >= 8.4
    completion = user.controller.records[0]["memory_response_completion"]
    assert completion["status"] == "completed"
    assert user.controller.records[0]["status"] == "observed"


def test_acoustic_backchannel_account_keeps_floor_during_brief_acknowledgement(
    monkeypatch, tmp_path
):
    text = "I am preparing the food for Riverside."
    decision = DuplexDecision(
        action="trigger_event",
        event_id="proactive.backchannel",
        text=text,
        keywords=[{"text": "Riverside", "occurrence": 0}],
        reason="Describe the ongoing preparation while retaining the floor.",
    )
    monkeypatch.setattr(
        "tau2.user.duplex_scenario_user.generate",
        lambda **kwargs: AssistantMessage(
            role="assistant", content=decision.model_dump_json()
        ),
    )
    monkeypatch.setattr(
        "tau2.user.duplex_scenario_user.synthesize_voice",
        lambda *args: AudioData(
            data=np.full(PCM_SAMPLE_RATE * 10, 1600, dtype="<i2").tobytes(),
            format=AudioFormat(
                encoding=AudioEncoding.PCM_S16LE, sample_rate=PCM_SAMPLE_RATE
            ),
        ),
    )
    monkeypatch.setattr(
        "tau2.user.acoustic_clarification.forced_alignment",
        lambda *args: {
            "characters": [
                {"text": char, "start": i * 0.1, "end": (i + 1) * 0.1}
                for i, char in enumerate(text)
            ]
        },
    )
    user = duplex_bench.build_dynamic_user(
        DuplexScenario(
            selected_events=["proactive.backchannel"],
            include_proactive_opportunities=False,
            acoustic_clarification={"mode": "dropchunk"},
            assistant_observation="proportional",
        ),
        tmp_path,
        AudioNativeConfig(),
        "nova",
        "onyx",
        "test-tts",
    )
    state, outputs = user.get_init_state(), []
    for tick in range(100):
        output, state = user.get_next_chunk(
            state, assistant_chunk(user, 20 <= tick < 24)
        )
        outputs.append(output)
    report = user.save_report()
    target = report["acoustic_clarification"]["utterances"][0]
    assert target["status"] == "delivered"
    assert target["delivered_samples"] == PCM_SAMPLE_RATE * 10
    assert target["cancelled_samples"] == []
    assert not any(row.turn_taking_action.action == "stop_talking" for row in outputs)
    assert not report["events"][0]["user_stopped_for_agent"]


@pytest.mark.parametrize(
    "options,expected_provider,expected_asr",
    [
        ([], "volcengine", "gemini-3.8-flash"),
        (["--tts-provider", "openai"], "openai", "gemini-3.8-flash"),
        (["--byted-gateway"], "elevenlabs", "scribe_v1"),
        (["--acoustic-clarification", "dropchunk"], "openai", "gemini-3.8-flash"),
    ],
)
def test_runner_keeps_provider_asr_and_partial_evidence_on_failure(
    monkeypatch, tmp_path, options, expected_provider, expected_asr
):
    scenario_file = tmp_path / "scenario.json"
    scenario_file.write_text(
        DuplexScenario(
            selected_events=["follow_up"],
            topic="Plan a picnic",
            user_context="Use the earlier budget.",
            transcription_model="gemini-3.8-flash",
        ).model_dump_json(),
        encoding="utf-8",
    )
    output = tmp_path / "run"
    users, cleanups, audio_exports = [], [], []
    real_builder = duplex_bench.build_dynamic_user

    def build_user(*args, **kwargs):
        user = real_builder(*args, **kwargs)
        users.append(user)
        return user

    class InterruptedOrchestrator:
        def __init__(self, **kwargs):
            self.ticks = [
                SimpleNamespace(tick_id=0, model_dump=lambda **kw: {"tick_id": 0})
            ]

        def run(self):
            raise RuntimeError("controlled interruption")

        def _cleanup(self):
            cleanups.append(True)

    monkeypatch.setattr(duplex_bench, "build_dynamic_user", build_user)
    monkeypatch.setattr(
        "tau2.runner.build.build_agent",
        lambda *a, **kw: SimpleNamespace(input_audio_conversions=[]),
    )
    monkeypatch.setattr(
        "tau2.orchestrator.full_duplex_orchestrator.FullDuplexOrchestrator",
        InterruptedOrchestrator,
    )
    monkeypatch.setattr(
        "tau2.voice.synthesis.conversation_builder.generate_simulation_audio",
        lambda run, directory: audio_exports.append(len(run.ticks)),
    )
    args = duplex_bench.build_parser().parse_args(
        [
            "run",
            "--scenario",
            str(scenario_file),
            "--output",
            str(output),
            "--env-file",
            str(tmp_path / "unused.env"),
            "--no-semantic-judge",
            *options,
        ]
    )
    with pytest.raises(RuntimeError, match="controlled interruption"):
        args.func(args)
    user = users[0]
    assert user.voice_settings.synthesis_config.provider == expected_provider
    assert user.perception.config.model == expected_asr
    assert cleanups == [True] and audio_exports == [1]
    assert (output / "ticks.json").exists()
    assessment = json.loads((output / "assessment.json").read_text(encoding="utf-8"))
    assert assessment["summary"]["total"] == 3
    assert assessment["summary"]["not_evaluated"] == 3
    if "--acoustic-clarification" in options:
        assert (output / "agent_input_conversion.json").exists()
        assert (
            assessment["acoustic_clarification"]["assessment"]["input_validity"]
            != "valid"
        )
