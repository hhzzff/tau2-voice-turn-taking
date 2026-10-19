#!/usr/bin/env python3
"""LLM judge workflow for contextual turn-taking interruption events."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from tau2.config import DEFAULT_LLM_EVAL_USER_SIMULATOR
from tau2.data_model.message import SystemMessage, Tick, UserMessage
from tau2.utils.llm_utils import extract_json_from_llm_response, generate

DEFAULT_EVENTS_CSV = Path(
    "data/simulations/interruption_type_experiment_notools/events.csv"
)
DEFAULT_OUTPUT = Path(
    "data/simulations/interruption_type_experiment_notools/contextual_judge_example.json"
)

FREE_DISCUSSION_POLICY_MARKERS = (
    "turn-taking and interruption test",
    "not a customer-service benchmark",
    "natural voice conversation",
)

EVENT_ASSISTANT_QUALITY_PROMPT = """
You are an LLM judge for model response quality in a full-duplex voice
conversation.

Your task is to evaluate one duplex/interruption event. Evaluate ONLY the
assistant/model behavior after the user speaks over or interrupts it.

Do NOT evaluate whether the user simulator sounded natural. Do NOT score user
quality, user timing, user topic choice, task success, tool calls, or business
policy compliance except where they directly affect the assistant response.

Focus on the local window around the event:
- what the assistant was saying before overlap
- the user interruption as input context only
- the assistant response immediately after the interruption
- whether the assistant's response is relevant, correct, and conversationally
  coherent

The transcript may contain streaming chunk artifacts such as split words or
extra spaces. Mentally normalize those artifacts before judging the assistant.

Scoring rubric:
- 1 = poor / clearly wrong
- 2 = weak
- 3 = acceptable
- 4 = good
- 5 = excellent

Judge these model-only dimensions:
- assistant_response_correctness
- assistant_conversational_flow
- assistant_response_relevance

Definitions:
- assistant_response_correctness: whether the answer was reasonable and contextually grounded.
- assistant_conversational_flow: whether the response felt coherent and natural in a duplex call.
- assistant_response_relevance: whether the assistant addressed the user's interruption.

Decision rule:
- pass: the assistant answered the interruption correctly, relevantly, and naturally.
- borderline: usable response, but one important model behavior is weak.
- fail: the assistant ignored the interruption, answered poorly, or made the conversation incoherent.

Return JSON only. No markdown, no prose outside JSON.
""".strip()

CONVERSATION_USER_NATURALNESS_PROMPT = """
You are an LLM judge for user simulator naturalness in a full-duplex voice
conversation.

Your task is to evaluate the USER SIMULATOR across the complete conversation
after the conversation has ended.

Evaluate ONLY user behavior. Do NOT evaluate assistant/model reply quality,
assistant correctness, assistant task success, tool calls, or business policy.

Focus on:
- whether the user sounded like a natural person in a voice conversation
- whether the user maintained a concrete main topic
- whether interruptions or side questions appeared at plausible moments
- whether the user returned to the main topic after side questions
- whether the user avoided tool/query/customer-service framing when the topic is free discussion
- whether the final conversation felt coherent rather than scripted or empty

The transcript may contain streaming chunk artifacts such as split words or
extra spaces. Mentally normalize those artifacts before judging the user.

Scoring rubric:
- 1 = poor / clearly unnatural
- 2 = weak
- 3 = acceptable
- 4 = good
- 5 = excellent

Judge these user-only dimensions:
- overall_user_naturalness

Definitions:
- overall_user_naturalness: holistic user simulator naturalness.

Decision rule:
- pass: the user is natural, topic-grounded, and the conversation progresses plausibly.
- borderline: usable conversation, but one important user behavior is weak.
- fail: the user is unnatural, scripted, off-topic, query/tool-framed, or incoherent.

Return JSON only. No markdown, no prose outside JSON.
""".strip()


def _as_bool(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return Path.cwd() / resolved


def read_events(events_csv: Path) -> list[dict[str, Any]]:
    """Read interruption analysis events from CSV."""
    with events_csv.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_ticks(simulation_file: Path) -> list[Tick]:
    """Load full-duplex ticks from a saved simulation JSON file."""
    with simulation_file.open(encoding="utf-8-sig") as f:
        simulation = json.load(f)
    return [Tick.model_validate(tick) for tick in simulation.get("ticks", [])]


def load_user_instructions(results_dir: Path, task_id: str) -> str:
    """Load judge-safe user instructions from the run's results.json.

    Free discussion runs are launched on top of a legacy tau-bench domain, so
    the saved task still contains an unrelated customer-service scenario. In
    that mode, use the run policy/topic plan as the governing context and avoid
    leaking the obsolete task into the judge prompt.
    """
    results_path = results_dir / "results.json"
    if not results_path.exists():
        return ""
    with results_path.open(encoding="utf-8-sig") as f:
        results = json.load(f)
    policy = results.get("info", {}).get("environment_info", {}).get("policy", "")
    if any(marker in policy for marker in FREE_DISCUSSION_POLICY_MARKERS):
        return (
            "Free discussion conversation. Use the Topic Plan and the full "
            "conversation context as the governing instructions. Ignore any "
            "legacy domain task data from the underlying benchmark."
        )
    for task in results.get("tasks", []):
        if str(task.get("id")) == str(task_id):
            return json.dumps(task.get("user_scenario", {}), ensure_ascii=False)
    return ""


def load_topic_plan(results_dir: Path) -> str:
    """Load the contextual topic plan for a run, if available."""
    txt_path = results_dir / "topic_plan.txt"
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8")

    json_path = results_dir / "topic_plan.json"
    if json_path.exists():
        with json_path.open(encoding="utf-8") as f:
            return json.dumps(json.load(f), ensure_ascii=False)

    return ""


def _chunk_has_text(chunk: Any) -> bool:
    if not chunk:
        return False
    text = (chunk.content or chunk.audio_script_gold or "").strip()
    return bool(chunk.contains_speech and text)


def _chunk_text(chunk: Any) -> str:
    if not chunk:
        return ""
    return (chunk.content or chunk.audio_script_gold or "").strip()


def _append_unique(parts: list[str], text: str) -> None:
    if text and (not parts or parts[-1] != text):
        parts.append(text)


def _format_segment(
    *,
    start_tick: int,
    end_tick: int,
    tick_duration_sec: float,
    agent_parts: list[str],
    user_parts: list[str],
) -> str:
    start_sec = start_tick * tick_duration_sec
    end_sec = (end_tick + 1) * tick_duration_sec
    lines = [f"[ticks {start_tick}-{end_tick}, {start_sec:.1f}s-{end_sec:.1f}s]"]
    if agent_parts:
        lines.append(f"Agent: {' '.join(agent_parts)}")
    if user_parts:
        lines.append(f"User: {' '.join(user_parts)}")
    return "\n".join(lines)


def format_conversation_window(
    ticks: list[Tick],
    *,
    tick_duration_sec: float,
) -> str:
    """Format a local tick window as compact plain text for the LLM judge."""
    segments: list[str] = []
    current_key: tuple[bool, bool] | None = None
    current_start: int | None = None
    current_end: int | None = None
    agent_parts: list[str] = []
    user_parts: list[str] = []

    def flush() -> None:
        nonlocal current_start, current_end, agent_parts, user_parts
        if current_start is None or current_end is None:
            return
        if not agent_parts and not user_parts:
            return
        segments.append(
            _format_segment(
                start_tick=current_start,
                end_tick=current_end,
                tick_duration_sec=tick_duration_sec,
                agent_parts=agent_parts,
                user_parts=user_parts,
            )
        )

    for tick in ticks:
        agent_active = _chunk_has_text(tick.agent_chunk)
        user_active = _chunk_has_text(tick.user_chunk)
        key = (agent_active, user_active)
        if key == (False, False):
            flush()
            current_key = None
            current_start = None
            current_end = None
            agent_parts = []
            user_parts = []
            continue

        if current_key != key:
            flush()
            current_key = key
            current_start = tick.tick_id
            agent_parts = []
            user_parts = []

        current_end = tick.tick_id
        _append_unique(agent_parts, _chunk_text(tick.agent_chunk))
        _append_unique(user_parts, _chunk_text(tick.user_chunk))

    flush()
    return "\n\n".join(segments)


def build_sample(
    event: dict[str, Any],
    *,
    context_before_sec: float,
    context_after_sec: float,
) -> dict[str, Any]:
    """Build one structured judge sample from an event row."""
    simulation_file = _resolve_path(event["simulation_file"])
    results_dir = _resolve_path(event["results_dir"])
    ticks = load_ticks(simulation_file)

    start_tick = int(event["interruption_start_tick"])
    user_end_tick = int(event["user_end_tick"])
    tick_duration_sec = float(event.get("tick_duration_sec") or 0.2)
    before_ticks = int(context_before_sec / tick_duration_sec)
    after_ticks = int(context_after_sec / tick_duration_sec)
    window_start = max(0, start_tick - before_ticks)
    window_end = min(len(ticks), max(user_end_tick, start_tick) + after_ticks)
    conversation_window = format_conversation_window(
        ticks[window_start:window_end],
        tick_duration_sec=tick_duration_sec,
    )

    sample_id = f"{event['simulation_id']}:{start_tick}"
    return {
        "sample_id": sample_id,
        "simulation_id": event["simulation_id"],
        "task_id": event["task_id"],
        "interruption_type": event["interruption_type"],
        "user_instructions": load_user_instructions(results_dir, event["task_id"]),
        "topic_plan": load_topic_plan(results_dir),
        "conversation_window": conversation_window,
        "metadata": {
            "interruption_start_sec": _as_float(event["interruption_start_sec"]),
            "interruption_success": _as_bool(event["interruption_success"]),
            "post_interrupt_stop_latency_sec": _as_float(
                event["post_interrupt_stop_latency_sec"]
            ),
            "post_interrupt_answer_success": _as_bool(
                event["post_interrupt_answer_success"]
            ),
            "agent_response_latency_sec": _as_float(
                event["agent_response_latency_sec"]
            ),
            "agent_yield_tick": _as_int(event["agent_yield_tick"]),
            "agent_response_start_tick": _as_int(event["agent_response_start_tick"]),
        },
        "event": {
            "user_transcript": event["user_transcript"],
            "agent_overlap_transcript": event["agent_overlap_transcript"],
            "agent_response_transcript": event["agent_response_transcript"],
        },
    }


def build_conversation_sample(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a conversation-level judge sample from all selected events in one simulation."""
    first_event = events[0]
    simulation_file = _resolve_path(first_event["simulation_file"])
    results_dir = _resolve_path(first_event["results_dir"])
    ticks = load_ticks(simulation_file)
    tick_duration_sec = float(first_event.get("tick_duration_sec") or 0.2)
    conversation_window = format_conversation_window(
        ticks,
        tick_duration_sec=tick_duration_sec,
    )

    return {
        "simulation_id": first_event["simulation_id"],
        "task_id": first_event["task_id"],
        "user_instructions": load_user_instructions(
            results_dir, first_event["task_id"]
        ),
        "topic_plan": load_topic_plan(results_dir),
        "conversation_window": conversation_window,
        "events": [
            {
                "sample_id": f"{event['simulation_id']}:{event['interruption_start_tick']}",
                "interruption_type": event["interruption_type"],
                "interruption_start_sec": _as_float(event["interruption_start_sec"]),
                "user_transcript": event["user_transcript"],
                "agent_overlap_transcript": event["agent_overlap_transcript"],
                "agent_response_transcript": event["agent_response_transcript"],
            }
            for event in events
        ],
    }


def build_user_prompt(sample: dict[str, Any]) -> str:
    """Build the per-event user prompt for assistant quality judging."""
    return f"""
Sample ID: {sample["sample_id"]}
Interruption Type: {sample["interruption_type"]}

Scenario / User Instructions:
{sample["user_instructions"]}

Topic Plan:
{sample["topic_plan"]}

Conversation Window:
{sample["conversation_window"]}

Relevant Metadata:
{json.dumps(sample["metadata"], ensure_ascii=False, indent=2)}

Raw Event Transcript:
{json.dumps(sample["event"], ensure_ascii=False, indent=2)}

Return JSON in this shape:
{{
  "sample_id": "{sample["sample_id"]}",
  "interruption_type": "{sample["interruption_type"]}",
  "scores": {{
    "assistant_response_correctness": 1,
    "assistant_conversational_flow": 1,
    "assistant_response_relevance": 1
  }},
  "overall_decision": "pass | borderline | fail",
  "failure_modes": [
    "short label for each issue"
  ],
  "notes": "brief explanation"
}}
""".strip()


def build_conversation_user_naturalness_prompt(sample: dict[str, Any]) -> str:
    """Build one conversation-level prompt for user naturalness judging."""
    return f"""
Simulation ID: {sample["simulation_id"]}
Task ID: {sample["task_id"]}

Scenario / User Instructions:
{sample["user_instructions"]}

Topic Plan:
{sample["topic_plan"]}

Complete Conversation:
{sample["conversation_window"]}

Observed Interruption Events:
{json.dumps(sample["events"], ensure_ascii=False, indent=2)}

Return JSON in this shape:
{{
  "simulation_id": "{sample["simulation_id"]}",
  "task_id": "{sample["task_id"]}",
  "scores": {{
    "overall_user_naturalness": 1
  }},
  "overall_decision": "pass | borderline | fail",
  "failure_modes": [
    "short label for each issue"
  ],
  "notes": "brief explanation"
}}
""".strip()


def judge_event_sample(
    sample: dict[str, Any],
    *,
    review_model: str,
) -> dict[str, Any]:
    """Call prompt A to judge assistant quality for one duplex event."""
    messages = [
        SystemMessage(role="system", content=EVENT_ASSISTANT_QUALITY_PROMPT),
        UserMessage(role="user", content=build_user_prompt(sample)),
    ]
    assistant_message = generate(
        model=review_model,
        messages=messages,
        call_name="duplex_event_assistant_quality_judge",
    )
    payload = json.loads(extract_json_from_llm_response(assistant_message.content))
    return {
        "sample": {
            "sample_id": sample["sample_id"],
            "simulation_id": sample["simulation_id"],
            "task_id": sample["task_id"],
            "interruption_type": sample["interruption_type"],
            "metadata": sample["metadata"],
            "event": sample["event"],
            "topic_plan": sample["topic_plan"],
        },
        "judgement": payload,
        "cost": assistant_message.cost,
    }


def judge_conversation_sample(
    sample: dict[str, Any],
    *,
    review_model: str,
) -> dict[str, Any]:
    """Call prompt B to judge user naturalness for one full conversation."""
    messages = [
        SystemMessage(role="system", content=CONVERSATION_USER_NATURALNESS_PROMPT),
        UserMessage(
            role="user", content=build_conversation_user_naturalness_prompt(sample)
        ),
    ]
    assistant_message = generate(
        model=review_model,
        messages=messages,
        call_name="conversation_user_naturalness_judge",
    )
    payload = json.loads(extract_json_from_llm_response(assistant_message.content))
    return {
        "sample": {
            "simulation_id": sample["simulation_id"],
            "task_id": sample["task_id"],
            "topic_plan": sample["topic_plan"],
            "events": sample["events"],
        },
        "judgement": payload,
        "cost": assistant_message.cost,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize structured judge records."""
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        interruption_type = record["sample"]["interruption_type"]
        by_type[interruption_type].append(record)

    summary: dict[str, Any] = {"total_records": len(records), "by_type": {}}
    for interruption_type, type_records in sorted(by_type.items()):
        decisions = Counter(
            record["judgement"].get("overall_decision", "unknown")
            for record in type_records
        )
        score_names = set()
        for record in type_records:
            score_names.update((record["judgement"].get("scores") or {}).keys())
        score_means = {}
        for score_name in sorted(score_names):
            values = [
                record["judgement"].get("scores", {}).get(score_name)
                for record in type_records
            ]
            numeric = [float(value) for value in values if value is not None]
            score_means[score_name] = mean(numeric) if numeric else None
        summary["by_type"][interruption_type] = {
            "records": len(type_records),
            "decisions": dict(decisions),
            "score_means": score_means,
        }
    return summary


def summarize_conversations(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize conversation-level user naturalness judge records."""
    decisions = Counter(
        record["judgement"].get("overall_decision", "unknown") for record in records
    )
    score_names = set()
    for record in records:
        score_names.update((record["judgement"].get("scores") or {}).keys())
    score_means = {}
    for score_name in sorted(score_names):
        values = [
            record["judgement"].get("scores", {}).get(score_name) for record in records
        ]
        numeric = [float(value) for value in values if value is not None]
        score_means[score_name] = mean(numeric) if numeric else None
    return {
        "total_records": len(records),
        "decisions": dict(decisions),
        "score_means": score_means,
    }


def select_events(
    rows: list[dict[str, Any]],
    *,
    interruption_type: str | None,
    event_index: int,
    max_events: int,
) -> list[dict[str, Any]]:
    """Filter and slice events for judging."""
    if interruption_type:
        rows = [row for row in rows if row["interruption_type"] == interruption_type]
    if event_index < 0:
        raise ValueError("--event-index must be >= 0")
    return rows[event_index : event_index + max_events]


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_events(args.events_csv)
    selected = select_events(
        rows,
        interruption_type=args.interruption_type,
        event_index=args.event_index,
        max_events=args.max_events,
    )
    if not selected:
        raise ValueError("No events selected")

    event_records = []
    for event in selected:
        sample = build_sample(
            event,
            context_before_sec=args.context_before_sec,
            context_after_sec=args.context_after_sec,
        )
        if args.dry_run:
            event_records.append(
                {
                    "sample": {
                        "sample_id": sample["sample_id"],
                        "simulation_id": sample["simulation_id"],
                        "task_id": sample["task_id"],
                        "interruption_type": sample["interruption_type"],
                        "metadata": sample["metadata"],
                        "event": sample["event"],
                        "topic_plan": sample["topic_plan"],
                    },
                    "prompts": {
                        "system": EVENT_ASSISTANT_QUALITY_PROMPT,
                        "user": build_user_prompt(sample),
                    },
                    "judgement": None,
                    "cost": 0.0,
                }
            )
        else:
            event_records.append(
                judge_event_sample(sample, review_model=args.review_model)
            )

    conversation_records = []
    selected_by_simulation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in selected:
        selected_by_simulation[event["simulation_id"]].append(event)
    for events in selected_by_simulation.values():
        sample = build_conversation_sample(events)
        if args.dry_run:
            conversation_records.append(
                {
                    "sample": {
                        "simulation_id": sample["simulation_id"],
                        "task_id": sample["task_id"],
                        "topic_plan": sample["topic_plan"],
                        "events": sample["events"],
                    },
                    "prompts": {
                        "system": CONVERSATION_USER_NATURALNESS_PROMPT,
                        "user": build_conversation_user_naturalness_prompt(sample),
                    },
                    "judgement": None,
                    "cost": 0.0,
                }
            )
        else:
            conversation_records.append(
                judge_conversation_sample(sample, review_model=args.review_model)
            )

    output = {
        "mode": "dry_run" if args.dry_run else "llm_judge",
        "review_model": args.review_model,
        "events_csv": str(args.events_csv),
        "summary": {
            "event_assistant_quality": summarize(event_records),
            "conversation_user_naturalness": summarize_conversations(
                conversation_records
            ),
        }
        if not args.dry_run
        else None,
        "event_records": event_records,
        "conversation_records": conversation_records,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-csv", type=Path, default=DEFAULT_EVENTS_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--interruption-type", default="")
    parser.add_argument("--event-index", type=int, default=0)
    parser.add_argument("--max-events", type=int, default=1)
    parser.add_argument("--context-before-sec", type=float, default=12.0)
    parser.add_argument("--context-after-sec", type=float, default=14.0)
    parser.add_argument("--review-model", default=DEFAULT_LLM_EVAL_USER_SIMULATOR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build samples and prompts without calling the LLM judge.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = run(args)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
