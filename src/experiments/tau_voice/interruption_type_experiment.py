#!/usr/bin/env python3
"""Run and analyze typed user-interruption experiments for audio-native agents."""

import argparse
import csv
import json
import os
import subprocess
import sys
from argparse import BooleanOptionalAction
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Optional

INTERRUPTION_TYPES = [
    "follow_up",
    "denial",
    "repeat_request",
    "topic_switch",
    "silence_stop",
]

INTERRUPTION_INFO = "Callback decided to interrupt."
DEFAULT_OUTPUT_DIR = Path("interruption_type_experiment")
SIMULATIONS_ROOT = Path("data/simulations")

DOMAIN_TOPIC_PROFILES = {
    "retail": {
        "main_topic": (
            "resolve the customer's order issue while clarifying eligibility, "
            "required identity/order details, return shipping, and next steps"
        ),
        "side_topics": [
            "whether original packaging, tags, or a receipt are required",
            "how price differences, refunds, or store credit will be handled",
            "whether replacement shipping timing changes after the exchange starts",
            "whether only one item can be exchanged while keeping the rest",
        ],
    },
    "airline": {
        "main_topic": (
            "resolve the customer's flight change or cancellation while clarifying "
            "fees, available options, baggage, and travel timing"
        ),
        "side_topics": [
            "whether checked bags or seats transfer to the new itinerary",
            "how change fees or fare differences are calculated",
            "whether the passenger can keep the same travel day",
            "what happens to credits or refunds after the change",
        ],
    },
    "telecom": {
        "main_topic": (
            "resolve the customer's account or service issue while clarifying "
            "plan details, billing impact, troubleshooting, and next steps"
        ),
        "side_topics": [
            "whether the change affects the next bill",
            "whether service downtime is expected",
            "which device or line is affected",
            "how long the fix or plan change will take",
        ],
    },
    "banking_knowledge": {
        "main_topic": (
            "resolve the customer's banking question while clarifying policy "
            "requirements, timelines, risk, and next steps"
        ),
        "side_topics": [
            "whether the policy has exceptions",
            "what documents or verification are needed",
            "how long the process usually takes",
            "what the customer should do if the first path fails",
        ],
    },
    "mock": {
        "main_topic": (
            "solve the customer's stated problem while clarifying constraints, "
            "available options, and next steps"
        ),
        "side_topics": [
            "whether there is an easier workaround",
            "what information the assistant needs first",
            "how long the next step should take",
            "what happens if the first option does not work",
        ],
    },
}

DISCUSSION_TOPIC_PROFILES = [
    {
        "main_topic": (
            "help the user think through a realistic weekly meal-prep plan for "
            "busy weekdays, including tradeoffs around time, cost, nutrition, "
            "and variety"
        ),
        "side_topics": [
            "whether leftovers can stay appealing by changing sauces or sides",
            "how to handle one night when the user gets home late",
            "whether frozen vegetables are a reasonable shortcut",
            "how to avoid buying ingredients that only get used once",
        ],
    },
    {
        "main_topic": (
            "help the user plan a quieter home-office routine, including desk "
            "setup, meeting blocks, focus time, and practical boundaries"
        ),
        "side_topics": [
            "whether a noise-canceling headset is enough in a shared apartment",
            "how to handle short breaks without losing focus",
            "whether morning or afternoon deep-work blocks are more realistic",
            "how to keep notifications from disrupting calls",
        ],
    },
    {
        "main_topic": (
            "help the user compare two weekend hiking options, including "
            "weather risk, difficulty, travel time, gear, and backup plans"
        ),
        "side_topics": [
            "whether bringing a lightweight rain shell changes the decision",
            "how early the user should start to avoid crowds",
            "whether a shorter trail is better if a friend is new to hiking",
            "what to do if the trailhead parking lot is full",
        ],
    },
    {
        "main_topic": (
            "help the user prepare for a first meeting with a new project team, "
            "including agenda, expectations, risks, and next steps"
        ),
        "side_topics": [
            "whether to send a short pre-read before the meeting",
            "how much background context to give at the start",
            "how to handle one stakeholder who may dominate the discussion",
            "whether to end with decisions or open questions",
        ],
    },
]

INTERRUPTION_TOPIC_GUIDANCE = {
    "follow_up": (
        "Interrupt with one concrete clarification about the assistant's current "
        "explanation or one of the related side topics."
    ),
    "denial": (
        "Interrupt when the user can naturally push back on, reject, or express "
        "skepticism about something the assistant just said. The assistant does "
        "not need to be wrong; the user may knowingly resist a reasonable "
        "suggestion because it feels unrealistic, inconvenient, costly, boring, "
        "or mismatched with their preferences."
    ),
    "repeat_request": (
        "Interrupt by restating the main topic if the assistant is drifting, "
        "skipping the core request, or over-explaining a side point."
    ),
    "topic_switch": (
        "Interrupt by briefly moving to a related side topic, then return to the "
        "main topic after the assistant addresses it."
    ),
    "silence_stop": (
        "Interrupt by asking the assistant to pause or hold on, then resume the "
        "main topic with a concrete next question."
    ),
}

DISCUSSION_INTERRUPTION_TOPIC_GUIDANCE = {
    **INTERRUPTION_TOPIC_GUIDANCE,
    "denial": (
        "Interrupt when the assistant makes a suggestion the user can naturally "
        "push back on. The assistant does not need to be wrong; briefly object "
        "because the suggestion feels unrealistic, inconvenient, costly, boring, "
        "or mismatched with the user's preferences or constraints."
    ),
}


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _contextual_topic_plan(
    *,
    domain: str,
    interruption_type: str,
    type_sequence: list[str],
    sample_idx: int,
    main_topic: str,
    topic_source: str,
) -> dict[str, Any]:
    if topic_source == "discussion":
        profile = DISCUSSION_TOPIC_PROFILES[sample_idx % len(DISCUSSION_TOPIC_PROFILES)]
    else:
        profile = DOMAIN_TOPIC_PROFILES.get(domain, DOMAIN_TOPIC_PROFILES["mock"])
    side_topics = profile["side_topics"]
    interruption_guidance = INTERRUPTION_TOPIC_GUIDANCE
    if topic_source == "discussion":
        interruption_guidance = DISCUSSION_INTERRUPTION_TOPIC_GUIDANCE
    first_side_topic = side_topics[sample_idx % len(side_topics)]
    second_side_topic = side_topics[(sample_idx + 1) % len(side_topics)]
    style = (
        "Natural customer speech. Be concise but specific. Avoid repeatedly "
        "asking the same clarification or interrupting during greetings."
    )
    if topic_source == "discussion":
        style = (
            "Natural spoken language. Be concise but specific. Share practical "
            "preferences or constraints when useful, and avoid generic filler."
        )
    first_arc_step = "Start by discussing the main topic concretely."
    if topic_source == "domain":
        first_arc_step = (
            "Start by discussing the main topic concretely, using only facts "
            "from the scenario."
        )
    effective_sequence = type_sequence or [interruption_type]
    sequence_steps = [
        {
            "event_index": idx + 1,
            "interruption_type": event_type,
            "guidance": interruption_guidance[event_type],
        }
        for idx, event_type in enumerate(effective_sequence)
    ]
    conversation_arc = [
        first_arc_step,
        "Let the assistant provide at least one substantive step, condition, or explanation before interrupting.",
    ]
    for step in sequence_steps:
        conversation_arc.append(
            f"Interruption {step['event_index']}: use {step['interruption_type']}. "
            f"{step['guidance']}"
        )
    conversation_arc.extend(
        [
            "After each interruption is addressed, steer naturally back to the main topic.",
            "Do not add extra interruption types beyond the requested sequence.",
        ]
    )
    return {
        "main_topic": main_topic or profile["main_topic"],
        "interruption_type": interruption_type,
        "interruption_sequence": effective_sequence,
        "interruption_sequence_steps": sequence_steps,
        "related_side_topics": [first_side_topic, second_side_topic],
        "conversation_arc": conversation_arc,
        "interruption_guidance": (
            "Follow the interruption sequence in order: "
            + " -> ".join(effective_sequence)
        ),
        "style": style,
    }


def _free_conversation_agent_policy(topic_plan_text: str) -> str:
    return f"""
You are the assistant in a natural voice conversation.

This is a turn-taking and interruption test, not a customer-service benchmark.
Do not use tools, do not ask for account/order identifiers, and do not try to
look anything up. Discuss the topic directly from common reasoning and the
conversation context.

Use the topic plan as the conversation anchor:
{topic_plan_text}

When the user interrupts with a related side question, stop your current answer
promptly, address that question, then naturally return to the main topic. Keep
responses conversational and substantive, with concrete suggestions and tradeoffs.
""".strip()


def _free_conversation_user_instructions(topic_plan_text: str) -> str:
    return f"""
You are a person having a natural voice conversation with an assistant.

This is not a customer-service task. Do not ask the assistant to look up orders,
accounts, flights, policies, records, or any private data. Do not provide IDs,
emails, order numbers, or authentication details. Your goal is to discuss the
topic below, ask practical follow-up questions, briefly interrupt once or twice
with related side questions when it feels natural, and then return to the main
topic.

Topic plan:
{topic_plan_text}

Start by stating the concrete situation you want help thinking through. Let the
assistant give at least one useful explanation before interrupting. After the
assistant answers your side question, steer back to the main topic. When the
discussion feels complete, say ###STOP###.
""".strip()


def _format_topic_plan(plan: dict[str, Any]) -> str:
    side_topics = "\n".join(f"- {topic}" for topic in plan["related_side_topics"])
    arc = "\n".join(f"- {step}" for step in plan["conversation_arc"])
    sequence_steps = "\n".join(
        (f"- #{step['event_index']} {step['interruption_type']}: {step['guidance']}")
        for step in plan.get("interruption_sequence_steps", [])
    )
    return f"""
Main topic:
{plan["main_topic"]}

Related side topics:
{side_topics}

Interruption sequence:
{sequence_steps or "- Use the configured interruption type."}

Conversation arc:
{arc}

Interruption guidance:
{plan["interruption_guidance"]}

Style:
{plan["style"]}
""".strip()


def _chunk_has_speech(chunk: Optional[dict[str, Any]]) -> bool:
    return bool(chunk and chunk.get("contains_speech"))


def _chunk_text(chunk: Optional[dict[str, Any]]) -> str:
    if not chunk:
        return ""
    return (chunk.get("content") or chunk.get("audio_script_gold") or "").strip()


def _segment_end(ticks: list[dict[str, Any]], start: int, role: str) -> int:
    chunk_key = f"{role}_chunk"
    idx = start
    while idx < len(ticks) and _chunk_has_speech(ticks[idx].get(chunk_key)):
        idx += 1
    return idx


def _segment_text(ticks: list[dict[str, Any]], start: int, end: int, role: str) -> str:
    chunk_key = f"{role}_chunk"
    parts = []
    seen = set()
    for idx in range(start, end):
        text = _chunk_text(ticks[idx].get(chunk_key))
        if text and text not in seen:
            parts.append(text)
            seen.add(text)
    return " ".join(parts)


def _find_agent_yield(
    ticks: list[dict[str, Any]],
    start: int,
    window_ticks: int,
) -> Optional[int]:
    stop = min(start + window_ticks + 1, len(ticks))
    for idx in range(start, stop):
        if not _chunk_has_speech(ticks[idx].get("agent_chunk")):
            return idx
    return None


def _find_agent_response(
    ticks: list[dict[str, Any]],
    user_end: int,
    response_window_ticks: int,
) -> tuple[Optional[int], Optional[int]]:
    stop = min(user_end + response_window_ticks + 1, len(ticks))
    next_user_start = None
    agent_start = None
    for idx in range(user_end, stop):
        prev_user = idx > 0 and _chunk_has_speech(ticks[idx - 1].get("user_chunk"))
        user_started = _chunk_has_speech(ticks[idx].get("user_chunk")) and not prev_user
        if user_started:
            next_user_start = idx
            break

        prev_agent = idx > 0 and _chunk_has_speech(ticks[idx - 1].get("agent_chunk"))
        agent_started = (
            _chunk_has_speech(ticks[idx].get("agent_chunk")) and not prev_agent
        )
        if agent_started:
            agent_start = idx
            break
    return agent_start, next_user_start


def _tick_duration(results_dir: Path) -> float:
    results_path = results_dir / "results.json"
    with results_path.open(encoding="utf-8-sig") as f:
        results = json.load(f)
    return (
        results.get("info", {})
        .get("audio_native_config", {})
        .get("tick_duration_seconds", 0.2)
    )


def _infer_type_from_path(results_dir: Path) -> Optional[str]:
    for part in reversed(results_dir.parts):
        if part in INTERRUPTION_TYPES:
            return part
    return None


def _simulation_paths(results_dir: Path) -> list[Path]:
    sims_dir = results_dir / "simulations"
    if sims_dir.exists():
        return sorted(sims_dir.glob("*.json"))

    with (results_dir / "results.json").open(encoding="utf-8-sig") as f:
        results = json.load(f)
    sims = results.get("simulations") or []
    if sims:
        merged_path = results_dir / "results.json"
        return [merged_path]
    return []


def _iter_simulations(results_dir: Path) -> Iterable[tuple[Path, dict[str, Any]]]:
    for sim_path in _simulation_paths(results_dir):
        with sim_path.open(encoding="utf-8-sig") as f:
            payload = json.load(f)
        if "simulations" in payload:
            for sim in payload["simulations"]:
                yield sim_path, sim
        else:
            yield sim_path, payload


def _analyze_results_dir(
    results_dir: Path,
    *,
    yield_window_sec: float,
    response_window_sec: float,
) -> list[dict[str, Any]]:
    tick_duration_sec = _tick_duration(results_dir)
    yield_window_ticks = int(yield_window_sec / tick_duration_sec)
    response_window_ticks = int(response_window_sec / tick_duration_sec)
    inferred_type = _infer_type_from_path(results_dir)
    rows = []

    for sim_path, sim in _iter_simulations(results_dir):
        ticks = sim.get("ticks") or []
        for idx, tick in enumerate(ticks):
            user_chunk = tick.get("user_chunk")
            prev_user_speaking = idx > 0 and _chunk_has_speech(
                ticks[idx - 1].get("user_chunk")
            )
            user_started = _chunk_has_speech(user_chunk) and not prev_user_speaking
            agent_speaking = _chunk_has_speech(tick.get("agent_chunk"))
            if not (user_started and agent_speaking):
                continue

            action = (user_chunk or {}).get("turn_taking_action") or {}
            if action.get("info") != INTERRUPTION_INFO:
                continue

            interruption_type = action.get("interruption_type") or inferred_type
            interruption_type = interruption_type or "unknown"
            user_end = _segment_end(ticks, idx, "user")
            yield_tick = _find_agent_yield(ticks, idx, yield_window_ticks)
            agent_start, next_user_start = _find_agent_response(
                ticks, user_end, response_window_ticks
            )
            response_text = (
                _segment_text(
                    ticks,
                    agent_start,
                    _segment_end(ticks, agent_start, "agent"),
                    "agent",
                )
                if agent_start is not None
                else ""
            )

            rows.append(
                {
                    "interruption_type": interruption_type,
                    "results_dir": str(results_dir),
                    "simulation_file": str(sim_path),
                    "simulation_id": sim.get("id", ""),
                    "task_id": sim.get("task_id", ""),
                    "interruption_start_tick": idx,
                    "interruption_start_sec": idx * tick_duration_sec,
                    "user_end_tick": user_end,
                    "user_transcript": _segment_text(ticks, idx, user_end, "user"),
                    "agent_overlap_transcript": _segment_text(
                        ticks, idx, min(user_end, len(ticks)), "agent"
                    ),
                    "interruption_success": yield_tick is not None,
                    "agent_yield_tick": yield_tick,
                    "post_interrupt_stop_latency_sec": (
                        (yield_tick - idx) * tick_duration_sec
                        if yield_tick is not None
                        else None
                    ),
                    "post_interrupt_answer_success": agent_start is not None,
                    "agent_response_start_tick": agent_start,
                    "agent_response_latency_sec": (
                        (agent_start - user_end) * tick_duration_sec
                        if agent_start is not None
                        else None
                    ),
                    "next_user_start_tick": next_user_start,
                    "agent_response_transcript": response_text,
                    "tick_duration_sec": tick_duration_sec,
                }
            )
    return rows


def _find_results_dirs(path: Path) -> list[Path]:
    if (path / "results.json").exists():
        return [path]
    return sorted(
        parent for parent in path.rglob("*") if (parent / "results.json").exists()
    )


def write_analysis(
    results_root: Path,
    *,
    output_dir: Path,
    yield_window_sec: float,
    response_window_sec: float,
) -> tuple[Path, Path]:
    rows = []
    for results_dir in _find_results_dirs(results_root):
        rows.extend(
            _analyze_results_dir(
                results_dir,
                yield_window_sec=yield_window_sec,
                response_window_sec=response_window_sec,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.csv"
    summary_path = output_dir / "summary.json"

    fieldnames = [
        "interruption_type",
        "results_dir",
        "simulation_file",
        "simulation_id",
        "task_id",
        "interruption_start_tick",
        "interruption_start_sec",
        "user_end_tick",
        "user_transcript",
        "agent_overlap_transcript",
        "interruption_success",
        "agent_yield_tick",
        "post_interrupt_stop_latency_sec",
        "post_interrupt_answer_success",
        "agent_response_start_tick",
        "agent_response_latency_sec",
        "next_user_start_tick",
        "agent_response_transcript",
        "tick_duration_sec",
    ]
    with events_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_events(rows)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return events_path, summary_path


def summarize_events(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["interruption_type"]].append(row)

    summary = {}
    for interruption_type in INTERRUPTION_TYPES + sorted(
        set(grouped) - set(INTERRUPTION_TYPES)
    ):
        group = grouped.get(interruption_type, [])
        total = len(group)
        yielded = [row for row in group if row["interruption_success"]]
        answered = [row for row in group if row["post_interrupt_answer_success"]]
        latencies = [
            row["post_interrupt_stop_latency_sec"]
            for row in yielded
            if row["post_interrupt_stop_latency_sec"] is not None
        ]
        summary[interruption_type] = {
            "attempted_interruptions": total,
            "simulations_with_interruptions": len(
                {row["simulation_id"] for row in group}
            ),
            "interruption_success_rate": len(yielded) / total if total else None,
            "post_interrupt_answer_success_rate": (
                len(answered) / total if total else None
            ),
            "post_interrupt_stop_latency_mean_sec": mean(latencies)
            if latencies
            else None,
            "post_interrupt_stop_latency_samples": len(latencies),
        }
    return {"total_events": len(rows), "by_type": summary}


def sanitize_results(path: Path, secret_values: list[str]) -> None:
    secrets = [value for value in secret_values if value]
    if not secrets:
        return
    for json_path in path.rglob("*.json"):
        text = json_path.read_text(encoding="utf-8")
        original = text
        for secret in secrets:
            text = text.replace(secret, "<REDACTED_API_KEY>")
        if text != original:
            json_path.write_text(text, encoding="utf-8")


def _run_one(args: argparse.Namespace, interruption_type: str, sample_idx: int) -> int:
    save_to = args.save_to / interruption_type / f"sample_{sample_idx:02d}"
    save_to_arg = str(save_to)
    topic_plan = None
    topic_plan_text = ""
    type_sequence = _split_csv(args.type_sequence)
    plan_interruption_type = type_sequence[0] if type_sequence else interruption_type
    if args.contextual_topic_plan:
        topic_plan = _contextual_topic_plan(
            domain=args.domain,
            interruption_type=plan_interruption_type,
            type_sequence=type_sequence,
            sample_idx=sample_idx,
            main_topic=args.contextual_main_topic,
            topic_source=args.topic_source,
        )
        topic_plan_text = _format_topic_plan(topic_plan)

        sample_results_dir = SIMULATIONS_ROOT / save_to
        sample_results_dir.mkdir(parents=True, exist_ok=True)
        with (sample_results_dir / "topic_plan.json").open("w", encoding="utf-8") as f:
            json.dump(topic_plan, f, ensure_ascii=False, indent=2)
        (sample_results_dir / "topic_plan.txt").write_text(
            topic_plan_text, encoding="utf-8"
        )

    user_llm_args = {"temperature": args.user_llm_temperature}
    user_llm_api_base = args.user_llm_api_base.strip()
    if user_llm_api_base.lower() in {"none", "default"}:
        user_llm_api_base = ""
    if user_llm_api_base:
        user_llm_args["api_base"] = user_llm_api_base
    user_llm_api_key = os.getenv(args.user_llm_api_key_env, "")
    if user_llm_api_key and user_llm_api_base:
        user_llm_args["api_key"] = user_llm_api_key

    cmd = [
        sys.executable,
        "-m",
        "tau2.cli",
        "run",
        "--domain",
        args.domain,
        "--audio-native",
        "--audio-native-provider",
        args.audio_native_provider,
        "--audio-native-model",
        args.audio_native_model,
        "--user-llm",
        args.user_llm,
        "--user-llm-args",
        json.dumps(user_llm_args),
        "--num-trials",
        "1",
        "--num-tasks",
        str(args.num_tasks),
        "--max-concurrency",
        "1",
        "--max-steps-seconds",
        str(args.max_steps_seconds),
        "--speech-complexity",
        args.speech_complexity,
        "--hallucination-retries",
        "0",
        "--save-to",
        save_to_arg,
        "--log-level",
        args.log_level,
    ]
    if args.audio_taps:
        cmd.append("--audio-taps")
    if args.task_ids:
        cmd.extend(["--task-ids", args.task_ids])
    env = os.environ.copy()
    env["TAU2_INTERRUPT_EXPERIMENT_TYPE"] = interruption_type
    if type_sequence:
        env["TAU2_INTERRUPT_EXPERIMENT_SEQUENCE"] = ",".join(type_sequence)
    effective_interruption_rate = (
        args.contextual_interruption_rate
        if args.contextual_topic_plan
        else args.interruption_rate
    )
    env["TAU2_INTERRUPT_EXPERIMENT_RATE"] = str(effective_interruption_rate)
    if topic_plan_text:
        env["TAU2_CONTEXTUAL_TURN_TAKING_PLAN"] = topic_plan_text
        env["TAU2_INTERRUPT_EXPERIMENT_MAX_EVENTS"] = str(
            len(type_sequence) if type_sequence else args.contextual_max_interruptions
        )
        env["TAU2_INTERRUPT_EXPERIMENT_MIN_USER_TURNS"] = str(
            args.contextual_min_user_turns_before_interrupt
        )
        env["TAU2_INTERRUPT_EXPERIMENT_MIN_ELAPSED_SECONDS"] = str(
            args.contextual_min_elapsed_seconds_before_interrupt
        )
        env["TAU2_INTERRUPT_EXPERIMENT_MIN_SECONDS_SINCE_LAST_EVENT"] = str(
            args.contextual_min_seconds_since_last_interruption
        )
    if args.disable_domain_tools:
        env["TAU2_DISABLE_DOMAIN_TOOLS"] = "1"
    if args.topic_source == "discussion" and topic_plan_text:
        env["TAU2_FREE_CONVERSATION_AGENT_POLICY"] = _free_conversation_agent_policy(
            topic_plan_text
        )
        env["TAU2_FREE_CONVERSATION_USER_INSTRUCTIONS"] = (
            _free_conversation_user_instructions(topic_plan_text)
        )
    if args.force_interruptions:
        env["TAU2_INTERRUPT_EXPERIMENT_FORCE"] = "1"
        effective_min_words = (
            args.contextual_min_agent_words_before_interrupt
            if args.contextual_topic_plan
            else args.min_agent_words_before_interrupt
        )
        env["TAU2_INTERRUPT_EXPERIMENT_MIN_AGENT_WORDS"] = str(effective_min_words)
    if args.tts_provider:
        env["TAU2_TTS_PROVIDER"] = args.tts_provider
    if args.proxy:
        env["HTTP_PROXY"] = args.proxy
        env["HTTPS_PROXY"] = args.proxy
        env["ALL_PROXY"] = args.proxy
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    print(f"[{interruption_type} sample {sample_idx:02d}] {save_to_arg}")
    result = subprocess.run(cmd, env=env)
    sanitize_results(save_to, [user_llm_api_key])
    return result.returncode


def run_experiment(args: argparse.Namespace) -> int:
    args.save_to = Path(args.save_to)
    types = _split_csv(args.types)
    type_sequence = _split_csv(args.type_sequence)
    for event_type in type_sequence:
        if event_type not in INTERRUPTION_TYPES:
            raise ValueError(
                f"Unknown interruption type in --type-sequence: {event_type}"
            )
    if type_sequence:
        types = ["sequence"]
    failures = 0
    for interruption_type in types:
        if not type_sequence and interruption_type not in INTERRUPTION_TYPES:
            raise ValueError(f"Unknown interruption type: {interruption_type}")
        for sample_idx in range(args.samples_per_type):
            returncode = _run_one(args, interruption_type, sample_idx)
            if returncode != 0:
                failures += 1
                print(f"WARNING: sample failed with exit code {returncode}")

    results_root = SIMULATIONS_ROOT / args.save_to
    events_path, summary_path = write_analysis(
        results_root,
        output_dir=results_root,
        yield_window_sec=args.yield_window_sec,
        response_window_sec=args.response_window_sec,
    )
    print(f"events: {events_path}")
    print(f"summary: {summary_path}")
    return 1 if failures else 0


def analyze(args: argparse.Namespace) -> int:
    events_path, summary_path = write_analysis(
        Path(args.results_root),
        output_dir=Path(args.output_dir),
        yield_window_sec=args.yield_window_sec,
        response_window_sec=args.response_window_sec,
    )
    print(f"events: {events_path}")
    print(f"summary: {summary_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run typed interruption samples")
    run_parser.add_argument("--types", default=",".join(INTERRUPTION_TYPES))
    run_parser.add_argument(
        "--type-sequence",
        default="",
        help=(
            "Comma-separated ordered interruption sequence for each sample, e.g. "
            "follow_up,denial. When set, --types is ignored and results are saved "
            "under sequence/sample_XX."
        ),
    )
    run_parser.add_argument("--samples-per-type", type=int, default=10)
    run_parser.add_argument("--save-to", type=Path, default=DEFAULT_OUTPUT_DIR)
    run_parser.add_argument("--domain", default="retail")
    run_parser.add_argument("--num-tasks", type=int, default=1)
    run_parser.add_argument("--task-ids", default="")
    run_parser.add_argument(
        "--disable-domain-tools",
        action=BooleanOptionalAction,
        default=True,
        help="Do not expose benchmark domain tools to the tested agent or user simulator.",
    )
    run_parser.add_argument("--audio-native-provider", default="openai")
    run_parser.add_argument("--audio-native-model", default="gpt-realtime")
    run_parser.add_argument("--user-llm", default="openai/doubao-seed-character-260628")
    run_parser.add_argument(
        "--user-llm-api-base",
        default="https://ark.cn-beijing.volces.com/api/v3",
    )
    run_parser.add_argument("--user-llm-api-key-env", default="ARK_API_KEY")
    run_parser.add_argument("--user-llm-temperature", type=float, default=0.0)
    run_parser.add_argument("--tts-provider", default="volcengine")
    run_parser.add_argument("--speech-complexity", default="control_behavior")
    run_parser.add_argument("--max-steps-seconds", type=int, default=180)
    run_parser.add_argument("--interruption-rate", type=float, default=0.5)
    run_parser.add_argument(
        "--contextual-interruption-rate",
        type=float,
        default=0.35,
        help="Interruption sampling rate when contextual topic planning is enabled.",
    )
    run_parser.add_argument(
        "--force-interruptions",
        action=BooleanOptionalAction,
        default=False,
        help="Force the interruption timing after the agent has spoken enough; LLM still generates the utterance.",
    )
    run_parser.add_argument("--min-agent-words-before-interrupt", type=int, default=20)
    run_parser.add_argument(
        "--contextual-min-agent-words-before-interrupt",
        type=int,
        default=24,
        help="Minimum agent words before forced interruption when contextual topic planning is enabled.",
    )
    run_parser.add_argument(
        "--contextual-min-user-turns-before-interrupt",
        type=int,
        default=2,
        help="Minimum completed user turns before contextual interruptions are allowed.",
    )
    run_parser.add_argument(
        "--contextual-min-elapsed-seconds-before-interrupt",
        type=float,
        default=20.0,
        help="Minimum elapsed seconds before contextual interruptions are allowed.",
    )
    run_parser.add_argument(
        "--contextual-min-seconds-since-last-interruption",
        type=float,
        default=0.0,
        help="Minimum seconds between contextual interruption events.",
    )
    run_parser.add_argument(
        "--contextual-max-interruptions",
        type=int,
        default=2,
        help="Maximum interruptions per sample when contextual topic planning is enabled.",
    )
    run_parser.add_argument(
        "--contextual-topic-plan",
        action=BooleanOptionalAction,
        default=True,
        help="Generate and inject a contextual topic plan for each sample.",
    )
    run_parser.add_argument(
        "--contextual-main-topic",
        default="",
        help="Override the automatically generated main topic for all samples.",
    )
    run_parser.add_argument(
        "--topic-source",
        choices=["discussion", "domain"],
        default="discussion",
        help="Use free discussion topics or original domain/task-shaped topics.",
    )
    run_parser.add_argument("--yield-window-sec", type=float, default=2.0)
    run_parser.add_argument("--response-window-sec", type=float, default=12.0)
    run_parser.add_argument("--proxy", default="")
    run_parser.add_argument("--log-level", default="ERROR")
    run_parser.add_argument(
        "--audio-taps",
        action=BooleanOptionalAction,
        default=True,
        help="Save real user/agent audio tap WAV files for each simulation.",
    )
    run_parser.set_defaults(func=run_experiment)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze existing results")
    analyze_parser.add_argument("results_root", type=Path)
    analyze_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    analyze_parser.add_argument("--yield-window-sec", type=float, default=2.0)
    analyze_parser.add_argument("--response-window-sec", type=float, default=12.0)
    analyze_parser.set_defaults(func=analyze)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
