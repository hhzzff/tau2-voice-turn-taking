"""Prepare, execute and score a selected subset of duplex events.

Specify only event types. Terra generates a topic, then chooses the event,
timing and content during the live conversation. Preparation saves policy only.
"""

import argparse
import json
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from tau2.user.duplex_events import (
    EVENT_CATALOG,
    DuplexScenario,
    generate_topic,
)


@contextmanager
def conversation_environment(language):
    """Scope existing free-conversation hooks without exposing the private plan."""
    updates = {
        "TAU2_DISABLE_DOMAIN_TOOLS": "1",
        "TAU2_FREE_CONVERSATION_USER_INSTRUCTIONS": "Private dynamic voice conversation.",
        "TAU2_FREE_CONVERSATION_AGENT_POLICY": (
            "You are a helpful voice assistant having an ordinary conversation. "
            f"Speak {language}. Respond naturally to the user."
        ),
        "TAU2_CONTEXTUAL_TURN_TAKING_PLAN": "",
        "TAU2_INTERRUPT_EXPERIMENT_SEQUENCE": "",
        "TAU2_INTERRUPT_EXPERIMENT_TYPE": "",
    }
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_scenario(path):
    return DuplexScenario.model_validate_json(path.read_text(encoding="utf-8"))


def prepare(args):
    """Save event types and options, without event content, ordering or timing."""
    if args.output.exists():
        raise FileExistsError(f"Scenario already exists: {args.output}")
    scenario = DuplexScenario(
        selected_events=[
            value.strip() for value in args.events.split(",") if value.strip()
        ],
        seed=args.seed,
        language=args.language,
        user_model=args.user_model,
        assistant_observation=args.assistant_observation,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(scenario.model_dump_json(indent=2), encoding="utf-8")
    print(f"Prepared event policy (no API calls or fixed script): {args.output}")
    return 0


def build_dynamic_user(
    scenario, output, config, primary_voice, background_voice, tts_model
):
    """Disable unrelated random effects for attribution to the selected events."""
    from tau2.data_model.audio_effects import (
        ChannelEffectsConfig,
        SourceEffectsConfig,
        SpeechEffectsConfig,
    )
    from tau2.data_model.voice import (
        OpenAITTSConfig,
        SpeechEnvironment,
        SynthesisConfig,
        VoiceSettings,
    )
    from tau2.user.duplex_scenario_user import DuplexUserSimulator

    settings = VoiceSettings(
        transcription_config=None,
        synthesis_config=SynthesisConfig(
            provider="openai",
            provider_config=OpenAITTSConfig(model=tts_model, voice=primary_voice),
            channel_effects_config=ChannelEffectsConfig(enable_frame_drops=False),
            source_effects_config=SourceEffectsConfig(
                enable_background_noise=False, enable_burst_noise=False
            ),
            speech_effects_config=SpeechEffectsConfig(
                enable_dynamic_muffling=False,
                enable_vocal_tics=False,
                enable_non_directed_phrases=False,
                speech_insert_events_per_minute=0,
            ),
        ),
        speech_environment=SpeechEnvironment(
            complexity="control", use_llm_backchannel=False
        ),
    )
    return DuplexUserSimulator(
        scenario=scenario,
        output_dir=output,
        primary_voice=primary_voice,
        background_voice=background_voice,
        tts_model=tts_model,
        instructions=scenario.user_context,
        tools=[],
        llm=scenario.user_model,
        voice_settings=settings,
        chunk_size=config.user_chunk_size,
        tick_duration_seconds=config.tick_duration_seconds,
        use_llm_backchannel=False,
    )


def run(args):
    """Record every topic, runtime and judge request/response for this trial."""
    from tau2.utils.llm_utils import llm_log_dir, llm_log_mode

    directory_token = llm_log_dir.set(args.output / "llm_calls")
    mode_token = llm_log_mode.set("all")
    try:
        return _run(args)
    finally:
        llm_log_mode.reset(mode_token)
        llm_log_dir.reset(directory_token)


def _run(args):
    """Run the runtime user policy and preserve partial evidence on failure."""
    from dotenv import load_dotenv

    from tau2.config import DEFAULT_AUDIO_NATIVE_MODELS
    from tau2.data_model.simulation import AudioNativeConfig
    from tau2.data_model.tasks import Task, UserScenario
    from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator
    from tau2.runner.build import build_agent, build_environment
    from tau2.user.duplex_evaluation import (
        assess_report,
        attach_asr_evidence,
        judge_report,
    )
    from tau2.voice.synthesis.conversation_builder import generate_simulation_audio

    load_dotenv()
    scenario = (
        _load_scenario(args.scenario)
        if args.scenario
        else DuplexScenario(
            selected_events=[
                value.strip() for value in args.events.split(",") if value.strip()
            ]
        )
    )
    if args.user_model:
        scenario.user_model = args.user_model
    if args.language:
        scenario.language = args.language
    if args.assistant_observation:
        scenario.assistant_observation = args.assistant_observation
    # Refuse accidental reuse of artifacts, including an interrupted earlier run.
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose an empty output directory for a new trial")
    args.output.mkdir(parents=True, exist_ok=True)
    config = AudioNativeConfig(
        provider=args.provider,
        model=args.model or DEFAULT_AUDIO_NATIVE_MODELS[args.provider],
        max_steps_seconds=args.max_seconds or (60 + 75 * len(scenario.selected_events)),
    )
    (args.output / "scenario.json").write_text(
        scenario.model_dump_json(indent=2), encoding="utf-8"
    )
    (args.output / "run_config.json").write_text(
        json.dumps(
            {
                "audio_native_config": config.model_dump(),
                "primary_voice": args.primary_voice,
                "background_voice": args.background_voice,
                "tts_model": args.tts_model,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    user = None
    orchestrator = None
    simulation = None
    assessment = None
    try:
        if not scenario.topic:
            topic = generate_topic(scenario)
            scenario.topic = topic.topic
            scenario.user_context = topic.user_context
            (args.output / "scenario.json").write_text(
                scenario.model_dump_json(indent=2), encoding="utf-8"
            )
        with conversation_environment(scenario.language):
            environment = build_environment("mock")
            task = Task(
                id="duplex_event_subset",
                user_scenario=UserScenario(instructions=scenario.user_context),
            )
            user = build_dynamic_user(
                scenario,
                args.output,
                config,
                args.primary_voice,
                args.background_voice,
                args.tts_model,
            )
            agent = build_agent(
                "discrete_time_audio_native_agent",
                environment,
                audio_native_config=config,
            )
            orchestrator = FullDuplexOrchestrator(
                domain="mock",
                agent=agent,
                user=user,
                environment=environment,
                task=task,
                max_steps=config.max_steps_ticks,
                tick_duration_seconds=config.tick_duration_seconds,
                seed=scenario.seed,
            )
            simulation = orchestrator.run()
            (args.output / "simulation.json").write_text(
                simulation.model_dump_json(indent=2), encoding="utf-8"
            )
            generate_simulation_audio(simulation, args.output / "audio")
    except Exception as exc:
        (args.output / "error.json").write_text(
            json.dumps({"error_type": type(exc).__name__}), encoding="utf-8"
        )
        raise
    finally:
        if user is not None:
            user.save_report()
            report = user.controller.report()
            if user.perception is not None:
                report = attach_asr_evidence(report, user.perception.snapshots)
                (args.output / "events.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            assessment = assess_report(scenario, report, config.tick_duration_seconds)
            if args.judge_model:
                assessment = judge_report(scenario, assessment, args.judge_model)
            (args.output / "assessment.json").write_text(
                json.dumps(assessment, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        if orchestrator is not None and orchestrator.ticks:
            # A run can fail before a SimulationRun is returned.
            (args.output / "ticks.json").write_text(
                json.dumps(
                    [tick.model_dump(mode="json") for tick in orchestrator.ticks],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        if orchestrator is not None and simulation is None:
            orchestrator._cleanup()
    print(f"Saved trajectory, audio, events and assessment: {args.output}")
    return (
        0
        if assessment
        and assessment["summary"]["observed"] == len(scenario.selected_events)
        else 1
    )


def assess(args):
    """Reassess saved evidence without opening an agent session or synthesizing."""
    from tau2.user.duplex_evaluation import (
        assess_report,
        attach_asr_evidence,
        judge_report,
    )

    scenario = _load_scenario(args.output / "scenario.json")
    report = json.loads((args.output / "events.json").read_text(encoding="utf-8"))
    perception_path = args.output / "perception.json"
    if perception_path.exists():
        perception = json.loads(perception_path.read_text(encoding="utf-8"))
        report = attach_asr_evidence(report, perception["snapshots"])
    config = json.loads((args.output / "run_config.json").read_text(encoding="utf-8"))
    assessment = assess_report(
        scenario, report, config["audio_native_config"]["tick_duration_seconds"]
    )
    if args.judge_model:
        from dotenv import load_dotenv

        load_dotenv()
        assessment = judge_report(scenario, assessment, args.judge_model)
    (args.output / "assessment.json").write_text(
        json.dumps(assessment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(assessment["summary"], ensure_ascii=False))
    return 0


def build_parser():
    """Expose event selection, preparation, execution and offline assessment."""
    from tau2.config import DEFAULT_DUPLEX_USER_MODEL
    from tau2.data_model.voice import OpenAITTSConfig

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List supported event IDs")
    listing.set_defaults(
        func=lambda args: print(
            "\n".join(f"{key}\t{spec.label}" for key, spec in EVENT_CATALOG.items())
        )
        or 0
    )
    planning = commands.add_parser(
        "prepare", help="Save event types and runtime options, with no fixed script"
    )
    planning.add_argument(
        "--events",
        required=True,
        help="Comma-separated event subset; execution order is chosen at runtime",
    )
    planning.add_argument("--seed", type=int, default=42)
    planning.add_argument(
        "--language", choices=["Chinese", "English"], default="Chinese"
    )
    planning.add_argument("--user-model", default=DEFAULT_DUPLEX_USER_MODEL)
    planning.add_argument(
        "--assistant-observation",
        choices=["played_audio", "proportional"],
        default="played_audio",
        help="Default: transcribe already played assistant audio before Terra decisions",
    )
    planning.add_argument("--output", type=Path, required=True)
    planning.set_defaults(func=prepare)
    execution = commands.add_parser(
        "run", help="Let Terra choose live duplex actions against a voice model"
    )
    source = execution.add_mutually_exclusive_group(required=True)
    source.add_argument("--scenario", type=Path)
    source.add_argument(
        "--events",
        help="Comma-separated event types; no scripted content or trigger timing",
    )
    execution.add_argument(
        "--output",
        type=Path,
        default=Path("data/simulations")
        / datetime.now().strftime("duplex_%Y%m%d_%H%M%S"),
    )
    execution.add_argument(
        "--user-model",
        help=f"Runtime timing and content model; default {DEFAULT_DUPLEX_USER_MODEL}",
    )
    execution.add_argument(
        "--language",
        choices=["Chinese", "English"],
        help="Conversation language; defaults to the scenario language or Chinese",
    )
    execution.add_argument(
        "--assistant-observation",
        choices=["played_audio", "proportional"],
        help="Override scenario perception; default uses ASR of already played audio",
    )
    execution.add_argument(
        "--provider",
        choices=["openai", "gemini", "nova", "xai", "qwen"],
        default="openai",
    )
    execution.add_argument(
        "--model", help="Default: configured model for the chosen provider"
    )
    execution.add_argument("--primary-voice", default="nova")
    execution.add_argument("--background-voice", default="onyx")
    execution.add_argument("--tts-model", default=OpenAITTSConfig().model)
    execution.add_argument("--max-seconds", type=int)
    execution.add_argument(
        "--judge-model",
        default=DEFAULT_DUPLEX_USER_MODEL,
        help="Semantic judge; defaults to the project's Terra model",
    )
    execution.add_argument(
        "--no-semantic-judge",
        dest="judge_model",
        action="store_const",
        const=None,
        help="Save timing evidence without model judging",
    )
    execution.set_defaults(func=run)
    scoring = commands.add_parser("assess", help="Assess an existing run")
    scoring.add_argument("--output", type=Path, required=True)
    scoring.add_argument("--judge-model")
    scoring.set_defaults(func=assess)
    return parser


def main():
    """Run the selected command."""
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
