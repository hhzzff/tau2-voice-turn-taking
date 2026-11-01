"""Prepare or run one two-speaker background-interference conversation.

Preparation is the default and does not call paid APIs. Run the saved scenario
with --run after reviewing it. No interference instructions reach the agent.
"""

import argparse
import json
import os
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from tau2.data_model.audio_effects import (
    ChannelEffectsConfig,
    SourceEffectsConfig,
    SpeechEffectsConfig,
)
from tau2.data_model.simulation import AudioNativeConfig
from tau2.data_model.tasks import Task, UserScenario
from tau2.data_model.voice import (
    OpenAITTSConfig,
    SpeechEnvironment,
    SynthesisConfig,
    VoiceSettings,
)
from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator
from tau2.runner.build import build_agent, build_environment
from tau2.user.third_party_simulator import (
    ThirdPartyScenario,
    ThirdPartyVoiceUser,
    scenario_llm_args,
)
from tau2.voice.synthesis.conversation_builder import generate_simulation_audio


def build_user(scenario, output, audio_config):
    """Build the two-speaker user with unrelated stochastic effects disabled."""
    voice = VoiceSettings(
        transcription_config=None,
        synthesis_config=SynthesisConfig(
            provider="openai",
            provider_config=OpenAITTSConfig(
                model=scenario.tts_model, voice=scenario.primary_voice
            ),
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
    return ThirdPartyVoiceUser(
        scenario=scenario,
        scenario_dir=output / "audio",
        tools=[],
        instructions=scenario.topic,
        llm=scenario.user_model,
        llm_args=scenario_llm_args(scenario.user_model, temperature=0.7),
        voice_settings=voice,
        chunk_size=audio_config.user_chunk_size,
        tick_duration_seconds=audio_config.tick_duration_seconds,
        wait_to_respond_threshold_other=audio_config.wait_to_respond_threshold_other_ticks,
        wait_to_respond_threshold_self=audio_config.wait_to_respond_threshold_self_ticks,
        yield_threshold_when_interrupted=audio_config.yield_threshold_when_interrupted_ticks,
        yield_threshold_when_interrupting=audio_config.yield_threshold_when_interrupting_ticks,
        integration_ticks=audio_config.integration_ticks,
        silence_annotation_threshold_ticks=audio_config.silence_annotation_threshold_ticks,
        use_llm_backchannel=False,
    )


def save_mono(stereo_path):
    """Make a listenable mono mix so both channels play on any audio device."""
    with wave.open(str(stereo_path), "rb") as source:
        channels, width, rate, count, _, _ = source.getparams()
        if width != 2:
            raise ValueError("Expected PCM16 both.wav")
        samples = np.frombuffer(source.readframes(count), dtype="<i2").reshape(
            -1, channels
        )
    mono = np.rint(samples.astype(float).mean(axis=1)).astype("<i2")
    with wave.open(str(stereo_path.with_name("both_mono.wav")), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(rate)
        target.writeframes(mono.tobytes())


def run(scenario, output):
    """Execute one live conversation, preserving partial evidence on failure."""
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in the repository .env")
    # Scope these overrides to this standalone experiment process.
    os.environ["TAU2_DISABLE_DOMAIN_TOOLS"] = "1"
    os.environ["TAU2_FREE_CONVERSATION_USER_INSTRUCTIONS"] = scenario.topic
    os.environ["TAU2_FREE_CONVERSATION_AGENT_POLICY"] = (
        "You are a helpful voice assistant having an ordinary conversation. "
        f"Speak {scenario.language}. Offer practical advice and respond naturally to the user's questions."
    )
    os.environ.pop("TAU2_CONTEXTUAL_TURN_TAKING_PLAN", None)
    config = AudioNativeConfig(
        provider="openai",
        model=scenario.agent_model,
        max_steps_seconds=scenario.max_seconds,
    )
    task = Task(
        id="third_party_interference",
        user_scenario=UserScenario(instructions=scenario.topic),
    )
    environment = build_environment("mock")
    user = build_user(scenario, output, config)
    # Synthesize before opening the agent's realtime connection.
    user.prepare_audio()
    agent = build_agent(
        "discrete_time_audio_native_agent", environment, audio_native_config=config
    )
    orchestrator = FullDuplexOrchestrator(
        domain="mock",
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        max_steps=config.max_steps_ticks,
        tick_duration_seconds=config.tick_duration_seconds,
        seed=42,
    )
    try:
        simulation = orchestrator.run()
        generate_simulation_audio(simulation, output / "audio")
        save_mono(output / "audio" / "both.wav")
        (output / "simulation.json").write_text(
            simulation.model_dump_json(indent=2), encoding="utf-8"
        )
    except Exception as exc:
        (output / "error.json").write_text(
            json.dumps({"error_type": type(exc).__name__}), encoding="utf-8"
        )
        raise
    finally:
        user.save_report()
        adapter = getattr(agent, "_adapter", None)
        if getattr(adapter, "model", None) == "gpt-live-1":
            (output / "live_session.json").write_text(
                json.dumps(
                    {
                        "model": adapter.model,
                        "backend_model": adapter.backend_model,
                        "session_id": adapter.session_id,
                        "usage": adapter.latest_usage,
                        "close_event": adapter.session_closed,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )


def main():
    """Write a scenario by default; require --run for paid live execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    scenario = (
        ThirdPartyScenario.model_validate_json(
            args.scenario.read_text(encoding="utf-8")
        )
        if args.scenario
        else ThirdPartyScenario()
    )
    output = args.output or Path("data/simulations") / datetime.now().strftime(
        "third_party_%Y%m%d_%H%M%S"
    )
    output.mkdir(parents=True, exist_ok=True)
    if args.run and any(
        (output / name).exists()
        for name in (
            "simulation.json",
            "error.json",
            "audio/events.json",
        )
    ):
        raise FileExistsError("Choose a fresh output directory for a new trial")
    scenario_path = output / "scenario.json"
    scenario_path.write_text(scenario.model_dump_json(indent=2), encoding="utf-8")
    if args.run:
        run(scenario, output)
    else:
        print(f"Prepared (no API calls): {scenario_path}")
        print(
            f'.venv\\Scripts\\python.exe -m experiments.tau_voice.third_party_interference --scenario "{scenario_path}" --output "{output}" --run'
        )


if __name__ == "__main__":
    main()
