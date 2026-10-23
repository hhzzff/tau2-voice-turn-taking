"""Run and export frozen-time voice demos on the existing orchestrator.

Launch with ``python -m tau2.scripts.open_model_demo --help``.
"""

import argparse
import hashlib
import html
import json
import math
import os
import time
import wave
from pathlib import Path

import numpy as np

from tau2.config import OPEN_MODEL_DEMO_SECONDS, OPEN_MODEL_DEMO_USER_LLM
from tau2.data_model.audio import AudioData
from tau2.data_model.simulation import AudioNativeConfig, OpenModelConfig
from tau2.user.open_model_scenarios import SCENARIOS, OpenModelScenarioUser
from tau2.voice.utils.audio_preprocessing import convert_to_pcm16, resample_audio


def _track(message, samples):
    raw = message.get_audio_bytes() if message and message.audio_content else b""
    if not raw:
        return raw, np.zeros(samples, dtype="<i2")
    audio = resample_audio(
        convert_to_pcm16(AudioData(data=raw, format=message.audio_format)), 8000
    )
    values = np.frombuffer(audio.data, dtype="<i2")
    if audio.format.channels != 1 or len(values) > samples:
        raise ValueError("Demo participant output exceeds one mono simulation tick")
    return raw, np.pad(values, (0, samples - len(values)))


def _active_frames(values):
    # A declared, simple 20ms energy detector; not a semantic speech classifier.
    frames = values.astype(float).reshape(-1, 160)
    return np.sqrt(np.mean(frames * frames, axis=1)) > 128


def export_demo(
    ticks,
    config,
    scenario,
    output,
    *,
    provenance,
    user,
    status,
    error=None,
    startup_seconds=0,
):
    """Write aligned playback and evidence; never infer semantics from overlap."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    samples = config.tick_duration_ms * 8
    rows, stereo = [], []
    previous = {"user": None, "agent": None}
    overlap_seconds = 0.0
    for index, tick in enumerate(ticks):
        uraw, u = _track(tick.user_chunk, samples)
        araw, a = _track(tick.agent_chunk, samples)
        ua, aa = _active_frames(u), _active_frames(a)
        overlap = float(np.count_nonzero(ua & aa) * 0.02)
        overlap_seconds += overlap
        raw_result = (tick.agent_chunk.raw_data or {}) if tick.agent_chunk else {}
        row = {
            "tick_index": index,
            "sim_start_seconds": index * config.tick_duration_seconds,
            "sim_end_seconds": (index + 1) * config.tick_duration_seconds,
            "computation_seconds": tick.wall_clock_duration_seconds,
            "initialization_interval": index == 0,
            "native_events": raw_result.get("events", []),
            "adapter_computation_seconds": raw_result.get(
                "computation_duration_seconds"
            ),
            "adapter_input_bytes": raw_result.get("audio_sent_bytes"),
            "playback_truncated": raw_result.get("was_truncated"),
            "playback_truncated_bytes": raw_result.get("truncated_audio_bytes"),
            "skip_item_id": raw_result.get("skip_item_id"),
            "interruption_audio_start_ms": raw_result.get(
                "interruption_audio_start_ms"
            ),
            "user_audio_sha256": hashlib.sha256(uraw).hexdigest(),
            "agent_audio_sha256": hashlib.sha256(araw).hexdigest(),
            "user_input_sha256": previous["agent"],
            "agent_input_sha256": previous["user"],
            "user_audio_bytes": len(uraw),
            "agent_audio_bytes": len(araw),
            "user_active_seconds": float(np.count_nonzero(ua) * 0.02),
            "agent_active_seconds": float(np.count_nonzero(aa) * 0.02),
            "overlap_seconds": overlap,
            "user_intended_text": tick.user_chunk.content if tick.user_chunk else None,
            "model_text_metadata": tick.agent_chunk.content
            if tick.agent_chunk
            else None,
            "model": config.model_identity,
            "tick_duration_ms": config.tick_duration_ms,
            "scenario": scenario,
        }
        rows.append(row)
        stereo.append(np.column_stack((u, a)).astype("<i2").tobytes())
        previous = {
            "user": row["user_audio_sha256"],
            "agent": row["agent_audio_sha256"],
        }
    events = []
    for requested in getattr(user, "events", []):
        event = dict(requested)
        start = event["tick_index"]
        active = []
        for row in rows[start:]:
            if row["user_active_seconds"] <= 0:
                break
            active.append(row)
        overlap = sum(row["overlap_seconds"] for row in active)
        if scenario in {"correction", "user_backchannel"}:
            observed = overlap > 0
        elif scenario == "model_backchannel":
            observed = (
                len(active) >= 2
                and sum(row["user_active_seconds"] for row in active) >= 1.0
            )
        else:
            observed = bool(active) and any(
                row["agent_active_seconds"] > 0 for row in rows[:start]
            )
        event.update(
            trigger_requested=True,
            observed_trigger=observed,
            observed_overlap_seconds=overlap,
            user_active_seconds=sum(row["user_active_seconds"] for row in active),
        )
        events.append(event)
    provenance = dict(provenance)
    runtime = getattr(getattr(user, "demo_adapter", None), "provenance", None)
    if runtime:
        provenance["worker"] = runtime
    report = {
        "status": status,
        "error": error,
        "model": config.model_identity,
        "tick_duration_ms": config.tick_duration_ms,
        "pacing": config.pacing,
        "scenario": scenario,
        "scenario_intent": SCENARIOS[scenario],
        "trigger_requested": bool(events),
        "scenario_triggered": any(event["observed_trigger"] for event in events),
        "trigger_events": events,
        "observed_overlap_seconds": overlap_seconds,
        "semantic_outcome": "not_assessed",
        "semantic_note": "Listen to the separate channels to assess correction handling or backchannels. Audio overlap alone does not establish either behavior.",
        "energy_detector": {"frame_ms": 20, "pcm16_rms_threshold": 128},
        "asr_events": getattr(user, "asr_events", []),
        "user_timing_ticks": {
            name: getattr(user, name, None)
            for name in [
                "wait_to_respond_threshold_other",
                "wait_to_respond_threshold_self",
                "yield_threshold_when_interrupted",
                "yield_threshold_when_interrupting",
                "interruption_check_interval",
                "integration_ticks",
            ]
        },
        "user_policy": "deterministic scenario actions on VoiceStreamingUserSimulator",
        "ticks": len(rows),
        "sim_duration_seconds": len(rows) * config.tick_duration_seconds,
        "startup_seconds": startup_seconds,
        "provenance": provenance,
        "channels": {"left": "user", "right": "model"},
    }
    (output / "ticks.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    with wave.open(str(output / "conversation.wav"), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"".join(stereo))
    table = "".join(
        f'<tr><td><button onclick="p.currentTime={r["sim_start_seconds"]};p.play()">{r["tick_index"]}</button></td>'
        f"<td>{r['sim_start_seconds']:.2f}–{r['sim_end_seconds']:.2f}</td>"
        f"<td>{r['user_active_seconds']:.2f}</td><td>{r['agent_active_seconds']:.2f}</td>"
        f"<td>{r['overlap_seconds']:.2f}</td><td>{r['computation_seconds']}</td></tr>"
        for r in rows
    )
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>τ-voice demo</title>'
        "<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto}td,th{padding:.4rem;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}button{cursor:pointer}</style>"
        f"<h1>{html.escape(config.model_identity)} · {html.escape(scenario)}</h1>"
        "<p>Left: user · Right: model. Simulation time freezes during computation. Click a tick to seek.</p>"
        '<audio id="p" controls src="conversation.wav"></audio>'
        "<p>Speech activity uses audio energy. Semantic outcomes require listening.</p>"
        "<table><thead><tr><th>Tick</th><th>Simulation seconds</th><th>User active s</th><th>Model active s</th><th>Overlap s</th><th>Compute s</th></tr></thead>"
        f"<tbody>{table}</tbody></table><h2>Run evidence</h2><pre>{html.escape(json.dumps(report, indent=2, ensure_ascii=False))}</pre>"
    )
    return report


def run_demo(orchestrator, config, scenario, output, *, steps, provenance):
    """Drive the established initialize/step seam and retain partial failure evidence."""
    if scenario not in SCENARIOS or steps < 0:
        raise ValueError("Unknown scenario or negative step count")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Demo output directory must be empty to preserve evidence")
    started = time.perf_counter()
    startup = 0
    status, error = "completed", None
    try:
        orchestrator.initialize()
        startup = time.perf_counter() - started
        for _ in range(steps):
            orchestrator.step()
            if orchestrator.done:
                break
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report = export_demo(
            orchestrator.get_ticks(),
            config,
            scenario,
            output,
            provenance=provenance,
            user=orchestrator.user,
            status=status,
            error=error,
            startup_seconds=startup,
        )
    return report


def source_hashes():
    """Hash the controller and worker source used by this checkout."""
    root = Path(__file__).resolve().parents[3]
    files = [
        Path(__file__),
        root / "src/tau2/user/open_model_scenarios.py",
        root / "src/tau2/utils/codex_llm.py",
        root / "src/tau2/utils/llm_utils.py",
        root / "src/tau2/user/user_simulator_streaming.py",
        root / "src/tau2/orchestrator/full_duplex_orchestrator.py",
        root / "src/tau2/agent/discrete_time_audio_native_agent.py",
        root / "src/tau2/voice/audio_native/adapter.py",
        root / "src/tau2/voice/audio_native/tick_result.py",
        root / "src/tau2/config.py",
        root / "src/tau2/data_model/simulation.py",
        root / "src/tau2/voice/transcription/transcribe.py",
    ]
    files += sorted((root / "scripts/open_model_workers").glob("*.py"))
    files += sorted((root / "src/tau2/voice/audio_native/open_model").glob("*.py"))
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def build_demo_user(config, scenario):
    """Build the real τ user with deterministic scenarios and selected providers."""
    from tau2.data_model.audio_effects import (
        ChannelEffectsConfig,
        SourceEffectsConfig,
        SpeechEffectsConfig,
    )
    from tau2.data_model.voice import (
        SpeechEnvironment,
        SynthesisConfig,
        VoiceSettings,
        VolcengineTTSConfig,
    )

    timing = AudioNativeConfig(
        tick_duration_seconds=config.tick_duration_seconds, round_thresholds_up=True
    )
    voice = VoiceSettings(
        transcription_config=None,
        synthesis_config=SynthesisConfig(
            provider="volcengine",
            provider_config=VolcengineTTSConfig(),
            speech_effects_config=SpeechEffectsConfig(
                enable_dynamic_muffling=False,
                enable_vocal_tics=False,
                enable_non_directed_phrases=False,
            ),
            source_effects_config=SourceEffectsConfig(
                enable_background_noise=False, enable_burst_noise=False
            ),
            channel_effects_config=ChannelEffectsConfig(enable_frame_drops=False),
        ),
        speech_environment=SpeechEnvironment(voice_seed=300),
    )
    user = OpenModelScenarioUser(
        tools=[],
        instructions=SCENARIOS[scenario]
        + " Speak naturally in English. Do not call tools.",
        scenario=scenario,
        llm=OPEN_MODEL_DEMO_USER_LLM,
        llm_args={},
        chunk_size=timing.user_chunk_size,
        voice_settings=voice,
        tick_duration_seconds=config.tick_duration_seconds,
        use_llm_backchannel=False,
        wait_to_respond_threshold_other=timing.wait_to_respond_threshold_other_ticks,
        wait_to_respond_threshold_self=timing.wait_to_respond_threshold_self_ticks,
        yield_threshold_when_interrupted=max(
            2, timing.yield_threshold_when_interrupted_ticks
        ),
        yield_threshold_when_interrupting=max(
            2, timing.yield_threshold_when_interrupting_ticks
        ),
        interruption_check_interval=timing.interruption_check_interval_ticks,
        integration_ticks=timing.integration_ticks,
    )
    return user


def main(argv=None):
    """Validate model granularity before connecting the user or GPU worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["salmonn", "minicpm"])
    parser.add_argument("--tick-ms", type=int)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--mailbox", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=OPEN_MODEL_DEMO_SECONDS)
    parser.add_argument("--runtime-config", type=Path)
    args = parser.parse_args(argv)
    config = OpenModelConfig(
        model=args.model,
        **(
            {"tick_duration_seconds": args.tick_ms / 1000}
            if args.tick_ms is not None
            else {}
        ),
    )
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    for key in ["VOLCENGINE_TTS_API_KEY", "VOLCENGINE_ASR_API_KEY"]:
        if not os.getenv(key):
            parser.error(f"{key} is required on the authenticated controller")
    from tau2.agent.discrete_time_audio_native_agent import DiscreteTimeAudioNativeAgent
    from tau2.data_model.tasks import Task, UserScenario
    from tau2.domains.mock.environment import get_environment
    from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator
    from tau2.voice.audio_native.open_model.adapter import OpenModelAdapter
    from tau2.voice.audio_native.open_model.transport import MailboxTransport

    user = build_demo_user(config, args.scenario)
    runtime = (
        json.loads(args.runtime_config.read_text()) if args.runtime_config else None
    )
    adapter = OpenModelAdapter(
        config, runtime_config=runtime, transport=MailboxTransport(args.mailbox)
    )
    user.demo_adapter = adapter
    agent = DiscreteTimeAudioNativeAgent(
        tools=[],
        domain_policy="Have a helpful natural spoken conversation. No tools are available.",
        tick_duration_ms=config.tick_duration_ms,
        adapter=adapter,
        provider=config.model,
        model=config.model_identity,
        vad_config={},
        max_inactive_seconds=0,
    )
    orchestrator = FullDuplexOrchestrator(
        domain="mock",
        agent=agent,
        user=user,
        environment=get_environment(),
        task=Task(
            id=f"demo-{args.scenario}",
            user_scenario=UserScenario(instructions=SCENARIOS[args.scenario]),
        ),
        tick_duration_seconds=config.tick_duration_seconds,
        pacing="frozen",
    )
    try:
        run_demo(
            orchestrator,
            config,
            args.scenario,
            args.output,
            steps=max(0, math.ceil(args.seconds / config.tick_duration_seconds) - 1),
            provenance={
                "runtime_config": adapter.runtime_config,
                "user_llm": OPEN_MODEL_DEMO_USER_LLM,
                "tts": "volcengine",
                "asr": "doubao-asr",
                "seed": 300,
                "mailbox": str(args.mailbox),
                "threshold_rounding": "ceil",
                "requested_seconds": args.seconds,
                "source_sha256": source_hashes(),
            },
        )
    finally:
        adapter.disconnect()
        user.stop(state=getattr(orchestrator, "user_state", None))


if __name__ == "__main__":
    main()
