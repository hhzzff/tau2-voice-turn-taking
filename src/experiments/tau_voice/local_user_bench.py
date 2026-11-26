"""Controlled local-user trials through tau's continuous driver and provider."""

import asyncio
import hashlib
import inspect
import json
import os
import subprocess
import traceback
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
from dotenv import load_dotenv

from tau2.config import DEFAULT_LOCAL_USER_TARGET_OUTPUT_TOKENS
from tau2.orchestrator.continuous_voice import ContinuousVoiceDriver
from tau2.user.local.config import LocalUserConfig
from tau2.user.local.recording import Recorder, now
from tau2.user.local.scenario import ScenarioRuntime
from tau2.user.local.scenarios import SCENE_DIR, SCENE_EVENTS, load_clip, load_scenes
from tau2.user.local_streaming_user import LocalStreamingUser
from tau2.voice.audio_native.openai.continuous import ContinuousOpenAITarget
from tau2.voice.audio_native.openai.provider import OpenAIRealtimeProvider

ROOT = Path(__file__).resolve().parents[3]
TARGET_INSTRUCTIONS = (
    "你是自然、友好的中文语音助手，帮助用户讨论旅行或餐食规划。认真听用户的新要求，"
    "记住之前的约束，并使用更正后的最新信息。需要补充信息时一次问一两个问题。"
    "通常每次说两三句，约40到80个汉字；用户要求总结时可以稍长，要求简短时就简短。"
    "用户是在做方案讨论，你没有查询、支付、订票或下单工具，不要声称查到了真实库存、"
    "实时价格或已完成购买。可以明确说明是假设或估算。被要求停下时停止，"
    "不要把用户的附和当作新问题，不要把对旁人的话当成对你的要求。"
)


def digest(value):
    """Stable UTF-8 JSON digest for configuration and frozen scenes."""
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def provenance(cfg, case, model):
    """Capture actual imported source and reproducible trial inputs, no secrets."""
    import importlib.metadata

    import tau2.user.local.llm as prompts
    import tau2.user.local.runtime as runtime

    return {
        "commit": git_output("rev-parse", "HEAD"),
        "dirty": git_output("status", "--porcelain"),
        "branch": git_output("branch", "--show-current"),
        "source_root": str(ROOT),
        "runtime_import": runtime.__file__,
        "scenario_sha256": digest(case),
        "prompt_source_sha256": hashlib.sha256(
            Path(prompts.__file__).read_bytes()
        ).hexdigest(),
        "config": cfg.model_dump(mode="json"),
        "config_sha256": digest(cfg.model_dump(mode="json")),
        "target_model": model,
        "event_type": SCENE_EVENTS[case["id"]],
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("sherpa-onnx", "soxr", "websockets", "httpx", "soundfile")
        },
        "delivery_boundary": "client_transport_submit, not remote hearing acknowledgement",
        "target_text_visibility": "audit-only; B gets local ASR of paced played PCM",
    }


def save_tracks(runtime, path, length):
    """Write external lossless audio; keep wall-clock and listening views."""
    for timing in (False, True):
        user = runtime.user_track.render(length, timing=timing)
        agent = runtime.agent_track.render(length, timing=timing)
        n = max(len(user), len(agent))
        user = np.pad(user, (0, n - len(user)))
        agent = np.pad(agent, (0, n - len(agent)))
        stereo = np.column_stack((user, agent))
        sf.write(
            path / ("stereo-timing.flac" if timing else "stereo.flac"),
            stereo,
            16000,
            subtype="PCM_16",
        )
        if not timing:
            sf.write(path / "user.wav", user, 16000, subtype="PCM_16")
            sf.write(path / "agent.flac", agent, 16000, subtype="PCM_16")
            mix = np.clip(
                user.astype(np.int32) + agent.astype(np.int32), -32768, 32767
            ).astype("int16")
            sf.write(path / "mix.flac", mix, 16000, subtype="PCM_16")


async def run_case(cfg, case, output, model):
    """Preserve partial evidence and attempt every cleanup, including on cancellation.

    Cleanup and export failures make a trial fail without hiding its first error.
    Journal failures propagate: a trial without durable evidence cannot succeed.
    """
    rec = Recorder(output)
    status, error, origin = "error", None, None
    runtime = target = provider = client = None
    cancellation = None
    failures = []

    def record_error(stage, exc):
        nonlocal status, error
        if status != "cancelled":
            status = "error"
        error = error or type(exc).__name__
        failure = {
            "stage": stage,
            "error_class": type(exc).__name__,
            "message": str(exc)[:600],
            "stack": [
                {"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
                for f in traceback.extract_tb(exc.__traceback__)
            ],
        }
        failures.append(failure)
        try:
            rec.emit("live_error", **failure)
        except Exception as journal_error:
            failures.append(
                {"stage": "error_journal", "error_class": type(journal_error).__name__}
            )

    async def finalize(stage, operation, *args):
        try:
            result = operation(*args)
            return await result if inspect.isawaitable(result) else result
        except Exception as exc:
            record_error(stage, exc)
            return None

    try:
        try:
            rec.save_json("provenance.json", provenance(cfg, case, model))
            rec.save_json("case.json", case)
            client = httpx.AsyncClient(timeout=cfg.api_timeout_seconds, trust_env=False)
            runtime = ScenarioRuntime(cfg, case, rec, client)
            await runtime.start()
            # All stimuli are verified/preloaded before clocks, avoiding CPFS reads live.
            for ident in json.loads(
                (SCENE_DIR / "audio" / "manifest.json").read_text()
            )["clips"]:
                load_clip(ident)
            user = LocalStreamingUser(runtime)
            provider = OpenAIRealtimeProvider(
                api_key=os.environ.get("OPENAI_REALTIME_API_KEY")
                or os.environ.get("OPENAI_API_KEY"),
                model=model,
            )
            target = ContinuousOpenAITarget(provider, user, rec)
            async with asyncio.timeout(45):
                await target.start(
                    case.get("target_instructions", TARGET_INSTRUCTIONS),
                    max_output_tokens=case.get(
                        "target_max_output_tokens",
                        DEFAULT_LOCAL_USER_TARGET_OUTPUT_TOKENS,
                    ),
                )
            origin = rec.emit("live_start", model=model, scene_id=case["id"])
            runtime.begin_live(origin)
            driver = ContinuousVoiceDriver(user, target, rec)
            status = await driver.run(case["max_seconds"])
            if status == "ok" and runtime.progress.error:
                status, error = "incomplete", runtime.progress.error
        except asyncio.CancelledError as exc:
            cancellation = exc
            status, error = "cancelled", "CancelledError"
        except Exception as exc:
            record_error("run", exc)
        finally:
            duration = (now() - origin) / 1e9 if origin is not None else 0
            if target:
                await finalize("finish_playback", target.finish_playback)
            if runtime:
                await finalize("runtime_close", runtime.close)
            if provider:
                await finalize("provider_disconnect", provider.disconnect)
            if client:
                await finalize("client_close", client.aclose)
        extra = {}
        if origin is not None:
            await finalize(
                "save_tracks",
                asyncio.to_thread,
                save_tracks,
                runtime,
                output,
                round(duration * 16000),
            )
            extra = (
                await finalize(
                    "additional_summary", asyncio.to_thread, runtime.additional_summary
                )
                or {}
            )
            await finalize(
                "recording_layout",
                rec.save_json,
                "recording-layout.json",
                {
                    "agent": runtime.agent_track.manifest(),
                    "user": runtime.user_track.manifest(),
                    "listening": "stereo.flac: original sample clock within each segment",
                    "timing": "stereo-timing.flac: transmission/playback timestamps, gaps preserved",
                },
            )
        summary = {
            "id": case["id"],
            "title": case["title"],
            "status": status,
            "error_class": error,
            "failures": failures,
            "duration_s": duration,
            "observations_drained": getattr(runtime, "observations_drained", False),
            "origin_ns": origin,
            "turns": [runtime.serial_turn(t) for t in runtime.turns] if runtime else [],
            "backchannels": [runtime.serial_turn(t) for t in runtime.backchannels]
            if runtime
            else [],
            "aborted_before_voice": [
                runtime.serial_turn(t) for t in runtime.aborted_turns
            ]
            if runtime
            else [],
            "history": runtime.history if runtime else [],
            "script_progress": runtime.progress.manifest() if runtime else None,
            "target_responses": list(runtime.playback.responses.values())
            if runtime
            else [],
            "target_transcripts_audit_only": target.transcripts if target else {},
            **extra,
        }
        rec.save_json("summary.json", summary)
        rec.emit("live_end", status=status, duration_s=duration)
    finally:
        try:
            await rec.close()
        finally:
            if cancellation is not None:
                raise cancellation
    return summary


async def run_suite(args):
    """Sequential trials avoid contention between B/TTS latency measurements."""
    load_dotenv(args.env_file)
    cfg = LocalUserConfig(asr={"model_path": args.asr_model_path})
    selected = set(args.scenes.split(",")) if args.scenes else set(SCENE_EVENTS)
    if unknown := selected - set(SCENE_EVENTS):
        raise ValueError(f"Unknown scene IDs: {sorted(unknown)}")
    if args.max_seconds is not None and args.max_seconds <= 0:
        raise ValueError("--max-seconds must be positive")
    if args.target_max_output_tokens is not None and args.target_max_output_tokens <= 0:
        raise ValueError("--target-max-output-tokens must be positive")
    if args.output.exists():
        raise FileExistsError(
            "Use a new output directory; trials are never overwritten"
        )
    args.output.mkdir(parents=True)
    async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
        models = await client.get(cfg.llm.base_url + "/models")
        models.raise_for_status()
        health = await client.get(cfg.tts.url.rsplit("/", 1)[0] + "/health")
        health.raise_for_status()
        if health.json().get("reference_sha256") != cfg.tts.reference_sha256:
            raise ValueError("TTS is not using the pinned reference speaker")
        (args.output / "services.json").write_text(
            json.dumps(
                {"llm": models.json(), "tts": health.json()},
                ensure_ascii=False,
                indent=2,
            )
        )
    results = []
    for case in load_scenes():
        if case["id"] not in selected:
            continue
        if args.max_seconds is not None:
            case["max_seconds"] = args.max_seconds
        if args.target_max_output_tokens is not None:
            case["target_max_output_tokens"] = args.target_max_output_tokens
        print(f"START {case['id']} {case['title']}", flush=True)
        result = await run_case(cfg, case, args.output / case["id"], args.model)
        results.append(
            {k: result[k] for k in ("id", "status", "error_class", "duration_s")}
        )
        (args.output / "suite.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2)
        )
        print(
            f"END {case['id']} {result['status']} {result['error_class']} {result['duration_s']:.1f}s",
            flush=True,
        )
    return 1 if any(r["status"] != "ok" for r in results) else 0
