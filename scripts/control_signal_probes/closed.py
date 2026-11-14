"""Probe real endpoints using generic synthetic repository test fixtures."""

import argparse
import audioop
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PI_BENCH = Path("/mnt/bn/audio-visual-llm-data6/terumi/pi-bench")
sys.path.insert(0, str(PI_BENCH))
from inference.realtime_tick import (  # noqa: E402
    TickConfig,
    TickSession,
    load_audio_segment_as_pcm,
)
from inference.realtime_tick_backend import FAMILIES  # noqa: E402
from inference.repo_credentials import load_repo_credentials  # noqa: E402

PROVIDERS = {
    "gpt_live": "gpt_live",
    "gpt_realtime": "gpt",
    "gemini_live": "gemini",
    "doubao_s2s": "doubao",
}
PROMPT = (
    "Speak English. When the user greets you, explain a three-day Beijing "
    "itinerary in detail for at least twenty seconds. Listen for corrections. "
    "A short yes is an acknowledgement, not a new request."
)


def run_case(model, scenario, root, model_override=None, after_speech_ms=0):
    family = PROVIDERS[model]
    model_id, endpoint, rate, names = FAMILIES[family]
    model_id = model_override or model_id
    config = TickConfig(
        family,
        model_id,
        endpoint,
        input_rate=rate,
        output_rate=24000,
        tick_ms=100,
        transport_ms=20,
        realtime_upload=True,
        observe_seconds=0,
        tail_ms=30000,
        completion_timeout=30,
    )
    key = next((os.environ[n] for n in names if os.environ.get(n)), "")
    directory = root / model / scenario
    session = TickSession(config, key, PROMPT, directory)
    greeting = load_audio_segment_as_pcm(str(root / "stimuli/greeting.wav"), rate)
    interjection = (
        load_audio_segment_as_pcm(str(root / f"stimuli/{scenario}.wav"), rate)
        if scenario != "normal"
        else b""
    )
    size = rate * 2 // 10
    sent = bytearray()
    trigger = None
    status, error, setup_ack = "failed", None, False
    after = 0

    def send(pcm):
        pcm = pcm.ljust(size, b"\0")
        sent.extend(pcm)
        if family == "gpt_live":
            session._send_pcm(pcm, source=True, burst=True)
        else:
            session.run_tick(pcm, source=True)

    try:
        session.connect()
        setup_ack = True
        for i in range(0, len(greeting), size):
            send(greeting[i : i + size])
        for _ in range(350):
            with session.cv:
                raw = bytes(session.raw)
                recent = raw[-4800:]
                audible = session.first_voice is not None
                buffered = session.queue.size
            if (
                scenario != "normal"
                and trigger is None
                and audible
                and session.stream_ms
                >= session.first_voice["stream_ms"] + after_speech_ms
                and audioop.rms(recent, 2) > 200
            ):
                trigger = {
                    "input_media_ms": session.stream_ms,
                    "raw_output_samples": len(raw) // 2,
                    "latest_100ms_rms": audioop.rms(recent, 2),
                    "queued_output_bytes": buffered,
                }
                session.emit("probe_interjection", scenario=scenario, **trigger)
                for i in range(0, len(interjection), size):
                    send(interjection[i : i + size])
            send(bytes(size))
            if trigger is not None:
                after += 1
                if after >= 150:
                    break
            elif scenario == "normal" and session.complete:
                break
        if scenario != "normal" and trigger is None:
            raise RuntimeError("No audible model output; interjection not injected")
        if session.first_voice is None:
            raise RuntimeError("No audible model output")
        status = "observed"
    except Exception as exc:
        error = str(exc).replace(key, "[REDACTED]") if key else str(exc)
    finally:
        session.closing = True
        if session.ws is not None and session.ws.connected:
            try:
                if family in {"doubao", "gpt_live"}:
                    session._send({"type": "session.close"})
                    session.close_ack.wait(15 if family == "gpt_live" else 3)
                session.ws.close(1000, "control signal probe finished")
            except Exception:
                pass
        session.closed.wait(2)
        if family == "gpt_live":
            session.playback = bytearray(session.raw)
        else:
            session._drain_playback()
        session.source_duration_ms = len(sent) * 500 / rate
        session.save(bytes(sent), status, error or "probe_complete", 0)
        session._events.close()
    counts = Counter()
    for line in (directory / "scheduler-events.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["kind"] == "provider_event":
            event = row["event"]
            counts[
                event.get("type")
                or (
                    "serverContent"
                    if "serverContent" in event
                    else next(iter(event), "unknown")
                )
            ] += 1
    raw = bytes(session.raw)
    frames = [raw[i : i + 4800] for i in range(0, len(raw), 4800)]
    report = {
        "model": model,
        "model_id": model_id,
        "endpoint": endpoint,
        "scenario": scenario,
        "status": status,
        "error": error,
        "trigger": trigger,
        "events": dict(counts),
        "controls": session.controls,
        "raw_output_seconds": len(raw) / 48000,
        "silent_output_100ms_frames_rms_le_100": sum(
            bool(x) and audioop.rms(x, 2) <= 100 for x in frames
        ),
        "audio_frames": len(frames),
        "input_seconds": len(sent) / (rate * 2),
        "setup_acknowledged": setup_ack,
    }
    (directory / "probe.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("model", "scenario", "status", "error", "raw_output_seconds")
            }
        ),
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scenarios", nargs="+", default=["normal", "interruption", "backchannel"]
    )
    parser.add_argument("--models", nargs="+", default=list(PROVIDERS))
    parser.add_argument("--after-speech-ms", type=int, default=0)
    parser.add_argument(
        "--model-id", help="Explicit model ID; requires a single provider"
    )
    args = parser.parse_args()
    if args.model_id and len(args.models) != 1:
        parser.error("--model-id requires one provider")
    load_repo_credentials()

    def run_model(model):
        results = []
        for scenario in args.scenarios:
            results.append(
                run_case(
                    model, scenario, args.output, args.model_id, args.after_speech_ms
                )
            )
            if (
                results[-1]["status"] == "failed"
                and not results[-1]["raw_output_seconds"]
            ):
                break
        return results

    with ThreadPoolExecutor(max_workers=4) as executor:
        reports = list(executor.map(run_model, args.models))
    (args.output / "closed-summary.json").write_text(
        json.dumps(reports, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
