"""Real FullDuplexOrchestrator versus directly paced native open models.

No closed API requests. Scheduled output is the actual saved tick audio, not a
rounded reconstruction. Native output retains measured inference availability.
"""

import argparse
import audioop
import base64
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

WIDTHS = {"salmonn": 80, "moshi": 80, "minicpm": 1000, "freeze_omni": 160}
IDENTITIES = {
    "moshi": "Moshika-pytorch-bf16",
    "freeze_omni": "Freeze-Omni",
    "minicpm": "MiniCPM-o-4.5",
    "salmonn": "SALMONN-Omni-0519-fc180-aes-er0-chunk30-llama31-checkpoint-44000",
}
PROMPT = "Speak English. Follow the spoken instructions. Count slowly when asked. Stop counting when the user asks a new question and answer it briefly."


class ReplayUser:
    """A fixed audio source, independent of model output and without API calls."""

    def __init__(self, audio, width):
        self.audio, self.size = audio, width * 8

    def get_init_state(self, message_history=None):
        return 0

    def get_next_chunk(self, state, participant_chunk=None, tool_results=None):
        from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT
        from tau2.data_model.message import UserMessage

        raw = self.audio[state * self.size : (state + 1) * self.size].ljust(
            self.size, b"\xff"
        )
        return UserMessage(
            role="user",
            content="",
            is_audio=True,
            audio_format=TELEPHONY_AUDIO_FORMAT,
            audio_content=base64.b64encode(raw).decode(),
            contains_speech=audioop.rms(audioop.ulaw2lin(raw, 2), 2) > 100,
            chunk_id=state,
            is_final_chunk=True,
        ), state + 1


def build_scheduler(
    model, audio, runtime_config, transport, *, realtime=False, window_ms=100
):
    """Build actual production step/agent/adapter; only user and runtime are injected."""
    from tau2.agent.discrete_time_audio_native_agent import DiscreteTimeAudioNativeAgent
    from tau2.domains.mock.environment import get_environment, get_tasks
    from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator
    from tau2.voice.audio_native.open_model.adapter import OpenModelAdapter

    width = window_ms if realtime else WIDTHS[model]
    # Moshi/Freeze are experimental runtime bindings to the existing adapter;
    # do not silently advertise them as supported production configuration.
    config = SimpleNamespace(
        model=model,
        model_identity=IDENTITIES[model],
        tick_duration_ms=width,
        pacing="realtime" if realtime else "frozen",
    )
    if realtime:
        from realtime_adapter import RealtimeOpenModelAdapter

        adapter = RealtimeOpenModelAdapter(
            config, runtime_config, transport, native_width_ms=WIDTHS[model]
        )
    else:
        adapter = OpenModelAdapter(config, runtime_config, transport)
    agent = DiscreteTimeAudioNativeAgent(
        tools=[],
        domain_policy=PROMPT,
        tick_duration_ms=width,
        adapter=adapter,
        provider=model,
        model=config.model_identity,
        vad_config={},
        max_inactive_seconds=0,
    )
    agent._owns_adapter = True
    task = next(t for t in get_tasks() if t.id == "create_task_1")
    scheduler = FullDuplexOrchestrator(
        domain="mock",
        agent=agent,
        user=ReplayUser(audio, width),
        environment=get_environment(),
        task=task,
        tick_duration_seconds=width / 1000,
        pacing="realtime" if realtime else "frozen",
        max_steps=10000,
    )
    return scheduler, agent


class RecordingTransport:
    """Record requests at the public transport boundary without changing them."""

    def __init__(self, transport, directory):
        self.transport, self.directory = transport, directory
        self.rows, self.raw = [], bytearray()
        self.inputs = bytearray()
        self.origin = None

    def request(self, op, **payload):
        sent = time.monotonic()
        result = self.transport.request(op, **payload)
        returned = time.monotonic()
        if op == "connect":
            (self.directory / "provenance.json").write_text(
                json.dumps(result, indent=2)
            )
            (self.directory / "prompt.txt").write_text(payload["system_prompt"])
        if op == "step":
            raw = base64.b64decode(result["audio_pcm16_b64"])
            pcm = base64.b64decode(payload["audio_pcm16_b64"])
            row = {k: v for k, v in result.items() if k != "audio_pcm16_b64"}
            row.update(
                index=len(self.rows),
                request_monotonic=sent,
                returned_monotonic=returned,
                adapter_input_sha256=hashlib.sha256(pcm).hexdigest(),
                raw_offset=len(self.raw),
                raw_bytes=len(raw),
            )
            self.raw.extend(raw)
            self.inputs.extend(pcm)
            self.rows.append(row)
            with (self.directory / "native.jsonl").open("a") as out:
                out.write(json.dumps(row) + "\n")
        return result

    def close(self):
        self.transport.close()


def converted_inputs(audio, width):
    """Match the existing public telephony converter and its first sample policy."""
    from tau2.voice.audio_native.audio_converter import StreamingTelephonyConverter

    converter = StreamingTelephonyConverter(16000, 24000)
    for offset in range(0, len(audio), width * 8):
        pcm = converter.convert_input(audio[offset : offset + width * 8])
        if len(pcm) == width * 32 - 2:
            pcm += pcm[-2:]
        if len(pcm) != width * 32:
            raise ValueError("Incomplete input interval")
        yield pcm


def main():
    sys.path.insert(0, str(Path(__file__).parent))
    from equivalence import activity, read_wav, render, write_wav

    from tau2.voice.audio_native.audio_converter import StreamingTelephonyConverter
    from tau2.voice.audio_native.open_model.transport import SubprocessTransport

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", choices=WIDTHS, required=True)
    parser.add_argument(
        "--mode",
        choices=["direct", "scheduled", "realtime", "native_realtime"],
        required=True,
    )
    parser.add_argument("--run", required=True, type=int)
    args = parser.parse_args()
    realtime = args.mode in {"realtime", "native_realtime"}
    root = args.root.resolve()
    config = json.loads((root / "configs" / f"{args.model}.json").read_text())
    directory = root / args.model / f"{args.run}-{args.mode}"
    directory.mkdir(parents=True, exist_ok=False)
    width = 100 if args.mode == "realtime" else WIDTHS[args.model]
    pcm8 = read_wav(root / "input.wav", 8000)
    audio = audioop.lin2ulaw(pcm8, 2)
    size = width * 8
    audio += b"\xff" * (-len(audio) % size)
    (directory / "input.ulaw").write_bytes(audio)
    command = [
        config["python"],
        str(Path(__file__).with_name("open_equivalence_worker.py")),
        "--model",
        args.model,
        "--mode",
        "scheduled" if realtime else args.mode,
    ]
    transport = RecordingTransport(
        SubprocessTransport(command, timeout_seconds=600), directory
    )
    scheduler, agent = build_scheduler(
        args.model, audio, config, transport, realtime=realtime, window_ms=width
    )
    events, raw_playback, tick_audio = [], bytearray(), bytearray()
    error = None
    try:
        if args.mode in {"scheduled", "realtime", "native_realtime"}:
            scheduler.initialize()
            origin = time.monotonic()
            tick_audio.extend(
                base64.b64decode(scheduler.get_ticks()[0].agent_chunk.audio_content)
            )
            for _ in range(len(audio) // size):
                scheduler.step()
                tick = scheduler.get_ticks()[-1]
                tick_audio.extend(base64.b64decode(tick.agent_chunk.audio_content))
                row = {
                    "tick": tick.tick_id,
                    "effective_start_ms": tick.tick_id * width,
                    "wall_duration_ms": tick.wall_clock_duration_seconds * 1000,
                    "delivered_monotonic": time.monotonic(),
                    "text": tick.agent_chunk.content,
                    "events": agent_state_events(scheduler),
                }
                with (directory / "ticks.jsonl").open("a") as out:
                    out.write(json.dumps(row, default=str) + "\n")
            if realtime:
                # Stop submitting model input after the fixed 40s; drain pending
                # results and normal playback tails through the same real step.
                agent.adapter.draining = True
                drain_start = time.monotonic()
                for _ in range((60000 + width - 1) // width):
                    scheduler.step()
                    tick = scheduler.get_ticks()[-1]
                    tick_audio.extend(base64.b64decode(tick.agent_chunk.audio_content))
                    if (
                        not agent.adapter.pending_native_steps
                        and time.monotonic() - drain_start >= 5
                    ):
                        break
                if agent.adapter.pending_native_steps:
                    raise TimeoutError("Native inference did not drain within 60s")
            played = audioop.ulaw2lin(bytes(tick_audio), 2)
        else:
            provenance = transport.request(
                "connect", config=config, system_prompt=agent.system_prompt
            )
            converter = StreamingTelephonyConverter(16000, provenance["sample_rate"])
            origin = time.monotonic()
            identity, ended = 0, True
            for index, pcm in enumerate(converted_inputs(audio, width)):
                time.sleep(
                    max(0, origin + (index + 1) * width / 1000 - time.monotonic())
                )
                event = transport.request(
                    "step", audio_pcm16_b64=base64.b64encode(pcm).decode()
                )
                at = (event["audit"]["ready_monotonic"] - origin) * 1000
                raw = base64.b64decode(event["audio_pcm16_b64"])
                output = audioop.ulaw2lin(converter.convert_output(raw), 2)
                if (
                    args.model == "minicpm"
                    and event.get("is_listen")
                    and not ended
                    and not event.get("end_of_turn")
                ):
                    events.append(
                        {"kind": "cancel", "at_ms": at, "response_id": str(identity)}
                    )
                    ended = True
                if any(
                    e.get("type") == "controller_stop_tts"
                    for e in event.get("events", [])
                ):
                    events.append(
                        {"kind": "cancel", "at_ms": at, "response_id": str(identity)}
                    )
                    ended = True
                if raw and any(raw):
                    if ended:
                        identity += 1
                        ended = False
                    events.append(
                        {
                            "kind": "audio",
                            "at_ms": at,
                            "response_id": str(identity),
                            "raw_offset_bytes": len(raw_playback),
                            "audio_bytes": len(output),
                        }
                    )
                    raw_playback.extend(output)
                if event.get("end_of_turn"):
                    ended = True
            horizon = max(
                len(audio) / 8 + 5000, (time.monotonic() - origin) * 1000 + 5000
            )
            played, _ = render(
                events, bytes(raw_playback), rate=8000, horizon_ms=horizon, quantum_ms=0
            )
        elapsed = (time.monotonic() - origin) * 1000
        write_wav(directory / "played.wav", played, 8000)
        regions = activity(played, 8000)
        result = {
            "model": args.model,
            "mode": args.mode,
            "run": args.run,
            "production_step": args.mode
            in {"scheduled", "realtime", "native_realtime"},
            "experimental_realtime_adapter": realtime,
            "native_width_ms": WIDTHS[args.model],
            "experimental_runtime_binding": args.model in {"moshi", "freeze_omni"},
            "input_sha256": hashlib.sha256(audio).hexdigest(),
            "width_ms": width,
            "first_voice_ms": regions[0][0] if regions else None,
            "speech_regions_ms": regions,
            "elapsed_wall_ms": elapsed,
            "steps": len(transport.rows),
            "text": "".join(r.get("text", "") for r in transport.rows),
            "post_interruption_voice_ms": next(
                (a for a, _ in regions if a >= 15200.25), None
            ),
            "status": "observed",
            "all_pass": False,
            "baseline_scope": "raw SDK/session/controller without tau scheduler; shared audited native reference for SALMONN/Moshi/Freeze",
        }
        (directory / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    except Exception as exc:
        error = repr(exc)
        (directory / "error.json").write_text(
            json.dumps({"error": error, "completed_steps": len(transport.rows)})
        )
        raise
    finally:
        (directory / "native-output.pcm16").write_bytes(transport.raw)
        (directory / "native-input-16k.pcm16").write_bytes(transport.inputs)
        (directory / "events.json").write_text(json.dumps(events, indent=2))
        if error is None:
            if args.mode in {"scheduled", "realtime", "native_realtime"}:
                agent.stop()
            else:
                transport.request("disconnect")
        transport.close()


def agent_state_events(scheduler):
    """Read the public last tick result after the production agent step."""
    return scheduler.agent_state.last_tick_result.serializable_events


if __name__ == "__main__":
    main()
