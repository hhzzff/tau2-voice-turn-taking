"""Real-weight streaming probes in each model's own Python environment."""

import argparse
import audioop
import base64
import importlib
import json
import sys
import time
import traceback
import wave
from collections import Counter, deque
from copy import deepcopy
from pathlib import Path

PI = Path("/mnt/bn/audio-visual-llm-data6/terumi/pi-bench")
PROMPT = (
    "Speak English. When greeted explain a relaxed three-day Beijing "
    "itinerary in detail. Listen to user corrections. A brief yes is an acknowledgement."
)


def read_pcm(path, rate):
    with wave.open(str(path), "rb") as handle:
        pcm = handle.readframes(handle.getnframes())
        assert handle.getsampwidth() == 2 and handle.getnchannels() == 1
        return audioop.ratecv(pcm, 2, 1, handle.getframerate(), rate, None)[0]


def write_wav(path, pcm, rate):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm)


class WorkerRuntime:
    def __init__(self, model, config):
        sys.path.insert(0, str(Path(__file__).parent / "workers"))
        module = importlib.import_module(
            "minicpm_worker" if model == "minicpm" else "salmonn_worker"
        )
        self.worker = (
            module.MiniCPMWorker if model == "minicpm" else module.SalmonnWorker
        )()
        self.provenance = self.worker.connect(
            config, config.get("probe_system_prompt", PROMPT)
        )
        self.rate = self.provenance.get(
            "sample_rate", 24000 if model == "minicpm" else 16000
        )
        self.input_rate = 16000
        self.width_ms = 1000 if model == "minicpm" else 80
        self.model = model

    def step(self, pcm, interrupt=False, force_listen=False):
        controls = {}
        if interrupt:
            controls["interrupt"] = True
        if force_listen:
            controls["force_listen"] = True
        event = self.worker.step(pcm, **controls)
        raw = base64.b64decode(event.pop("audio_pcm16_b64"))
        return raw, event

    def close(self):
        self.worker.disconnect()


class MoshiRuntime:
    def __init__(self, model, config):
        sys.path.insert(0, str(PI / "inference/moshi"))
        from _contract import Dataset
        from backend import MoshiBackend

        self.backend = MoshiBackend(
            config, Dataset(PI / "dataset/backchannel.json", "backchannel", None, [])
        )
        self.state = self.backend.state
        self.state.mimi.reset_streaming()
        self.state.lm_gen.reset_streaming()
        self.input_rate = self.rate = self.backend.sample_rate
        self.width_ms = self.backend.frame_size * 1000 / self.rate
        self.provenance = {
            "checkpoint": config["checkpoint"],
            "sample_rate": self.rate,
            "frame_size": self.backend.frame_size,
            "import_origins": self.backend.import_origins,
            "control_mode": "continuous_audio_frames",
        }
        self.first = True

    def step(self, pcm, interrupt=False):
        b, s = self.backend, self.state
        waveform = b.np.frombuffer(pcm, dtype="<i2").astype(b.np.float32) / 32768
        tensor = b.torch.from_numpy(waveform.copy()).to(b.device).reshape(1, 1, -1)
        codes = s.mimi.encode(tensor)
        if self.first:
            s.lm_gen.step(codes)
            self.first = False
        tokens = s.lm_gen.step(codes)
        if tokens is None:
            return b"", {"type": "native_frame", "warmup": True}
        audio = s.mimi.decode(tokens[:, 1:]).cpu().float().numpy().reshape(-1)
        pcm = (b.np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
        text_id = int(tokens[0, 0].item())
        return pcm, {
            "type": "native_frame",
            "text_token": text_id,
            "text_piece": s.text_tokenizer.id_to_piece(text_id),
            "text_eos": text_id == s.text_tokenizer.eos_id(),
        }

    def close(self):
        pass


class FreezeRuntime:
    """Interleave the original server's listening and generation state machines.

    Model stat=ss triggers the reference controller's stop_generate/stop_tts
    policy. That action is logged as external_policy, never native interruption.
    """

    def __init__(self, model, config):
        sys.path.insert(0, str(PI / "inference/freeze_omni"))
        from _contract import Dataset
        from backend import FreezeOmniBackend

        self.backend = FreezeOmniBackend(
            config, Dataset(PI / "dataset/backchannel.json", "backchannel", None, [])
        )
        b = self.backend
        b.vad.reset_vad()
        self.generation = b.pipeline.speech_dialogue(None, stat="pre", role=PROMPT)
        self.listening = None
        self.speaking = False
        self.started_generation = False
        self.input_rate = 16000
        self.rate = 24000
        self.width_ms = b.vad.get_chunk_size() * 1000 / self.input_rate
        self.hidden, self.pending = [], ""
        self.previous_text = ""
        self.generated = 0
        self.token_count = 0
        self.provenance = {
            "checkpoint": config["checkpoint"],
            "import_origins": b.import_origins,
            "control_mode": "native_stat_plus_reference_external_controller",
            "force_on_vad_end": False,
            "force_final_on_eof": False,
        }

    def step(self, pcm, interrupt=False):
        b = self.backend
        waveform = b.np.frombuffer(pcm, dtype="<i2").astype(b.np.float32) / 32768
        vad = b.vad.predict(waveform)
        events = []
        status = vad["status"]
        if status == "sl":
            self.listening = b._listen_outputs(self.generation)
            for index, feature in enumerate(vad["feature_last_chunk"]):
                self.listening = b._prefill(
                    self.listening, feature, "sl" if index == 0 else "cl", True
                )
            self.listening = b._prefill(self.listening, vad["feature"], "cl")
        elif status in {"cl", "el"} and self.listening is not None:
            if status == "cl":
                self.listening = b._prefill(self.listening, vad["feature"], "cl")
            else:
                b.vad.in_dialog = False
        if self.listening is not None:
            events.append(
                {
                    "type": "native_listen_stat",
                    "stat": self.listening["stat"],
                    "vad_status": status,
                }
            )
            if self.listening["stat"] == "el":
                b.vad.in_dialog = False
                self.listening = None
            elif self.listening["stat"] == "ss":
                if self.started_generation:
                    events.append(
                        {
                            "type": "controller_stop_generate",
                            "source": "external_policy",
                            "reason": "model_stat_ss",
                        }
                    )
                    events.append(
                        {
                            "type": "controller_stop_tts",
                            "source": "external_policy",
                            "reason": "model_stat_ss",
                        }
                    )
                self.generation = deepcopy(self.listening)
                self.listening = None
                b.vad.in_dialog = False
                self.speaking = True
                self.started_generation = True
                self.hidden, self.pending, self.previous_text = [], "", ""
                self.generation = b.pipeline.speech_dialogue(None, **self.generation)
                if self.generation["stat"] == "cs":
                    self.hidden.append(self.generation["hidden_state"])
                    self.previous_text = self.generation.get("text", "")
                    self.pending = self.previous_text
        audio = []
        # Bounded work per input tick. Never consume a future user input.
        for _ in range(16):
            if not self.speaking:
                break
            self.generation.pop("text", None)
            self.generation.pop("hidden_state", None)
            self.generation = b.pipeline.speech_dialogue(None, **self.generation)
            self.token_count += 1
            stat = self.generation["stat"]
            events.append({"type": "native_generate_stat", "stat": stat})
            if stat != "cs":
                self.speaking = False
                break
            text = self.generation["text"]
            delta = text[len(self.previous_text) :]
            self.previous_text = text
            self.pending += delta
            self.hidden.append(self.generation["hidden_state"])
            if len(self.hidden) >= 4 and (
                delta.endswith((".", ",", "?", "!", "\n")) or len(self.hidden) >= 45
            ):
                chunks, self.generated = b._decode_tts(
                    self.hidden, self.pending, self.generated
                )
                audio.extend(chunks)
                events.append(
                    {"type": "native_tts", "text": self.pending, "chunks": len(chunks)}
                )
                self.hidden, self.pending = [], ""
                break
            if self.token_count >= 1000:
                raise RuntimeError("token safety limit reached")
        if not self.speaking and self.pending and self.hidden:
            chunks, self.generated = b._decode_tts(
                self.hidden, self.pending, self.generated
            )
            audio.extend(chunks)
            self.hidden, self.pending = [], ""
        waveform = b.torch.cat(audio).numpy() if audio else b.np.zeros(0)
        raw = (b.np.clip(waveform, -1, 1) * 32767).astype("<i2").tobytes()
        return raw, {"events": events, "speaking": self.speaking, "vad_status": status}

    def close(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--after-speech-ms", type=int, default=0)
    parser.add_argument("--min-post-user-silence-ms", type=int, default=0)
    args = parser.parse_args()
    directory = args.root / args.model / args.scenario
    directory.mkdir(parents=True, exist_ok=False)
    result = {"model": args.model, "scenario": args.scenario, "status": "failed"}
    runtime = None
    import torch

    torch.set_grad_enabled(False)
    input_track, raw_output, played = bytearray(), bytearray(), bytearray()
    queue = bytearray()
    try:
        config = json.loads((args.root / "configs" / f"{args.model}.json").read_text())
        cls = (
            MoshiRuntime
            if args.model == "moshi"
            else FreezeRuntime
            if args.model == "freeze_omni"
            else WorkerRuntime
        )
        runtime = cls(args.model, config)
        result["provenance"] = runtime.provenance
        result["config"] = config
        size = round(runtime.input_rate * runtime.width_ms / 1000) * 2
        output_size = round(runtime.rate * runtime.width_ms / 1000) * 2

        def chunks(name):
            pcm = read_pcm(args.root / "stimuli" / f"{name}.wav", runtime.input_rate)
            return deque(
                pcm[i : i + size].ljust(size, b"\0") for i in range(0, len(pcm), size)
            )

        pending = chunks("greeting")
        greeting_end_ms = len(pending) * runtime.width_ms
        # Moshi's reference inference primes with three seconds of silence.
        if args.model == "moshi":
            pending.extendleft(reversed([bytes(size)] * round(3000 / runtime.width_ms)))
        interjection = None
        trigger = None
        after = 0
        natural_end = False
        compound = args.scenario.startswith("normal_then_")
        phase = "baseline" if compound else "target"
        baseline_ended = False
        followup_tick = None
        counts = Counter()
        last_frame = b""
        first_voice_ms = None
        with (directory / "native-events.jsonl").open("w") as log:
            for tick in range(
                round((120000 if compound else 70000) / runtime.width_ms)
            ):
                if phase == "gap" and tick >= followup_tick:
                    phase = "target"
                    pending = chunks("greeting")
                    greeting_end_ms = (tick + len(pending)) * runtime.width_ms
                    first_voice_ms = None
                    last_frame = b""
                if (
                    phase == "target"
                    and first_voice_ms is not None
                    and tick * runtime.width_ms >= first_voice_ms + args.after_speech_ms
                    and tick * runtime.width_ms
                    >= greeting_end_ms + args.min_post_user_silence_ms
                    and args.scenario != "normal"
                    and trigger is None
                    and not pending
                    and last_frame
                    and audioop.rms(last_frame, 2) > 200
                ):
                    interjection = (
                        "backchannel"
                        if "backchannel" in args.scenario
                        else "interruption"
                    )
                    pending = chunks(interjection)
                    trigger = {
                        "tick": tick,
                        "sim_ms": tick * runtime.width_ms,
                        "model_frame_rms": audioop.rms(last_frame, 2),
                        "queued_bytes": len(queue),
                        "user_chunks": len(pending),
                    }
                pcm = pending.popleft() if pending else bytes(size)
                forced = (
                    args.scenario in {"forced_interruption", "forced_interruption_hold"}
                    and trigger is not None
                    and after == 0
                )
                started = time.monotonic()
                hold = (
                    args.scenario == "forced_interruption_hold"
                    and trigger is not None
                    and 0 < after <= trigger["user_chunks"]
                )
                raw, event = (
                    runtime.step(pcm, force_listen=True)
                    if hold
                    else runtime.step(pcm, interrupt=forced)
                )
                wall = time.monotonic() - started
                cancelled = 0
                for item in event.get("events", []):
                    counts[item.get("type", item.get("event", "unknown"))] += 1
                    if (
                        item.get("type") == "controller_stop_tts"
                        or item.get("event") == "native_listening_transition"
                    ):
                        cancelled += len(queue)
                        queue.clear()
                    if item.get("event") == "playback_turn_ended" or (
                        item.get("type") == "native_generate_stat"
                        and item.get("stat") == "sl"
                    ):
                        natural_end = True
                    elif item.get("event") == "native_assistant_start":
                        natural_end = False
                # Native MiniCPM listening cancels old pending audio only while
                # the previous generation has not declared natural completion.
                if args.model == "minicpm":
                    if forced or (
                        event.get("is_listen")
                        and not natural_end
                        and not event.get("end_of_turn")
                    ):
                        cancelled = len(queue)
                        queue.clear()
                    if event.get("end_of_turn"):
                        natural_end = True
                    elif not event.get("is_listen", True):
                        natural_end = False
                queue.extend(raw)
                frame = bytes(queue[:output_size])
                del queue[:output_size]
                input_track.extend(pcm)
                raw_output.extend(raw)
                if first_voice_ms is None and frame and audioop.rms(frame, 2) > 200:
                    first_voice_ms = tick * runtime.width_ms
                last_frame = frame
                played.extend(frame.ljust(output_size, b"\0"))
                if compound and phase == "baseline" and natural_end:
                    baseline_ended = True
                    phase = "gap"
                    followup_tick = tick + max(2, round(1000 / runtime.width_ms))
                row = {
                    "phase": phase,
                    "tick": tick,
                    "input_sim_ms": tick * runtime.width_ms,
                    "input_pcm_samples": len(pcm) // 2,
                    "native_output_samples": len(raw) // 2,
                    "played_samples": len(frame) // 2,
                    "output_rms": audioop.rms(raw, 2) if raw else 0,
                    "native": event,
                    "external_interrupt_requested": forced,
                    "external_listen_hold_requested": hold,
                    "cancelled_pending_bytes": cancelled,
                    "wall_seconds": wall,
                }
                log.write(json.dumps(row) + "\n")
                log.flush()
                if trigger is not None:
                    after += 1
                    if after * runtime.width_ms >= 15000 and not pending:
                        break
                if (
                    phase == "target"
                    and first_voice_ms is not None
                    and tick * runtime.width_ms >= first_voice_ms + args.after_speech_ms
                    and args.scenario == "normal"
                    and natural_end
                    and not queue
                    and tick * runtime.width_ms > 4000
                ):
                    break
        result.update(
            status="observed",
            trigger=trigger,
            event_counts=dict(counts),
            naturally_ended=natural_end,
            baseline_ended=baseline_ended,
            input_seconds=len(input_track) / runtime.input_rate / 2,
            raw_output_seconds=len(raw_output) / runtime.rate / 2,
            played_seconds=len(played) / runtime.rate / 2,
        )
        if not played or audioop.rms(played, 2) <= 100:
            result.update(status="failed", error="No audible model output")
        elif compound and (not baseline_ended or counts["native_assistant_start"] < 2):
            result.update(
                status="failed",
                error="No second response after natural playback completion",
            )
        elif args.scenario != "normal" and trigger is None:
            result.update(status="failed", error="No speaking-time injection trigger")
    except Exception as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
        traceback.print_exc()
    finally:
        if runtime is not None:
            write_wav(directory / "input.wav", input_track, runtime.input_rate)
            write_wav(directory / "output-native.wav", raw_output, runtime.rate)
            write_wav(directory / "output-played.wav", played, runtime.rate)
            runtime.close()
        (directory / "probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("config", "provenance")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
