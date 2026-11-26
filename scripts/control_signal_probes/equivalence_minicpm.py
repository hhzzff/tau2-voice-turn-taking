"""Fresh-process native versus 100ms producer replay on the same MiniCPM GPU."""

import argparse
import audioop
import base64
import hashlib
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from equivalence import PROMPT, finish, read_wav


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["direct", "scheduled"], required=True)
    parser.add_argument("--run", type=int, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    sys.path.insert(0, config["assets"]["package_root"])
    import numpy as np
    import torch
    from core.processors import UnifiedProcessor
    from core.schemas import DuplexConfig

    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    def seed():
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    seed()
    settings = config["settings"]
    options = DuplexConfig(
        force_listen_count=0,
        chunk_ms=1000,
        sample_rate=16000,
        generate_audio=True,
        decode_mode=settings.get("decode_mode", "sampling"),
        temperature=settings.get("temperature", 0.7),
        tts_temperature=settings.get("tts_temperature", 0.8),
    )
    processor = UnifiedProcessor(
        model_path=config["checkpoint"], pt_path=None, duplex_config=options
    )
    duplex = processor.set_duplex_mode()
    duplex.config = options
    duplex.prepare(
        system_prompt_text=PROMPT, ref_audio_path=config["assets"]["reference_audio"]
    )
    # Generation RNG starts identically after any lazy preparation work.
    seed()
    torch.cuda.synchronize()
    pcm = read_wav(args.root / "input.wav", 16000)
    input_meta = json.loads((args.root / "input.json").read_text())
    directory = args.root / "minicpm" / f"{args.run}-{args.mode}"
    directory.mkdir(parents=True, exist_ok=False)
    rows, events, raw = [], [], bytearray()
    state = {
        "previous_user": False,
        "previous_model": False,
        "latch": False,
        "response": 0,
        "ended": True,
    }
    origin = time.monotonic()

    def now():
        return (time.monotonic() - origin) * 1000

    def step(block, tick, submitted_ms):
        started = now()
        user = any(
            tick * 1000 < u["end_ms"] and (tick + 1) * 1000 > u["start_ms"]
            for u in input_meta["utterances"]
        )
        interrupt = user and not state["previous_user"] and state["previous_model"]
        if interrupt:
            duplex.set_break()
            state["latch"] = True
            events.append(
                {
                    "kind": "cancel",
                    "at_ms": now(),
                    "response_id": str(state["response"]),
                    "source": "external_reference_controller",
                }
            )
            state["ended"] = True
        force = state["latch"]
        if force and not user:
            state["latch"] = False
        audio = np.frombuffer(block, dtype="<i2").astype(np.float32) / 32768
        duplex.prefill(audio_waveform=audio)
        result = duplex.generate(force_listen=force)
        torch.cuda.synchronize()
        ready = now()
        waveform = (
            np.frombuffer(base64.b64decode(result.audio_data), dtype="<f4")
            if result.audio_data
            else np.empty(0)
        )
        if not np.isfinite(waveform).all():
            raise ValueError("Nonfinite native output")
        rendered = (np.clip(waveform, -1, 1) * 32767).astype("<i2").tobytes()
        audible = bool(rendered and audioop.rms(rendered, 2) > 100)
        if not result.is_listen:
            if state["ended"]:
                state["response"] += 1
                state["ended"] = False
            events.append(
                {
                    "kind": "audio",
                    "at_ms": ready,
                    "response_id": str(state["response"]),
                    "raw_offset_bytes": len(raw),
                    "audio_bytes": len(rendered),
                }
            )
        raw.extend(rendered)
        if result.end_of_turn:
            state["ended"] = True
        duplex.finalize()
        torch.cuda.synchronize()
        row = {
            "tick": tick,
            "input_end_ms": (tick + 1) * 1000,
            "submitted_ms": submitted_ms,
            "started_ms": started,
            "audio_ready_ms": ready,
            "finished_ms": now(),
            "input_sha256": hashlib.sha256(block).hexdigest(),
            "audio_sha256": hashlib.sha256(rendered).hexdigest(),
            "audio_samples": len(waveform),
            "text": result.text or "",
            "is_listen": bool(result.is_listen),
            "end_of_turn": bool(result.end_of_turn),
            "interrupt": interrupt,
            "force_listen": force,
        }
        rows.append(row)
        with (directory / "native.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        state["previous_user"], state["previous_model"] = user, audible

    try:
        if args.mode == "direct":
            for tick, offset in enumerate(range(0, len(pcm), 32000)):
                time.sleep(max(0, origin + tick + 1 - time.monotonic()))
                step(pcm[offset : offset + 32000], tick, now())
        else:
            # 100ms input delivery remains active while the single native
            # inference worker consumes complete 1000ms blocks serially.
            with ThreadPoolExecutor(max_workers=1) as executor:
                pending, futures, tick = bytearray(), [], 0
                for offset in range(0, len(pcm), 3200):
                    end = min(offset + 3200, len(pcm))
                    time.sleep(max(0, origin + end / 32000 - time.monotonic()))
                    pending.extend(pcm[offset:end])
                    if len(pending) == 32000:
                        futures.append(
                            executor.submit(step, bytes(pending), tick, now())
                        )
                        pending.clear()
                        tick += 1
                for future in futures:
                    future.result()
        result = finish(
            directory,
            events,
            bytes(raw),
            rate=24000,
            horizon_ms=42000,
            quantum_ms=0 if args.mode == "direct" else 100,
            metadata={
                "model": "MiniCPM-o-4.5",
                "mode": args.mode,
                "run": args.run,
                "status": "observed",
                "input_sha256": hashlib.sha256(pcm).hexdigest(),
                "text": "".join(r["text"] for r in rows),
                "interruption_end_ms": input_meta["utterances"][1]["end_ms"],
                "prompt": PROMPT,
                "seed": 42,
                "process_isolation": "fresh process for each run",
                "gpu": torch.cuda.get_device_name(),
                "control_policy": "same pi-bench annotated set_break/force_listen on both paths",
                "player_start_buffer_ms": 0,
                "finalize_delivery": "audio exposed before finalize on both paths",
            },
        )
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "model",
                        "mode",
                        "status",
                        "first_voice_ms",
                        "post_interruption_voice_ms",
                    )
                }
            ),
            flush=True,
        )
    finally:
        duplex.stop()
        duplex.cleanup()


if __name__ == "__main__":
    main()
