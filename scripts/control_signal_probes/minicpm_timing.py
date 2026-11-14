"""Measure native MiniCPM audio availability on an actual GPU, without tau.

Uses the original synchronous SDK path. No credentials or external calls.
Each run replays identical fixed audio, seeded before prepare().
"""

import argparse
import audioop
import base64
import hashlib
import json
import math
import random
import sys
import time
import wave
from pathlib import Path


def distribution(values):
    """Describe measurements without averaging away deadline misses."""
    values = sorted(values)
    if not values:
        return {"count": 0, "min": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "min": values[0],
        "p50": values[math.ceil(len(values) * 0.5) - 1],
        "p95": values[math.ceil(len(values) * 0.95) - 1],
        "max": values[-1],
    }


def summarize(rows):
    """Compare completed GPU work to the native 1000 ms input deadline."""
    ready = [r["prefill_ms"] + r["generate_ms"] for r in rows]
    full = [v + r["finalize_ms"] for v, r in zip(ready, rows)]
    speech = [r for r in rows if r["audible"]]
    return {
        "ticks": len(rows),
        "speaking_ticks": len(speech),
        "audio_ready_ms": distribution(ready),
        "full_step_ms": distribution(full),
        "speaking_audio_ready_ms": distribution(
            [r["prefill_ms"] + r["generate_ms"] for r in speech]
        ),
        "speaking_full_step_ms": distribution(
            [r["prefill_ms"] + r["generate_ms"] + r["finalize_ms"] for r in speech]
        ),
        "audio_ready_over_1000ms": sum(v > 1000 for v in ready),
        "full_step_over_1000ms": sum(v > 1000 for v in full),
        "audible_output_ms": distribution([r["audio_ms"] for r in speech]),
    }


def read_audio(path):
    """Read generic synthetic mono audio at the model's input rate."""
    with wave.open(str(path), "rb") as f:
        if f.getnchannels() != 1 or f.getsampwidth() != 2:
            raise ValueError("Expected mono PCM16")
        pcm = f.readframes(f.getnframes())
        return audioop.ratecv(pcm, 2, 1, f.getframerate(), 16000, None)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stimuli", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=45)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
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
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    seed()
    settings = config["settings"]
    duplex_config = DuplexConfig(
        force_listen_count=0,
        chunk_ms=1000,
        sample_rate=16000,
        generate_audio=True,
        decode_mode=settings.get("decode_mode", "sampling"),
        temperature=settings.get("temperature", 0.7),
        tts_temperature=settings.get("tts_temperature", 0.8),
    )
    processor = UnifiedProcessor(
        model_path=config["checkpoint"],
        pt_path=config["assets"].get("pt_path"),
        duplex_config=duplex_config,
    )
    duplex = processor.set_duplex_mode()
    duplex.config = duplex_config
    source = bytearray(args.seconds * 32000)
    intervals = []
    for start, name in [(0, "greeting"), (20, "interruption"), (32, "backchannel")]:
        pcm = read_audio(args.stimuli / f"{name}.wav")
        offset = start * 32000
        if offset + len(pcm) > len(source):
            raise ValueError("Stimulus exceeds fixed input horizon")
        source[offset : offset + len(pcm)] = pcm
        intervals.append((start, start + len(pcm) / 32000, name))
    (args.root / "input.pcm16").write_bytes(source)
    report = {
        "model": "MiniCPM-o-4.5",
        "config": config,
        "seed": args.seed,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "input_sha256": hashlib.sha256(source).hexdigest(),
        "input_intervals_seconds": intervals,
        "output_interface": "synchronous full waveform; no sub-call callback",
        "clock": "wall time synchronized at each GPU stage; input media time fixed",
        "runs": {},
    }
    try:
        for run in ("native_1", "native_2", "reference_barge_in"):
            seed()
            duplex.prepare(
                system_prompt_text=(
                    "Speak English. When greeted explain a relaxed three-day Beijing "
                    "itinerary in detail. Listen to user corrections. "
                    "A brief yes is an acknowledgement."
                ),
                ref_audio_path=config["assets"]["reference_audio"],
            )
            torch.cuda.synchronize()
            rows, previous_user, previous_model, latch = [], False, False, 0
            output_offset = 0
            with (
                (args.root / f"{run}.jsonl").open("w") as trace,
                (args.root / f"{run}.pcm16").open("wb") as output,
            ):
                for tick in range(args.seconds):
                    pcm = source[tick * 32000 : (tick + 1) * 32000]
                    user = any(
                        tick < end and tick + 1 > start for start, end, _ in intervals
                    )
                    barge_in = (
                        run == "reference_barge_in"
                        and previous_model
                        and user
                        and not previous_user
                    )
                    if barge_in:
                        duplex.set_break()
                        latch = max(latch, 2)
                    force = latch > 0
                    if latch and not user:
                        latch = 0
                    waveform = (
                        np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
                    )
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    duplex.prefill(audio_waveform=waveform)
                    torch.cuda.synchronize()
                    prefilled = time.perf_counter()
                    result = duplex.generate(force_listen=force)
                    torch.cuda.synchronize()
                    generated = time.perf_counter()
                    duplex.finalize()
                    torch.cuda.synchronize()
                    finalized = time.perf_counter()
                    audio = (
                        np.frombuffer(base64.b64decode(result.audio_data), dtype="<f4")
                        if result.audio_data
                        else np.empty(0)
                    )
                    if not np.isfinite(audio).all():
                        raise ValueError("Nonfinite model audio")
                    rendered = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
                    audible = bool(rendered and audioop.rms(rendered, 2) > 100)
                    row = {
                        "tick": tick,
                        "input_start_ms": tick * 1000,
                        "input_sha256": hashlib.sha256(pcm).hexdigest(),
                        "prefill_ms": (prefilled - start) * 1000,
                        "generate_ms": (generated - prefilled) * 1000,
                        "finalize_ms": (finalized - generated) * 1000,
                        "sdk_cost_all_ms": result.cost_all_ms,
                        "audio_samples": len(audio),
                        "audio_ms": len(audio) / 24,
                        "audio_sha256": hashlib.sha256(rendered).hexdigest(),
                        "audio_offset_bytes": output_offset,
                        "audible": audible,
                        "is_listen": bool(result.is_listen),
                        "end_of_turn": bool(result.end_of_turn),
                        "text": result.text or "",
                        "external_break": barge_in,
                        "force_listen": force,
                    }
                    output.write(rendered)
                    output_offset += len(rendered)
                    trace.write(json.dumps(row) + "\n")
                    trace.flush()
                    rows.append(row)
                    previous_user, previous_model = user, audible
                    print(json.dumps({"run": run, **row}), flush=True)
            report["runs"][run] = summarize(rows)
            (args.root / "summary.json").write_text(json.dumps(report, indent=2))
    finally:
        duplex.stop()
        duplex.cleanup()


if __name__ == "__main__":
    main()
