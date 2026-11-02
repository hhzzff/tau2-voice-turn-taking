"""Measure preloaded waveform kernels without model or network requests.

Example: python -m experiments.tau_voice.clarification_benchmark --noise-path
/path/noise.wav --source-wav /path/tts.wav --output /tmp/kernel-benchmark.json
"""

import argparse
import hashlib
import json
import os
import platform
import time
import wave
from pathlib import Path

import numpy as np

from tau2.user.acoustic_clarification import (
    AcousticClarificationConfig,
    hard_dns_noise,
    hard_dropchunk,
    prepare_noise,
)


def benchmark(args):
    """Time warm calls, including full-waveform copy and local RNG setup."""
    if args.output.exists():
        raise FileExistsError("Choose a new benchmark output file")
    if args.iterations < 1 or args.warmup < 0:
        raise ValueError("Positive iterations and nonnegative warmup required")
    with wave.open(str(args.source_wav), "rb") as stream:
        if stream.getsampwidth() != 2 or stream.getnchannels() != 1:
            raise ValueError("Expected mono PCM16 source WAV")
        rate = stream.getframerate()
        source = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
    if not source.size:
        raise ValueError("Empty source WAV")
    started = time.perf_counter()
    noise = prepare_noise(
        AcousticClarificationConfig(mode="dns_noise", noise_path=str(args.noise_path)),
        rate,
        args.seed,
    )
    cold = time.perf_counter() - started
    if noise.failure_reason:
        raise ValueError(noise.failure_reason)
    cpu_model = (
        next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            platform.processor(),
        )
        if Path("/proc/cpuinfo").exists()
        else platform.processor()
    )
    result = {
        "method": "perf_counter_ns; warm calls include full PCM copy, RNG setup, target operation and returned metadata; percentiles use numpy linear interpolation; sequential CPU calls, no API calls",
        "device": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": cpu_model,
            "available_cpus": len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count(),
            "cpu_count": os.cpu_count(),
            "numpy": np.__version__,
        },
        "sample_rate": rate,
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "source_wav": str(args.source_wav),
        "source_sha256": hashlib.sha256(args.source_wav.read_bytes()).hexdigest(),
        "noise": noise.model_dump(),
        "cold_noise_preparation_seconds": cold,
        "cases": [],
        "limitations": "No millisecond SLA is specified. This measures CPU local corruption, not alignment, TTS, wall-clock realtime or clarification behavior. Cold value is one observation after Python/module import; process imports are excluded.",
    }
    for seconds in (1, 10, 60):
        count = round(rate * seconds)
        samples = np.resize(source, count).copy()
        for target_seconds in (0.1, 0.4):
            start = round(rate * 0.25)
            end = start + round(rate * target_seconds)
            for mode in ("dropchunk", "dns_noise"):

                def run(index):
                    if mode == "dropchunk":
                        return hard_dropchunk(samples, (start, end), args.seed + index)
                    return hard_dns_noise(
                        samples, (start, end), args.seed + index, noise
                    )

                for index in range(args.warmup):
                    run(index)
                elapsed = []
                for index in range(args.iterations):
                    begin = time.perf_counter_ns()
                    run(index)
                    elapsed.append(time.perf_counter_ns() - begin)
                values = np.asarray(elapsed) / 1e6
                result["cases"].append(
                    {
                        "mode": mode,
                        "audio_seconds": seconds,
                        "audio_samples": count,
                        "target_seconds": target_seconds,
                        "target_samples": [start, end],
                        "count": len(values),
                        "p50_ms": float(np.percentile(values, 50)),
                        "p95_ms": float(np.percentile(values, 95)),
                        "max_ms": float(values.max()),
                        "durations_ns": elapsed,
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved {len(result['cases'])} kernel cases to {args.output}")


def main():
    """Explicit local benchmark entrypoint; never reads credentials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-wav", type=Path, required=True)
    parser.add_argument("--noise-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
