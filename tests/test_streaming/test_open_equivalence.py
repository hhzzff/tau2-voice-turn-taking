"""Fixed input must actually cross the production orchestrator and adapter."""

import base64
import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "model,size,samples",
    [
        ("moshi", 640, 1280),
        ("salmonn", 640, 1280),
        ("minicpm", 8000, 16000),
        ("freeze_omni", 1280, 2560),
    ],
)
def test_replay_sends_every_interval_once_through_real_step(model, size, samples):
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/control_signal_probes/open_equivalence.py"
    )
    spec = importlib.util.spec_from_file_location("open_equivalence", path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    inputs = []

    class Runtime:
        def request(self, op, **payload):
            if op == "connect":
                return {"sample_rate": 16000}
            if op == "step":
                inputs.append(base64.b64decode(payload["audio_pcm16_b64"]))
                return {
                    "sample_rate": 16000,
                    "audio_pcm16_b64": base64.b64encode(b"\x00\x10" * samples).decode(),
                    "events": [],
                    "text": "hello",
                }
            return {}

        def close(self):
            pass

    raw = b"\x80" * size + b"\x90" * size
    scheduler, agent = probe.build_scheduler(model, raw, {}, Runtime())
    try:
        scheduler.initialize()
        assert len(inputs) == 0
        scheduler.step()
        scheduler.step()
        assert len(inputs) == 2
        assert all(len(pcm) == samples * 2 for pcm in inputs)
        # Independent mu-law reference values, excluding resampler boundary.
        assert int.from_bytes(inputs[0][4:6], "little", signed=True) == 32124
        assert int.from_bytes(inputs[1][4:6], "little", signed=True) == 15996
        assert len(scheduler.get_ticks()) == 3  # Includes initial silence tick.
        assert (
            len(base64.b64decode(scheduler.get_ticks()[1].agent_chunk.audio_content))
            == size
        )
    finally:
        agent.stop()
