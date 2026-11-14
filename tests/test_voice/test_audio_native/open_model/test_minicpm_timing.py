"""Timing probe reports model work separately from output audio duration."""

import importlib.util
from pathlib import Path


def test_native_deadline_includes_finalize_but_audio_ready_does_not():
    path = (
        Path(__file__).resolve().parents[4]
        / "scripts/control_signal_probes/minicpm_timing.py"
    )
    spec = importlib.util.spec_from_file_location("minicpm_timing", path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    summary = probe.summarize(
        [
            {
                "prefill_ms": 100,
                "generate_ms": 850,
                "finalize_ms": 100,
                "audio_ms": 1160,
                "audible": True,
            },
            {
                "prefill_ms": 100,
                "generate_ms": 200,
                "finalize_ms": 50,
                "audio_ms": 0,
                "audible": False,
            },
        ]
    )
    assert summary["ticks"] == 2
    assert summary["audio_ready_ms"]["max"] == 950
    assert summary["full_step_ms"]["max"] == 1050
    assert summary["audio_ready_over_1000ms"] == 0
    assert summary["full_step_over_1000ms"] == 1
    assert summary["audible_output_ms"]["max"] == 1160
    assert summary["speaking_ticks"] == 1
