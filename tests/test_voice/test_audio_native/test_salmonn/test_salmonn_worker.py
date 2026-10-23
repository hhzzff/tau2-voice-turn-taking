"""SALMONN native worker contract; no checkpoint or GPU required."""

import importlib.util
import sys
from pathlib import Path

import pytest

WORKERS = Path(__file__).resolve().parents[4] / "scripts" / "open_model_workers"


def load_worker():
    sys.path.insert(0, str(WORKERS))
    try:
        spec = importlib.util.spec_from_file_location(
            "salmonn_worker", WORKERS / "salmonn_worker.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.SalmonnWorker()
    finally:
        sys.path.remove(str(WORKERS))


def test_after_sft_checkpoint_rejected_before_runtime_load():
    worker = load_worker()
    with pytest.raises(ValueError, match="before-SFT"):
        worker.connect({"checkpoint": "/models/cla_bc/checkpoint-8000"}, "hello")


def test_disconnected_worker_cannot_consume_audio():
    worker = load_worker()
    with pytest.raises(RuntimeError, match="connected"):
        worker.step(bytes(2560))
    worker.disconnect()
    worker.disconnect()


def test_tick_mismatch_rejected_before_loading_checkpoint():
    worker = load_worker()
    with pytest.raises(ValueError, match="80"):
        worker.connect(
            {
                "checkpoint": "/models/cu12_stage2_v2_tfv2_short_sm1_sp_longs1/checkpoint-44000",
                "settings": {"chunk_ms": 1000},
            },
            "hello",
        )


def test_native_cpu_stream_contract():
    """Exercise real streaming code when a SALMONN interpreter is configured."""
    import os
    import subprocess

    python = os.environ.get("SALMONN_NATIVE_PYTHON")
    runtime = os.environ.get("SALMONN_RUNTIME_CONFIG")
    if not python or not runtime:
        pytest.skip(
            "set SALMONN_NATIVE_PYTHON and SALMONN_RUNTIME_CONFIG for CPU contract"
        )
    result = subprocess.run(
        [python, str(Path(__file__).with_name("native_cpu_contract.py")), runtime],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "32 causal ticks" in result.stdout
