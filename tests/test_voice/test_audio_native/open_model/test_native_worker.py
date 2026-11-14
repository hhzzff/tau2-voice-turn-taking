"""Native worker contracts with only the external model SDK substituted."""

import base64
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize(
    "interrupt,force_listen", [(False, False), (True, False), (False, True)]
)
def test_minicpm_worker_preserves_native_decision_and_consumes_current_unit(
    tmp_path, monkeypatch, interrupt, force_listen
):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[4] / "scripts/open_model_workers")
    )
    from minicpm_worker import MiniCPMWorker

    consumed = []
    operations = []

    class Duplex:
        def prepare(self, **kwargs):
            operations.append(("prepare", kwargs))

        def set_break(self):
            operations.append(("set_break",))

        def prefill(self, *, audio_waveform):
            operations.append(("prefill",))
            consumed.append(audio_waveform.copy())

        def generate(self, *, force_listen):
            operations.append(("generate", force_listen))
            return SimpleNamespace(
                audio_data=base64.b64encode(
                    np.zeros(24000, dtype="<f4").tobytes()
                ).decode(),
                text="",
                is_listen=True,
                end_of_turn=False,
                current_time=1.0,
                cost_all_ms=15.0,
            )

        def finalize(self):
            operations.append(("finalize",))

        def stop(self):
            pass

        def cleanup(self):
            pass

    duplex = Duplex()
    processors = ModuleType("core.processors")
    processors.__file__ = str(tmp_path / "core/processors/__init__.py")
    processors.UnifiedProcessor = lambda **kwargs: SimpleNamespace(
        set_duplex_mode=lambda: duplex
    )
    schemas = ModuleType("core.schemas")
    schemas.__file__ = str(tmp_path / "core/schemas/__init__.py")
    schemas.DuplexConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(__import__("sys").modules, "core.processors", processors)
    monkeypatch.setitem(__import__("sys").modules, "core.schemas", schemas)
    worker = MiniCPMWorker()
    worker.connect(
        {
            "checkpoint": "/mnt/bn/audio-visual-llm-data6/base_models/MiniCPM-o-4_5",
            "assets": {"package_root": str(tmp_path), "reference_audio": "test.wav"},
        },
        "Answer in English and listen to corrections.",
    )
    try:
        result = worker.step(
            b"\x00\x20" * 16000, interrupt=interrupt, force_listen=force_listen
        )
        assert consumed[0].shape == (16000,)
        assert np.all(consumed[0] == 0.25)
        assert operations[1:] == (
            ([("set_break",)] if interrupt else [])
            + [("prefill",), ("generate", interrupt or force_listen), ("finalize",)]
        )
        assert result["control"]["source"] == (
            "external_policy" if interrupt or force_listen else "native_runtime"
        )
        assert result["control"]["interrupt_requested"] is interrupt
        assert result["control"]["force_listen"] is (interrupt or force_listen)
        assert result["is_listen"] is True
        assert result["end_of_turn"] is False
        assert result["current_time"] == 1.0
        assert result["cost_all_ms"] == 15.0
    finally:
        worker.disconnect()
