"""Native connect rejects unsupported identity before loading GPU packages."""

import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "config",
    [
        {"checkpoint": "/models/other-version", "assets": {}},
        {
            "checkpoint": "/mnt/bn/audio-visual-llm-data6/base_models/MiniCPM-o-4_5",
            "assets": {"pt_path": "/models/finetuned.pt"},
        },
    ],
)
def test_native_connect_rejects_identity_overrides(config):
    path = (
        Path(__file__).resolve().parents[4]
        / "scripts/open_model_workers/minicpm_worker.py"
    )
    spec = importlib.util.spec_from_file_location("native_minicpm_identity_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="MiniCPM-o-4.5"):
        module.MiniCPMWorker().connect(config, "hello")
