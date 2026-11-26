"""Core CLI commands must not import optional local speech dependencies."""

import json
import subprocess
import sys
from argparse import Namespace

import pytest
from pydantic import ValidationError

from experiments.tau_voice.local_user_cli import load_config


@pytest.mark.parametrize("arguments", [["--help"], ["list"], ["local", "--help"]])
def test_duplex_cli_without_local_audio_packages(arguments):
    script = """
import importlib.abc
import runpy
import sys

class NoLocalAudio(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'soundfile', 'soxr', 'sherpa_onnx'}:
            raise ModuleNotFoundError(fullname, name=fullname)

sys.meta_path.insert(0, NoLocalAudio())
sys.argv = ['duplex_bench', *sys.argv[1:]]
runpy.run_module('experiments.tau_voice.duplex_bench', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, *arguments],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout


def test_core_registry_is_independent_and_experiment_registration_is_explicit():
    script = """
import importlib.abc
import sys

class NoExperiments(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'experiments' or fullname.startswith('experiments.'):
            raise AssertionError('core registry imported an experiment')

blocker = NoExperiments()
sys.meta_path.insert(0, blocker)
from tau2.registry import registry
assert 'local_streaming_user' not in registry.get_users()
sys.meta_path.remove(blocker)
from experiments.tau_voice.local_user.participant import register_local_user
from tau2.user.user_simulator_base import FullDuplexUser
first = register_local_user()
assert issubclass(first, FullDuplexUser)
assert register_local_user() is first
assert registry.get_user_constructor('local_streaming_user') is first
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


def test_local_config_supports_external_services_and_explicit_asr_override(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "asr": {"model_path": "/models/original"},
                "llm": {"base_url": "http://llm.internal:8000/v1"},
                "tts": {"url": "http://tts.internal:8001/synthesize"},
            }
        )
    )
    cfg = load_config(Namespace(config=path, asr_model_path=tmp_path / "weights"))
    assert cfg.asr.model_path == tmp_path / "weights"
    assert cfg.llm.base_url == "http://llm.internal:8000/v1"
    assert cfg.tts.url == "http://tts.internal:8001/synthesize"
    assert (
        str(load_config(Namespace(config=path, asr_model_path=None)).asr.model_path)
        == "/models/original"
    )


def test_local_config_rejects_unknown_fields_and_missing_asr(tmp_path):
    with pytest.raises(ValidationError):
        load_config(Namespace(config=None, asr_model_path=None))
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"asr": {"model_path": "/models"}, "llmm": {}}))
    with pytest.raises(ValidationError, match="Extra inputs"):
        load_config(Namespace(config=path, asr_model_path=None))
