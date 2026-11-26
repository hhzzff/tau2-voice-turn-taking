"""Core CLI commands must not import optional local speech dependencies."""

import subprocess
import sys

import pytest


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
