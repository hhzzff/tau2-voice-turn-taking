"""Dedicated OpenAI ASR uses its configured model and rejects silent input locally."""

import io
import wave
from types import SimpleNamespace

import pytest

from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat
from tau2.data_model.voice import TranscriptionConfig
from tau2.voice.transcription.transcribe import transcribe_audio


def audio(samples=b"\x10\x00" * 2400):
    return AudioData(
        data=samples,
        format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=24000),
    )


@pytest.fixture(autouse=True)
def forbid_realtime_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("File ASR must not open a realtime conversation")

    monkeypatch.setattr(
        "tau2.voice.transcription.transcribe.websockets.connect", forbidden
    )


@pytest.mark.parametrize(
    "model", ["whisper-1", "gpt-4o-mini-transcribe", "gpt-4o-transcribe"]
)
def test_configured_asr_uses_file_endpoint_and_original_audio(monkeypatch, model):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-key")
    calls = []

    def post(url, **kwargs):
        calls.append(url)
        assert url == "https://api.openai.com/v1/audio/transcriptions"
        assert kwargs["headers"] == {"Authorization": "Bearer fixture-key"}
        assert kwargs["data"] == {"model": model, "language": "en"}
        assert kwargs["timeout"] > 0
        with wave.open(io.BytesIO(kwargs["files"]["file"][1])) as wav:
            assert wav.getframerate() == 24000
            assert wav.getnchannels() == 1
            assert wav.readframes(wav.getnframes()) == audio().data
        return SimpleNamespace(status_code=200, json=lambda: {"text": "No sesame."})

    monkeypatch.setattr("tau2.voice.transcription.transcribe.requests.post", post)
    result = transcribe_audio(
        audio(), TranscriptionConfig(model=model, language="en-US")
    )
    assert result.transcript == "No sesame."
    assert result.error is None
    assert len(calls) == 1


@pytest.mark.parametrize("samples", [b"", b"\x00\x00" * 2400], ids=["empty", "silent"])
def test_digital_silence_returns_empty_without_credentials_or_service(
    monkeypatch, samples
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Silent input must not reach an ASR service")

    monkeypatch.setattr("tau2.voice.transcription.transcribe.requests.post", forbidden)
    result = transcribe_audio(
        audio(samples), TranscriptionConfig(model="gpt-4o-mini-transcribe")
    )
    assert result.transcript == ""
    assert result.error is None


def test_service_failure_does_not_expose_response_body(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-secret")
    monkeypatch.setattr(
        "tau2.voice.transcription.transcribe.requests.post",
        lambda *args, **kwargs: SimpleNamespace(status_code=401, text="fixture-secret"),
    )
    result = transcribe_audio(
        audio(), TranscriptionConfig(model="gpt-4o-mini-transcribe")
    )
    assert result.error == "OpenAI transcription HTTP 401"
    assert result.transcript == ""
