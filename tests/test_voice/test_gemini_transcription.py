"""Gemini transcription must receive audio alone and preserve incomplete speech."""

import base64
import io
import wave
from types import SimpleNamespace

from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat
from tau2.data_model.voice import TranscriptionConfig
from tau2.voice.transcription.transcribe import transcribe_audio


def test_gemini_receives_wav_audio_and_does_not_include_thoughts(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_LIVE_API_KEY", "fixture-key")
    captured = []

    def post(url, **kwargs):
        captured.append(kwargs)
        assert "fixture-key" not in url
        assert kwargs["headers"] == {"x-goog-api-key": "fixture-key"}
        parts = kwargs["json"]["contents"][0]["parts"]
        assert "Do not answer" in parts[0]["text"]
        with wave.open(
            io.BytesIO(base64.b64decode(parts[1]["inlineData"]["data"]))
        ) as wav:
            assert wav.getframerate() == 24000
            assert wav.getnchannels() == 1
            assert wav.getnframes() == 2400
        return SimpleNamespace(
            ok=True,
            json=lambda: {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "thought": True,
                                    "text": "Do not expose internal notes",
                                },
                                {"text": "My friend is allergic to..."},
                            ]
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr("tau2.voice.transcription.transcribe.requests.post", post)
    result = transcribe_audio(
        AudioData(
            data=b"\x01\x00" * 2400,
            format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=24000),
        ),
        TranscriptionConfig(model="gemini-3.8-flash", language="en"),
    )
    assert result.error is None
    assert result.transcript == "My friend is allergic to..."
    assert len(captured) == 1


def test_gemini_failure_does_not_copy_error_body_or_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fixture-secret")
    monkeypatch.setattr(
        "tau2.voice.transcription.transcribe.requests.post",
        lambda *a, **kw: SimpleNamespace(ok=False, status_code=401),
    )
    result = transcribe_audio(
        AudioData(
            data=b"\x01\x00" * 2400,
            format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=24000),
        ),
        TranscriptionConfig(model="gemini-3.8-flash"),
    )
    assert result.error == "Gemini transcription HTTP 401"
    assert not result.transcript
