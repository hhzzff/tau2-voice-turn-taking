"""Causal assistant perception and revision of partial ASR hypotheses."""

import json
import wave

import numpy as np
import pytest

from tau2.data_model.audio import AudioEncoding, AudioFormat, audio_bytes_to_string
from tau2.data_model.message import AssistantMessage
from tau2.data_model.voice import TranscriptionResult
from tau2.user.duplex_perception import PlayedAudioPerception


def chunk(value=1000, *, text="proportional text", uid="reply"):
    return AssistantMessage(
        role="assistant",
        content=text,
        utterance_ids=[uid] if uid else None,
        contains_speech=True,
        audio_content=audio_bytes_to_string(
            np.full(3200, value, dtype="<i2").tobytes()
        ),
        audio_format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=16000),
    )


def test_recognition_uses_only_played_audio_not_proportional_or_future_text(
    monkeypatch,
):
    perception = PlayedAudioPerception("Chinese")
    first = chunk(text="鸡蛋可以用（错位文本）")
    first.audio_script_gold = "未来还没播放的文字"
    received = []

    def recognize(audio, config):
        received.append(audio.data)
        assert config.language == "zh"
        return TranscriptionResult(transcript="鸡蛋放两颗。")

    monkeypatch.setattr("tau2.user.duplex_perception.transcribe_audio", recognize)
    perception.observe(first, 55.0)
    old_history = [
        {"role": "user", "text": "鸡蛋放几颗？", "seconds": 40.4},
        {"role": "assistant", "text": "鸡蛋可以用", "seconds": 45.8},
    ]
    result = perception.history(old_history, 55.2)
    assert received == [first.get_audio_bytes()]
    assert result[-1]["text"] == "鸡蛋放两颗。"
    assert result[-1]["audio_end_seconds"] <= 55.2
    assert result[-1]["text_source"] == "asr_of_played_audio"
    assert perception.snapshots[-1]["audio_bytes"] == len(first.get_audio_bytes())
    assert old_history[-1]["text"] == "鸡蛋可以用"


def test_updated_hypothesis_replaces_old_text_and_unchanged_audio_is_cached(
    monkeypatch,
):
    perception = PlayedAudioPerception("English")
    responses = iter(["Use", "Use two eggs."])
    calls = []

    def recognize(audio, config):
        calls.append(audio.data)
        return TranscriptionResult(transcript=next(responses))

    monkeypatch.setattr("tau2.user.duplex_perception.transcribe_audio", recognize)
    perception.observe(chunk(), 0)
    assert perception.history([], 0.2)[0]["text"] == "Use"
    assert perception.history([], 0.2)[0]["text"] == "Use"
    # Quiet chunks still carry sound/text within an utterance and must survive.
    perception.observe(chunk(10), 0.2)
    assert perception.history([], 0.4)[0]["text"] == "Use two eggs."
    assert len(calls) == 2
    assert len(calls[-1]) == 2 * len(calls[0])
    perception.observe(chunk(0, text="", uid=None), 0.4)
    assert perception.history([], 0.6)[0]["text"] == "Use two eggs."
    assert len(calls) == 2


def test_failed_recognition_does_not_fall_back_to_estimated_words(monkeypatch):
    perception = PlayedAudioPerception("Chinese")
    perception.observe(chunk(text="untrusted estimated text"), 0)
    monkeypatch.setattr(
        "tau2.user.duplex_perception.transcribe_audio",
        lambda *args: TranscriptionResult(transcript="", error="ASR unavailable"),
    )
    with pytest.raises(RuntimeError, match="transcription failed"):
        perception.history([], 0.2)
    assert perception.snapshots[-1]["error"] == "ASR unavailable"
    assert perception.current["recognized_bytes"] == 0


def test_user_turn_boundary_and_prefix_evidence_can_be_replayed(monkeypatch, tmp_path):
    perception = PlayedAudioPerception("Chinese")
    monkeypatch.setattr(
        "tau2.user.duplex_perception.transcribe_audio",
        lambda *args: TranscriptionResult(transcript="两颗鸡蛋"),
    )
    perception.observe(chunk(), 0)
    perception.history([], 0.2)
    perception.observe(chunk(), 0.2)
    perception.history([], 0.4)
    perception.start_user_turn()
    perception.observe(chunk(), 0.4)
    history = perception.history(
        [{"role": "user", "seconds": 0.4, "text": "然后呢？"}], 0.6
    )
    assert [entry["role"] for entry in history] == ["assistant", "user", "assistant"]
    assert [entry["text"] for entry in history] == ["两颗鸡蛋", "然后呢？", "两颗鸡蛋"]
    perception.save(tmp_path)
    perception.save(tmp_path)  # stop() and runner cleanup both persist evidence.
    evidence = json.loads((tmp_path / "perception.json").read_text(encoding="utf-8"))
    first, second, _ = evidence["snapshots"]
    assert first["audio_bytes"] * 2 == second["audio_bytes"]
    with wave.open(str(tmp_path / first["audio_file"]), "rb") as wav:
        assert len(wav.readframes(wav.getnframes())) == second["audio_bytes"]
    perception.observe(chunk(), 0.6)
    perception.history([], 0.8)
    perception.save(tmp_path)
    with wave.open(str(tmp_path / "heard_audio/assistant_001.wav"), "rb") as wav:
        assert len(wav.readframes(wav.getnframes())) == 2 * len(
            chunk().get_audio_bytes()
        )
