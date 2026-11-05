"""Multiple essential information spans travel through the existing tick seam."""

import io
import json
import wave

import httpx
import numpy as np
import pytest

from experiments.tau_voice.duplex_bench import build_dynamic_user
from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat
from tau2.data_model.message import AssistantMessage
from tau2.data_model.simulation import AudioNativeConfig
from tau2.user.duplex_events import DuplexScenario


@pytest.mark.parametrize("mode", ["dropchunk", "dns_noise"])
@pytest.mark.parametrize(
    "annotation_case", ["normal", "repeated", "missing", "overlap", "silent_second"]
)
def test_three_keywords_share_full_alignment_and_preserve_other_tick_samples(
    monkeypatch, tmp_path, mode, annotation_case
):
    script = "去南门，十点见，带三份。"
    annotations = [
        {"text": "南门", "occurrence": 0},
        {"text": "十点", "occurrence": 0},
        {"text": "三份", "occurrence": 0},
    ]
    if annotation_case == "repeated":
        script = "去南门，南门旁，带三份。"
        annotations[1] = {"text": "南门", "occurrence": 1}
    elif annotation_case == "missing":
        annotations[1]["occurrence"] = 1
    elif annotation_case == "overlap":
        annotations[1] = annotations[0].copy()
    alignment_calls, prompts = [], []
    clean = np.full(32000, 3000, dtype="<i2")
    if annotation_case == "silent_second":
        clean[6400:9600] = 0

    def generate(**kwargs):
        prompts.append(kwargs["messages"][0].content)
        decision = (
            {
                "action": "speak",
                "text": script,
                "keywords": annotations,
                "reason": "说明地点、时间和数量。",
            }
            if len(prompts) == 1
            else {"action": "wait", "reason": "等待对方回答。"}
        )
        return AssistantMessage(role="assistant", content=json.dumps(decision))

    def align(url, **kwargs):
        alignment_calls.append(kwargs)
        with wave.open(io.BytesIO(kwargs["files"]["file"][1]), "rb") as stream:
            assert stream.getnframes() == 32000
        return httpx.Response(
            200,
            json={
                "characters": [
                    {"text": char, "start": i / 10, "end": (i + 1) / 10}
                    for i, char in enumerate(script)
                ]
            },
        )

    monkeypatch.setenv("ELEVENLABS_API", "offline-multi-keyword")
    monkeypatch.delenv("TERUMI_BT_AK", raising=False)
    monkeypatch.setattr("tau2.user.duplex_scenario_user.generate", generate)
    monkeypatch.setattr("httpx.post", align)
    monkeypatch.setattr(
        "tau2.user.duplex_scenario_user.synthesize_voice",
        lambda *a, **k: AudioData(
            data=clean.tobytes(),
            format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=16000),
        ),
    )
    noise_path = tmp_path / "noise.wav"
    with wave.open(str(noise_path), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(np.full(16000, 1800, dtype="<i2").tobytes())
    scenario = DuplexScenario(
        selected_events=[],
        acoustic_clarification={"mode": mode, "noise_path": str(noise_path)},
        assistant_observation="proportional",
        topic="集合安排",
        user_context="地点、时间和数量都需要对方准确记录。",
    )
    user = build_dynamic_user(
        scenario, tmp_path, AudioNativeConfig(), "nova", "onyx", "test"
    )
    state, chunks = user.get_init_state(), []
    for _ in range(10):
        chunk, state = user.get_next_chunk(
            state, AssistantMessage(role="assistant", content="", contains_speech=False)
        )
        chunks.append(chunk)
    delivered = np.frombuffer(
        b"".join(chunk.get_audio_bytes() for chunk in chunks), dtype="<i2"
    )
    assert len(prompts) == 1
    assert len(delivered) == len(clean)
    record = user.save_report()["acoustic_clarification"]["utterances"][0]
    if annotation_case in {"missing", "overlap", "silent_second"}:
        np.testing.assert_array_equal(delivered, clean)
        assert record["status"] == "failed"
        assert record["failure_reason"] == (
            "keyword_occurrence_not_found"
            if annotation_case == "missing"
            else "overlapping_keyword_annotations"
            if annotation_case == "overlap"
            else "no_effective_target_modification"
        )
        assert len(alignment_calls) == (1 if annotation_case == "silent_second" else 0)
        assert not any(
            item.get("modified_intervals_samples") for item in record.get("targets", [])
        )
        return
    assert len(alignment_calls) == 1
    allowed = np.zeros(len(clean), dtype=bool)
    for start, end in ((1600, 4800), (6400, 9600), (14400, 17600)):
        allowed[start:end] = True
        if mode == "dropchunk":
            assert np.count_nonzero(delivered[start:end] == 0) >= (end - start) * 0.65
        else:
            assert np.all(delivered[start:end] == 32767)
    np.testing.assert_array_equal(delivered[~allowed], clean[~allowed])
    assert [row["keyword"]["text"] for row in record["targets"]] == [
        item["text"] for item in annotations
    ]
    assert len(record["target_ranges_samples"]) == 3
    assert record["delivered_target_samples"] == 9600
    if mode == "dns_noise":
        for item in record["targets"]:
            assert item["preclip_noise_rms"] / item[
                "clean_target_rms"
            ] == pytest.approx(10, rel=1e-5)
            assert item["clipped_samples"] == 3200
    assert all(not chunk.content and not chunk.audio_script_gold for chunk in chunks)
    for _ in range(25):
        _, state = user.get_next_chunk(
            state, AssistantMessage(role="assistant", content="", contains_speech=False)
        )
    assert len(prompts) > 1
    assert "dropchunk" not in "\n".join(prompts[1:])
    assert "target_ranges_samples" not in "\n".join(prompts[1:])
