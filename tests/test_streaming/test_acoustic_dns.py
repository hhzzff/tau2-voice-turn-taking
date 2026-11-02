"""DNS inputs observed through complete simulator output ticks."""

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


def write_noise(path, samples, sample_rate=16000):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(np.asarray(samples, dtype="<i2").tobytes())


def make_user(monkeypatch, output, noise_path, *, target=(0.2, 0.6), repair=False):
    payloads = []
    script = "Meet Paris today"
    reply_emitted = False

    def generate(**kwargs):
        nonlocal reply_emitted
        payload = kwargs["messages"][0].content
        payloads.append(payload)
        if len(payloads) == 1:
            decision = {
                "action": "speak",
                "text": script,
                "keyword": {"text": "Paris", "start_char": 5, "end_char": 10},
                "reason": "Describe the destination.",
            }
        elif repair and "Which city" in payload and not reply_emitted:
            reply_emitted = True
            decision = {
                "action": "speak",
                "text": "Paris",
                "reason": "Answer the question.",
            }
        else:
            decision = {"action": "wait", "reason": "Wait for an actual reply."}
        return AssistantMessage(role="assistant", content=json.dumps(decision))

    def align(*args, **kwargs):
        return httpx.Response(
            200,
            json={
                "words": [
                    {"text": "Meet", "start": 0.0, "end": target[0]},
                    {"text": "Paris", "start": target[0], "end": target[1]},
                    {"text": "today", "start": target[1], "end": 1.0},
                ]
            },
        )

    monkeypatch.setenv("ELEVENLABS_API", "offline-test")
    monkeypatch.delenv("TERUMI_BT_AK", raising=False)
    monkeypatch.setattr("tau2.user.duplex_scenario_user.generate", generate)
    monkeypatch.setattr("httpx.post", align)
    monkeypatch.setattr(
        "tau2.user.duplex_scenario_user.synthesize_voice",
        lambda *a, **k: AudioData(
            data=np.full(16000, 3000, dtype="<i2").tobytes(),
            format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=16000),
        ),
    )
    scenario = DuplexScenario(
        selected_events=[],
        acoustic_clarification={"mode": "dns_noise", "noise_path": str(noise_path)},
        assistant_observation="proportional",
        topic="Travel",
        user_context="Paris",
    )
    return build_dynamic_user(
        scenario, output, AudioNativeConfig(), "nova", "onyx", "test"
    ), payloads


def deliver(user, state, seconds=1.0, assistant=""):
    chunks = []
    for _ in range(round(seconds / user.tick_duration_seconds)):
        chunk, state = user.get_next_chunk(
            state,
            AssistantMessage(
                role="assistant", content=assistant, contains_speech=bool(assistant)
            ),
        )
        chunks.append(chunk)
    return (
        np.frombuffer(b"".join(c.get_audio_bytes() for c in chunks), dtype="<i2"),
        state,
        chunks,
    )


def report(user, output):
    user.save_report()
    return json.loads((output / "events.json").read_text())["acoustic_clarification"]


@pytest.mark.parametrize("pairs", [1, 3200])
def test_dns_cross_tick_mix_matches_independent_pcm_reference(
    monkeypatch, tmp_path, pairs
):
    path = tmp_path / "noise.wav"
    write_noise(path, [16384, -16384] * pairs)
    user, payloads = make_user(monkeypatch, tmp_path / "run", path)
    delivered, state, chunks = deliver(user, user.get_init_state())
    expected = np.full(16000, 3000, dtype="<i2")
    # clean RMS=3000, hard noise RMS=30000: +33000 clips, -27000 does not.
    expected[3200:9600] = [32767, -27000] * 3200
    np.testing.assert_array_equal(delivered, expected)
    assert all(not c.content and not c.audio_script_gold for c in chunks)
    row = report(user, tmp_path / "run")["utterances"][0]
    assert row["mode"] == "dns_noise"
    assert row["target_snr_db"] == -20
    assert row["clipped_samples"] == 3200
    assert row["preclip_noise_rms"] / row["clean_target_rms"] == pytest.approx(
        10, rel=1e-5
    )
    assert row["noise_source"] == str(path)
    assert row["noise_offset_samples"] == 0
    assert row["timing"]["noise_preparation_seconds"] >= 0
    assert row["timing"]["degradation_seconds"] >= 0
    deliver(user, state, seconds=2.0)
    future = "\n".join(payloads[1:])
    assert "dns_noise" not in future and str(path) not in future
    assert "clipped_samples" not in future and '"start_char": 5' not in future


def test_prepare_exposes_dns_mode_and_preload_asset_without_reading_it(tmp_path):
    from experiments.tau_voice.duplex_bench import build_parser

    output = tmp_path / "scenario.json"
    absent = tmp_path / "preload-later.wav"
    args = build_parser().parse_args(
        [
            "prepare",
            "--acoustic-clarification",
            "dns_noise",
            "--dns-noise-path",
            str(absent),
            "--output",
            str(output),
        ]
    )
    assert args.func(args) == 0
    saved = DuplexScenario.model_validate_json(output.read_text())
    assert saved.acoustic_clarification.mode == "dns_noise"
    assert saved.acoustic_clarification.noise_path == str(absent)


def test_short_dns_target_uses_cached_resampled_noise_after_source_removed(
    monkeypatch, tmp_path
):
    assets = tmp_path / "assets"
    assets.mkdir()
    path = assets / "noise.wav"
    write_noise(path, [16384] * 4800, sample_rate=48000)
    user, _ = make_user(
        monkeypatch, tmp_path / "run", assets, target=(0.2, 0.2 + 1 / 16000)
    )
    # Neither state reset nor runtime can reopen or scan the removed source.
    path.unlink()
    assets.rmdir()
    user.get_init_state()
    delivered, _, _ = deliver(user, user.get_init_state())
    expected = np.full(16000, 3000, dtype="<i2")
    expected[3200] = 32767
    np.testing.assert_array_equal(delivered, expected)
    row = report(user, tmp_path / "run")["utterances"][0]
    assert row["noise_sample_rate"] == 16000
    assert row["noise_segment_samples"] == 1
    assert row["noise_preparation"]["sha256"]


@pytest.mark.parametrize(
    "asset,reason",
    [
        ("missing", "noise_asset_unavailable_or_invalid"),
        ("malformed", "noise_asset_unavailable_or_invalid"),
        ("truncated_header", "noise_asset_unavailable_or_invalid"),
        ("silent", "noise_below_rms_threshold"),
        ("no_fragment", "noise_no_valid_fragment"),
    ],
)
def test_unusable_dns_is_explicit_failed_input_with_clean_ticks(
    monkeypatch, tmp_path, asset, reason
):
    path = tmp_path / "noise.wav"
    if asset == "malformed":
        path.write_bytes(b"not a wav")
    elif asset == "truncated_header":
        path.write_bytes(b"RIFF")
    elif asset in {"silent", "no_fragment"}:
        values = np.zeros(32000, dtype="<i2")
        if asset == "no_fragment":
            values[0] = (
                1000  # Above peak threshold, always below fragment RMS threshold.
            )
        write_noise(path, values)
    user, _ = make_user(monkeypatch, tmp_path / "run", path)
    delivered, _, _ = deliver(user, user.get_init_state())
    np.testing.assert_array_equal(delivered, np.full(16000, 3000, dtype="<i2"))
    row = report(user, tmp_path / "run")["utterances"][0]
    assert row["status"] == "failed"
    assert row["failure_reason"] == reason
    assert row["delivered_modified_samples"] == 0
    assert row["clean_sha256"] == row["degraded_sha256"]
    assert row["timing"]["noise_preparation_seconds"] >= 0


def test_dns_seed_reproduces_output_without_changing_global_random_state(
    monkeypatch, tmp_path
):
    path = tmp_path / "noise.wav"
    write_noise(path, np.random.default_rng(73).integers(-20000, 20001, 20000))
    before = np.random.get_state()
    results = []
    for name in ("one", "two"):
        user, _ = make_user(monkeypatch, tmp_path / name, path)
        delivered, _, _ = deliver(user, user.get_init_state())
        row = report(user, tmp_path / name)["utterances"][0]
        results.append(
            (delivered.tobytes(), row["noise_offset_samples"], row["noise_scale"])
        )
    assert results[0] == results[1]
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


def test_dns_only_corrupts_original_utterance_and_keeps_natural_repair_clear(
    monkeypatch, tmp_path
):
    path = tmp_path / "noise.wav"
    write_noise(path, [16384, -16384] * 3200)
    output = tmp_path / "run"
    user, payloads = make_user(monkeypatch, output, path, repair=True)
    _, state, _ = deliver(user, user.get_init_state())
    _, state, _ = deliver(user, state, seconds=0.2, assistant="Which city did you say?")
    deliver(user, state, seconds=3.0)
    evidence = report(user, output)
    assert len(evidence["utterances"]) == 1
    assert len(evidence["user_utterances"]) == 2
    reply = evidence["user_utterances"][1]
    assert reply["script_text"] == "Paris"
    assert "target_samples" not in reply and "mode" not in reply
    with wave.open(str(output / reply["delivered_audio_file"]), "rb") as wav:
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    np.testing.assert_array_equal(samples, np.full(16000, 3000, dtype="<i2"))
    future = "\n".join(payloads[1:])
    for private in ("dns_noise", "target_snr_db", str(path), "noise_scale"):
        assert private not in future


def test_prepare_rejects_noise_path_without_acoustic_mode(tmp_path):
    from experiments.tau_voice.duplex_bench import build_parser

    args = build_parser().parse_args(
        [
            "prepare",
            "--events",
            "follow_up",
            "--dns-noise-path",
            "noise.wav",
            "--output",
            str(tmp_path / "scenario.json"),
        ]
    )
    with pytest.raises(ValueError, match="requires an acoustic task"):
        args.func(args)
    assert not args.output.exists()
