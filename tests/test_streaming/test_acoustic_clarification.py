"""Acoustic clarification observed through actual simulator output ticks."""

import hashlib
import io
import json
import wave

import httpx
import numpy as np
import pytest

from experiments.tau_voice.duplex_bench import build_dynamic_user
from tau2.data_model.audio import (
    PCM_SAMPLE_RATE,
    AudioData,
    AudioEncoding,
    AudioFormat,
    audio_bytes_to_string,
)
from tau2.data_model.message import AssistantMessage
from tau2.data_model.simulation import AudioNativeConfig
from tau2.user.duplex_events import DuplexScenario


def test_complete_tts_dropchunk_crosses_ticks_without_leaking_annotation(
    monkeypatch, tmp_path
):
    script = "先到南门，再到南门。"
    payloads, alignment_calls = [], []
    clean = np.full(16000, 1200, dtype="<i2")

    def generate(**kwargs):
        payloads.append(kwargs["messages"][0].content)
        decision = (
            {
                "action": "speak",
                "text": script,
                "keyword": {"text": "南门", "start_char": 7, "end_char": 9},
                "reason": "说明集合路线。",
            }
            if len(payloads) == 1
            else {"action": "wait", "reason": "等待实际回答。"}
        )
        return AssistantMessage(role="assistant", content=json.dumps(decision))

    def align(url, **kwargs):
        alignment_calls.append((url, kwargs))
        with wave.open(io.BytesIO(kwargs["files"]["file"][1]), "rb") as wav:
            assert wav.getnframes() == PCM_SAMPLE_RATE
        return httpx.Response(
            200,
            json={
                "characters": [
                    {"text": char, "start": i / 10, "end": (i + 1) / 10}
                    for i, char in enumerate(script)
                ],
                "words": [],
                "loss": 0.1,
            },
        )

    monkeypatch.delenv("TERUMI_BT_AK", raising=False)
    monkeypatch.setenv("ELEVENLABS_API", "offline-test-token")
    monkeypatch.setattr("tau2.user.duplex_scenario_user.generate", generate)
    monkeypatch.setattr("httpx.post", align)
    monkeypatch.setattr(
        "tau2.user.duplex_scenario_user.synthesize_voice",
        lambda *a, **k: AudioData(
            data=clean.tobytes(),
            format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=16000),
        ),
    )
    config = DuplexScenario(
        selected_events=[],
        acoustic_clarification={"mode": "dropchunk"},
        assistant_observation="proportional",
        topic="集合安排",
        user_context="先南门后南门",
    )
    user = build_dynamic_user(
        config, tmp_path, AudioNativeConfig(), "nova", "onyx", "test"
    )
    state, chunks = user.get_init_state(), []
    for _ in range(round(1 / user.tick_duration_seconds)):
        chunk, state = user.get_next_chunk(
            state, AssistantMessage(role="assistant", content="", contains_speech=False)
        )
        chunks.append(chunk)
    delivered = np.frombuffer(
        b"".join(c.get_audio_bytes() for c in chunks), dtype="<i2"
    )
    assert len(delivered) == PCM_SAMPLE_RATE
    assert np.all(delivered[: int(0.7 * PCM_SAMPLE_RATE)] == 1200)
    assert np.all(delivered[int(0.9 * PCM_SAMPLE_RATE) :] == 1200)
    assert (
        np.count_nonzero(
            delivered[int(0.7 * PCM_SAMPLE_RATE) : int(0.9 * PCM_SAMPLE_RATE)] == 0
        )
        > 0.65 * 0.2 * PCM_SAMPLE_RATE
    )
    assert all(not c.content and not c.audio_script_gold for c in chunks)
    user.save_report()
    record = json.loads((tmp_path / "events.json").read_text())[
        "acoustic_clarification"
    ]["utterances"][0]
    assert record["keyword"]["start_char"] == 7
    assert record["target_samples"] == [
        int(0.7 * PCM_SAMPLE_RATE),
        int(0.9 * PCM_SAMPLE_RATE),
    ]
    assert len(record["drop_intervals_samples"]) == 2
    assert record["clean_sha256"] != record["degraded_sha256"]
    assert record["degraded_sha256"] == hashlib.sha256(delivered.tobytes()).hexdigest()
    assert record["delivered_target_samples"] == int(0.2 * PCM_SAMPLE_RATE)
    assert len(alignment_calls) == 1
    assert alignment_calls[0][0] == "https://api.elevenlabs.io/v1/forced-alignment"
    assert alignment_calls[0][1]["data"] == {"text": script}
    for key in ("alignment_seconds", "degradation_seconds", "tts_postprocess_seconds"):
        assert record["timing"][key] >= 0
    # Further model calls must contain no execution metadata or the neutral annotation.
    for _ in range(30):
        user.get_next_chunk(
            state, AssistantMessage(role="assistant", content="", contains_speech=False)
        )
    assert len(payloads) >= 2
    future = "\n".join(payloads[1:])
    assert "dropchunk" not in future and "degraded_sha256" not in future
    assert '"start_char": 7' not in future


@pytest.mark.parametrize(
    "query_at,event_id,span_step",
    [
        (0.3, None, 0.025),
        (2.2, None, 0.025),
        (0.3, "proactive.backchannel", 0.025),
        (0.3, None, 0.3),
    ],
)
def test_agent_question_uses_existing_yield_and_clear_natural_reply(
    monkeypatch, tmp_path, query_at, event_id, span_step
):
    scripts, calls = ["我先去南门，然后等大家一起出发。", "南门。"], []
    uttered = []

    def generate(**kwargs):
        payload = kwargs["messages"][0].content
        calls.append(payload)
        if not uttered:
            uttered.append(scripts[0])
            decision = {
                "action": "trigger_event" if event_id else "speak",
                "text": scripts[0],
                "keyword": {"text": "南门", "start_char": 3, "end_char": 5},
                "reason": "说明计划。",
            }
            if event_id:
                decision["event_id"] = event_id
        elif "哪个门" in payload and len(uttered) == 1:
            uttered.append(scripts[1])
            decision = {
                "action": "speak",
                "text": scripts[1],
                "keyword": {"text": "南门", "start_char": 0, "end_char": 2},
                "reason": "回答刚才实际的问题。",
            }
        else:
            decision = {"action": "wait", "reason": "等待对方。"}
        return AssistantMessage(role="assistant", content=json.dumps(decision))

    def tts(text, *args):
        return AudioData(
            data=np.full(
                (96000 if span_step == 0.3 else 32000) if text == scripts[0] else 8000,
                1500,
                dtype="<i2",
            ).tobytes(),
            format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=16000),
        )

    def align(*args, **kwargs):
        return httpx.Response(
            200,
            json={
                "characters": [
                    {"text": char, "start": i * span_step, "end": (i + 1) * span_step}
                    for i, char in enumerate(scripts[0])
                ]
            },
        )

    monkeypatch.setattr("tau2.user.duplex_scenario_user.generate", generate)
    monkeypatch.setattr("tau2.user.duplex_scenario_user.synthesize_voice", tts)
    monkeypatch.setattr("httpx.post", align)
    monkeypatch.setenv("ELEVENLABS_API", "test")
    monkeypatch.delenv("TERUMI_BT_AK", raising=False)
    scenario = DuplexScenario(
        selected_events=[event_id] if event_id else [],
        acoustic_clarification={"mode": "dropchunk"},
        assistant_observation="proportional",
        language="Chinese",
    )
    user = build_dynamic_user(
        scenario, tmp_path, AudioNativeConfig(), "nova", "onyx", "test"
    )
    state, outputs = user.get_init_state(), []
    for tick in range(120):
        now = tick * user.tick_duration_seconds
        asking = query_at <= now < query_at + 0.4
        incoming = AssistantMessage(
            role="assistant",
            content="哪个门？" if asking else "",
            is_audio=True,
            audio_content=audio_bytes_to_string(
                np.full(user.chunk_size, 2000 if asking else 0, dtype="<i2").tobytes()
            ),
            audio_format=AudioFormat(
                encoding=AudioEncoding.PCM_S16LE, sample_rate=PCM_SAMPLE_RATE
            ),
            contains_speech=asking,
        )
        output, state = user.get_next_chunk(state, incoming)
        outputs.append(output)
    user.save_report()
    acoustic = json.loads((tmp_path / "events.json").read_text())[
        "acoustic_clarification"
    ]
    target = acoustic["utterances"][0]
    assert uttered == scripts
    if query_at < 2:
        assert target["actual_stop_seconds"] < target["natural_end_seconds"]
        assert target["cancelled_samples"][0] < target["cancelled_samples"][1]
        assert any(
            chunk.turn_taking_action.action == "stop_talking" for chunk in outputs
        )
    else:
        assert target["actual_stop_seconds"] == target["natural_end_seconds"]
        assert target["cancelled_samples"] == []
    assert len(acoustic["user_utterances"]) == 2
    repair = acoustic["user_utterances"][1]
    assert repair["script_is_verified_transcript"] is False
    with wave.open(str(tmp_path / repair["delivered_audio_file"]), "rb") as wav:
        repaired = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        assert np.all(repaired[:7900] == 1500)
        assert np.all((repaired[7900:8000] >= 0) & (repaired[7900:8000] <= 1500))
        assert np.all(repaired[8000:] == 0)  # Existing final-tick padding.
    if span_step == 0.3:
        assert target["status"] == "incomplete"
        assert (
            target["delivered_modified_samples"]
            == target["delivered_target_samples"]
            == 0
        )
    else:
        assert target["delivered_modified_samples"] > 0
        assert (
            target["delivered_target_samples"]
            == target["target_samples"][1] - target["target_samples"][0]
        )


def test_gateway_alignment_uses_repeated_english_word_time_after_resampling(
    monkeypatch, tmp_path
):
    script, calls = "red or red", []

    def generate(**kwargs):
        decision = {
            "action": "speak",
            "text": script,
            "keyword": {"text": "red", "start_char": 7, "end_char": 10},
            "reason": "Choose a color.",
        }
        return AssistantMessage(role="assistant", content=json.dumps(decision))

    def align(url, **kwargs):
        calls.append((url, kwargs))
        return httpx.Response(
            200,
            json={
                "words": [
                    {"text": "red", "start": 0.1, "end": 0.2},
                    {"text": "or", "start": 0.3, "end": 0.4},
                    {"text": "red", "start": 0.6, "end": 0.8},
                ],
                "characters": [],
            },
        )

    monkeypatch.setenv("TERUMI_BT_AK", "gateway-offline-secret")
    monkeypatch.setenv("ELEVENLABS_API", "official-offline-secret")
    monkeypatch.setattr("tau2.user.duplex_scenario_user.generate", generate)
    monkeypatch.setattr("httpx.post", align)
    monkeypatch.setattr(
        "tau2.user.duplex_scenario_user.synthesize_voice",
        lambda *a, **k: AudioData(
            data=np.full(24000, 1000, dtype="<i2").tobytes(),
            format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=24000),
        ),
    )
    config = DuplexScenario(
        acoustic_clarification={"mode": "dropchunk"},
        assistant_observation="proportional",
        language="English",
    )
    user = build_dynamic_user(
        config, tmp_path, AudioNativeConfig(), "nova", "onyx", "test"
    )
    state, chunks = user.get_init_state(), []
    for _ in range(round(1 / user.tick_duration_seconds)):
        chunk, state = user.get_next_chunk(
            state, AssistantMessage(role="assistant", content="", contains_speech=False)
        )
        chunks.append(chunk)
    user.save_report()
    target = json.loads((tmp_path / "events.json").read_text())[
        "acoustic_clarification"
    ]["utterances"][0]
    output = np.frombuffer(b"".join(c.get_audio_bytes() for c in chunks), dtype="<i2")
    assert len(output) == PCM_SAMPLE_RATE
    assert np.all(
        (output[1600:3200] >= 999) & (output[1600:3200] <= 1001)
    )  # First occurrence is preserved.
    assert np.count_nonzero(output[9600:12800] == 0) > 2080
    assert target["target_samples"] == [9600, 12800]
    assert len(calls) == 1
    url, kwargs = calls[0]
    assert (
        url
        == "https://aidp.bytedance.net/api/modelhub/online/multimodal/crawl/11labs/v1/forced-alignment"
    )
    assert kwargs["params"] == {"ak": "gateway-offline-secret"}
    assert "xi-api-key" not in kwargs["headers"]
    assert "gateway-offline-secret" not in (tmp_path / "events.json").read_text()


@pytest.mark.parametrize(
    "failure,expected,attempts",
    [
        ("missing", "missing_keyword_annotation", 0),
        ("invalid_occurrence", "keyword_text_position_mismatch", 0),
        ("unauthorized", "alignment_http_401", 1),
        ("unavailable", "alignment_http_503", 3),
        ("timeout", "alignment_transport_failure", 3),
        ("mismatch", "alignment_text_mismatch", 1),
        ("silent", "no_effective_modification", 1),
    ],
)
def test_invalid_acoustic_input_is_explicit_and_never_claimed_as_delivered_damage(
    monkeypatch, tmp_path, failure, expected, attempts
):
    script, requests = "red or red", []
    waveform = np.full(PCM_SAMPLE_RATE, 1000, dtype="<i2")
    if failure == "silent":
        waveform[9600:12800] = 0

    def generate(**kwargs):
        return AssistantMessage(
            role="assistant",
            content=json.dumps(
                {
                    "action": "speak",
                    "text": script,
                    "keyword": None
                    if failure == "missing"
                    else {
                        "text": "blue" if failure == "invalid_occurrence" else "red",
                        "start_char": 7,
                        "end_char": 10,
                    },
                    "reason": "Choose a color.",
                }
            ),
        )

    def align(*args, **kwargs):
        requests.append(kwargs)
        if failure == "timeout":
            raise httpx.ReadTimeout("Do not copy this private URL or token")
        if failure in {"unauthorized", "unavailable"}:
            return httpx.Response(401 if failure == "unauthorized" else 503)
        return httpx.Response(
            200,
            json={
                "words": [
                    {
                        "text": "wrong" if failure == "mismatch" else "red",
                        "start": 0.1,
                        "end": 0.2,
                    },
                    {"text": "or", "start": 0.3, "end": 0.4},
                    {"text": "red", "start": 0.6, "end": 0.8},
                ]
            },
        )

    monkeypatch.delenv("TERUMI_BT_AK", raising=False)
    monkeypatch.setenv("ELEVENLABS_API", "test")
    monkeypatch.setattr("tau2.user.duplex_scenario_user.generate", generate)
    monkeypatch.setattr("httpx.post", align)
    monkeypatch.setattr(
        "tau2.user.duplex_scenario_user.synthesize_voice",
        lambda *a, **k: AudioData(
            data=waveform.tobytes(),
            format=AudioFormat(
                encoding=AudioEncoding.PCM_S16LE, sample_rate=PCM_SAMPLE_RATE
            ),
        ),
    )
    config = DuplexScenario(
        acoustic_clarification={"mode": "dropchunk"},
        assistant_observation="proportional",
    )
    user = build_dynamic_user(
        config, tmp_path, AudioNativeConfig(), "nova", "onyx", "test"
    )
    state, chunks = user.get_init_state(), []
    for _ in range(round(1 / user.tick_duration_seconds)):
        chunk, state = user.get_next_chunk(
            state, AssistantMessage(role="assistant", content="", contains_speech=False)
        )
        chunks.append(chunk)
    user.save_report()
    record = json.loads((tmp_path / "events.json").read_text())[
        "acoustic_clarification"
    ]["utterances"][0]
    output = np.frombuffer(b"".join(c.get_audio_bytes() for c in chunks), dtype="<i2")
    assert np.array_equal(output, waveform)
    assert record["status"] == "failed"
    assert record["failure_reason"] == expected
    assert record["delivered_modified_samples"] == 0
    assert len(requests) == attempts
    assert "Do not copy" not in (tmp_path / "events.json").read_text()


def test_standalone_conversation_cannot_finish_before_an_ordinary_exchange(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "tau2.user.duplex_scenario_user.generate",
        lambda **kwargs: AssistantMessage(
            role="assistant",
            content=json.dumps({"action": "finish", "reason": "No pending events."}),
        ),
    )
    user = build_dynamic_user(
        DuplexScenario(
            acoustic_clarification={"mode": "dropchunk"},
            assistant_observation="proportional",
        ),
        tmp_path,
        AudioNativeConfig(),
        "nova",
        "onyx",
        "test",
    )
    output, _ = user.get_next_chunk(
        user.get_init_state(),
        AssistantMessage(role="assistant", content="", contains_speech=False),
    )
    assert not user.is_stop(output)
    assert not user.controller.finished
    user.save_report()
    report = json.loads((tmp_path / "events.json").read_text())
    assert report["status"] == "incomplete"
    assert report["acoustic_clarification"]["status"] == "not_reached"
    assert "dropchunk" not in user.controller.feedback


def test_delivered_assistant_telephony_audio_is_saved_as_decoded_pcm(
    monkeypatch, tmp_path
):
    import audioop

    from tau2.data_model.audio import audio_bytes_to_string

    pcm = np.full(1600, 1000, dtype="<i2").tobytes()
    encoded = audioop.lin2ulaw(pcm, 2)
    incoming = AssistantMessage(
        role="assistant",
        content="哪个门？",
        is_audio=True,
        audio_content=audio_bytes_to_string(encoded),
        contains_speech=True,
        audio_format=AudioFormat(encoding=AudioEncoding.ULAW, sample_rate=8000),
    )
    user = build_dynamic_user(
        DuplexScenario(
            acoustic_clarification={"mode": "dropchunk"},
            assistant_observation="proportional",
        ),
        tmp_path,
        AudioNativeConfig(),
        "nova",
        "onyx",
        "test",
    )
    output, _ = user.get_next_chunk(user.get_init_state(), incoming)
    assert output.get_audio_bytes()
    user.save_report()
    evidence = json.loads((tmp_path / "events.json").read_text())[
        "acoustic_clarification"
    ]["assistant_observations"][0]
    with wave.open(str(tmp_path / evidence["audio_file"]), "rb") as wav:
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 8000
        assert wav.readframes(wav.getnframes()) == audioop.ulaw2lin(encoded, 2)
    assert evidence["sha256"] == hashlib.sha256(encoded).hexdigest()
    assert evidence["original_format"]["encoding"] == "ulaw"


@pytest.mark.parametrize(
    "encoding,rate", [(AudioEncoding.ULAW, 8000), (AudioEncoding.PCM_S16LE, 24000)]
)
def test_acoustic_ticks_reach_native_provider_in_its_configured_codec(
    monkeypatch, tmp_path, encoding, rate
):
    import audioop

    from tau2.agent.discrete_time_audio_native_agent import DiscreteTimeAudioNativeAgent
    from tau2.voice.audio_native.openai.provider import OpenAIRealtimeProvider

    sent, configured = [], []

    async def no_op(self, *args, **kwargs):
        return None

    async def configure(self, **kwargs):
        configured.append(kwargs)

    async def send(self, audio):
        sent.append(audio)

    async def receive(self, duration):
        return []

    for name, callback in (
        ("connect", no_op),
        ("disconnect", no_op),
        ("configure_session", configure),
        ("send_audio", send),
        ("receive_events_for_duration", receive),
    ):
        monkeypatch.setattr(OpenAIRealtimeProvider, name, callback)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-provider-key")
    monkeypatch.delenv("TERUMI_BT_AK", raising=False)
    monkeypatch.setenv("ELEVENLABS_API", "offline-alignment-key")
    monkeypatch.setattr(
        "tau2.user.duplex_scenario_user.generate",
        lambda **kwargs: AssistantMessage(
            role="assistant",
            content=json.dumps(
                {
                    "action": "speak",
                    "text": "red",
                    "keyword": {"text": "red", "start_char": 0, "end_char": 3},
                    "reason": "Choose a color.",
                }
            ),
        ),
    )
    monkeypatch.setattr(
        "tau2.user.duplex_scenario_user.synthesize_voice",
        lambda *a, **k: AudioData(
            data=np.full(16000, 1000, dtype="<i2").tobytes(),
            format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=16000),
        ),
    )
    monkeypatch.setattr(
        "httpx.post",
        lambda *a, **k: httpx.Response(
            200, json={"words": [{"text": "red", "start": 0.2, "end": 0.8}]}
        ),
    )
    config = DuplexScenario(
        acoustic_clarification={"mode": "dropchunk"},
        assistant_observation="proportional",
    )
    native_config = AudioNativeConfig()
    user = build_dynamic_user(config, tmp_path, native_config, "nova", "onyx", "test")
    agent = DiscreteTimeAudioNativeAgent(
        tools=[],
        domain_policy="Ordinary conversation.",
        tick_duration_ms=native_config.tick_duration_ms,
        send_audio_instant=True,
        audio_format=AudioFormat(encoding=encoding, sample_rate=rate),
    )
    state, agent_state = user.get_init_state(), agent.get_init_state()
    try:
        for _ in range(round(1 / user.tick_duration_seconds)):
            chunk, state = user.get_next_chunk(
                state,
                AssistantMessage(role="assistant", content="", contains_speech=False),
            )
            assert not chunk.content and not chunk.audio_script_gold
            _, agent_state = agent.get_next_chunk(agent_state, chunk)
    finally:
        agent.stop(state=agent_state)
    delivered = b"".join(sent)
    assert configured[0]["audio_format"].encoding == encoding
    assert configured[0]["audio_format"].sample_rate == rate
    assert len(delivered) == rate * (1 if encoding == AudioEncoding.ULAW else 2)
    decoded = (
        audioop.ulaw2lin(delivered, 2) if encoding == AudioEncoding.ULAW else delivered
    )
    samples = np.frombuffer(decoded, dtype="<i2")
    assert (
        np.count_nonzero(samples[int(0.2 * rate) : int(0.8 * rate)] == 0)
        > 0.6 * 0.6 * rate
    )
    assert np.max(np.abs(samples[: int(0.1 * rate)])) < 1200


def test_audio_only_assistant_reply_can_complete_a_standalone_conversation(
    monkeypatch, tmp_path
):
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        decision = (
            {
                "action": "speak",
                "text": "red",
                "keyword": {"text": "red", "start_char": 0, "end_char": 3},
                "reason": "Choose a color.",
            }
            if len(calls) == 1
            else {"action": "finish", "reason": "The ordinary exchange is complete."}
        )
        return AssistantMessage(role="assistant", content=json.dumps(decision))

    monkeypatch.delenv("TERUMI_BT_AK", raising=False)
    monkeypatch.setenv("ELEVENLABS_API", "test")
    monkeypatch.setattr("tau2.user.duplex_scenario_user.generate", generate)
    monkeypatch.setattr(
        "tau2.user.duplex_scenario_user.synthesize_voice",
        lambda *a, **k: AudioData(
            data=np.full(3200, 1000, dtype="<i2").tobytes(),
            format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=16000),
        ),
    )
    monkeypatch.setattr(
        "httpx.post",
        lambda *a, **k: httpx.Response(
            200, json={"words": [{"text": "red", "start": 0, "end": 0.1}]}
        ),
    )
    user = build_dynamic_user(
        DuplexScenario(
            acoustic_clarification={"mode": "dropchunk"},
            assistant_observation="proportional",
        ),
        tmp_path,
        AudioNativeConfig(),
        "nova",
        "onyx",
        "test",
    )
    state, stopped = user.get_init_state(), False
    for tick in range(20):
        audible = 0.4 <= tick * user.tick_duration_seconds < 0.8
        incoming = AssistantMessage(
            role="assistant",
            content="",
            contains_speech=audible,
            is_audio=True,
            audio_content=audio_bytes_to_string(
                np.full(user.chunk_size, 1500 if audible else 0, dtype="<i2").tobytes()
            ),
            audio_format=AudioFormat(
                encoding=AudioEncoding.PCM_S16LE, sample_rate=PCM_SAMPLE_RATE
            ),
        )
        chunk, state = user.get_next_chunk(state, incoming)
        if user.is_stop(chunk):
            stopped = True
            break
    assert stopped
