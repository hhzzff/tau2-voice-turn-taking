"""Gateway integration observed at simulator audio and final evidence seams."""

import json

import httpx
import numpy as np
import pytest

from experiments.tau_voice.duplex_bench import build_dynamic_user
from tau2.data_model.message import AssistantMessage
from tau2.data_model.simulation import AudioNativeConfig
from tau2.user.duplex_events import DuplexScenario


@pytest.mark.parametrize(
    "annotation",
    [
        {"text": "上海", "start_char": 1, "end_char": 3},
        {"text": "上海", "occurrence": 0},
        {"text": "上海", "occurrence": 1},
        {"text": "上海", "start_char": 1, "end_char": 3, "occurrence": 0},
    ],
)
def test_gateway_user_same_call_annotation_delivers_audio_without_logging_key(
    monkeypatch, tmp_path, annotation
):
    script = "去上海。"
    calls = []
    monkeypatch.setenv("TERUMI_BT_AK", "secret-fixture-key")

    def send(client, request, **kwargs):
        calls.append(request)
        body = json.loads(request.content)
        assert body["model"] == "gpt-5.6-luna"
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "chat-1",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-5.6-luna-2026-07-09",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "action": "speak",
                                    "text": script,
                                    "reason": "说明目的地。",
                                    "keyword": annotation,
                                },
                                ensure_ascii=False,
                            ),
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                },
            },
        )

    def post(url, **kwargs):
        assert kwargs["params"]["ak"] == "secret-fixture-key"
        if "/text-to-speech/" in url:
            return httpx.Response(
                200,
                content=np.full(16000, 1200, dtype="<i2").tobytes(),
                headers={"content-type": "audio/pcm"},
            )
        if url.endswith("/speech-to-text"):
            assert kwargs["data"] == {"model_id": "scribe_v1", "language_code": "zh"}
            return httpx.Response(200, json={"text": "去[不清楚]。"})
        assert url.endswith("/forced-alignment")
        return httpx.Response(
            200,
            json={
                "characters": [
                    {"text": c, "start": i * 0.15, "end": (i + 1) * 0.15}
                    for i, c in enumerate(script)
                ]
            },
        )

    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.setattr(httpx, "post", post)
    scenario = DuplexScenario(
        selected_events=[],
        user_model="byted/gpt-5.6-luna",
        topic="出行",
        user_context="去上海",
        acoustic_clarification={"mode": "dropchunk"},
    )
    user = build_dynamic_user(
        scenario,
        tmp_path,
        AudioNativeConfig(),
        "voice-one",
        "voice-two",
        "eleven_multilingual_v2",
        gateway=True,
    )
    state = user.get_init_state()
    outputs = []
    for _ in range(5):
        output, state = user.get_next_chunk(
            state, AssistantMessage(role="assistant", content="", contains_speech=False)
        )
        outputs.append(output.get_audio_bytes())
    assert calls and calls[0].headers["api-key"] == "secret-fixture-key"
    samples = np.frombuffer(b"".join(outputs), dtype="<i2")
    if "occurrence" in annotation and "start_char" in annotation:
        assert np.all(samples == 0)  # Rejected decision, no guessed utterance.
        assert user.save_report()["acoustic_clarification"]["utterances"] == []
        return
    if annotation.get("occurrence") == 1:
        assert np.all(samples == 1200)
        failed = user.save_report()["acoustic_clarification"]["utterances"][0]
        assert failed["status"] == "failed"
        assert failed["failure_reason"] == "keyword_occurrence_not_found"
        assert failed["timing"]["alignment_seconds"] == 0
        return
    assert len(samples) == 16000 and 0 < np.count_nonzero(samples == 0) < 4800
    assert user.perception.config.model == "scribe_v1"
    report = user.save_report()
    assert report["acoustic_clarification"]["utterances"][0]["status"] == "delivered"
    synthesis = json.loads((tmp_path / "synthesis.json").read_text())
    assert synthesis["utterances"][0]["model"] == "eleven_multilingual_v2"
    assert "secret-fixture-key" not in json.dumps(report)

    from tau2.user.duplex_evaluation import finalize_acoustic_evidence

    result = finalize_acoustic_evidence(
        scenario,
        report,
        tmp_path,
        0.2,
        [{"tick_id": n} for n in range(6)],
        transcription_config=user.perception.config,
    )
    evidence = result["acoustic_clarification"]["evidence"]
    assert evidence["transcripts"][0]["transcript"] == "去[不清楚]。"
    assert evidence["asr_config"]["model"] == "scribe_v1"


def test_gateway_native_agent_receives_actual_audio_through_documented_websocket(
    monkeypatch, tmp_path
):
    import asyncio
    import base64
    from collections import deque

    from websockets.protocol import State

    from tau2.agent.discrete_time_audio_native_agent import DiscreteTimeAudioNativeAgent
    from tau2.data_model.audio import AudioEncoding, AudioFormat, audio_bytes_to_string
    from tau2.data_model.message import UserMessage

    sent, connections = [], []

    class Socket:
        state = State.OPEN

        def __init__(self):
            self.queue = deque(
                [{"type": "session.created", "session": {"id": "test-session"}}]
            )

        async def send(self, value):
            event = json.loads(value)
            sent.append(event)
            if event["type"] == "session.update":
                self.queue.append({"type": "session.updated", "session": {}})

        async def recv(self):
            if self.queue:
                return json.dumps(self.queue.popleft())
            await asyncio.sleep(10)

        async def close(self):
            self.state = State.CLOSED

    async def connect(url, **kwargs):
        connections.append((url, kwargs))
        return Socket()

    monkeypatch.setenv("TERUMI_BT_AK", "secret-native-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("websockets.connect", connect)
    agent = DiscreteTimeAudioNativeAgent(
        tools=[],
        domain_policy="Speak naturally.",
        model="byted/gpt-realtime-2.1",
        tick_duration_ms=200,
        send_audio_instant=True,
    )
    state = agent.get_init_state()
    try:
        _, state = agent.get_next_chunk(
            state,
            UserMessage(
                role="user",
                content="",
                is_audio=True,
                audio_content=audio_bytes_to_string(
                    np.full(3200, 1200, dtype="<i2").tobytes()
                ),
                audio_format=AudioFormat(
                    encoding=AudioEncoding.PCM_S16LE, sample_rate=16000
                ),
                contains_speech=True,
            ),
        )
    finally:
        agent.stop(state=state)
    assert (
        connections[0][0]
        == "wss://aidp.bytedance.net/openai/realtime?model=gpt-realtime-2.1"
    )
    assert connections[0][1]["additional_headers"] == {"Api-Key": "secret-native-key"}
    audio = b"".join(
        base64.b64decode(item["audio"])
        for item in sent
        if item["type"] == "input_audio_buffer.append"
    )
    assert len(audio) == 1600  # 200ms of native 8kHz mu-law, not source PCM bytes.


def test_final_conversation_audio_preserves_duration_with_different_track_rates(
    tmp_path,
):
    import audioop
    import wave
    from types import SimpleNamespace

    from tau2.data_model.audio import AudioEncoding, AudioFormat, audio_bytes_to_string
    from tau2.data_model.message import Tick, UserMessage
    from tau2.orchestrator.modes import CommunicationMode
    from tau2.voice.synthesis.conversation_builder import generate_simulation_audio

    user = UserMessage(
        role="user",
        content="",
        is_audio=True,
        audio_content=audio_bytes_to_string(np.full(3200, 1200, dtype="<i2").tobytes()),
        audio_format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=16000),
    )
    assistant = AssistantMessage(
        role="assistant",
        content="",
        is_audio=True,
        audio_content=audio_bytes_to_string(
            audioop.lin2ulaw(np.full(1600, 800, dtype="<i2").tobytes(), 2)
        ),
        audio_format=AudioFormat(encoding=AudioEncoding.ULAW, sample_rate=8000),
    )
    simulation = SimpleNamespace(
        mode=CommunicationMode.FULL_DUPLEX,
        ticks=[
            Tick(
                tick_id=0,
                timestamp="2026-11-02T00:00:00",
                tick_duration_seconds=0.2,
                user_chunk=user,
                agent_chunk=assistant,
            )
        ],
    )
    generate_simulation_audio(simulation, tmp_path)
    with wave.open(str(tmp_path / "both.wav"), "rb") as stream:
        assert (stream.getnchannels(), stream.getframerate(), stream.getnframes()) == (
            2,
            16000,
            3200,
        )


def test_final_acoustic_judgment_requests_json_on_the_gateway(monkeypatch):
    from tau2.user.duplex_evaluation import assess_report, judge_report

    monkeypatch.setenv("TERUMI_BT_AK", "offline-key")
    user = {
        "id": "u1",
        "role": "user",
        "transcript": "I enjoy birdwatching.",
        "start_seconds": 0,
        "end_seconds": 1,
    }
    answer = {
        "id": "a1",
        "role": "assistant",
        "transcript": "Try a pocket bird guide.",
        "start_seconds": 2,
        "end_seconds": 3,
    }
    target = {
        "status": "delivered",
        "target_samples": [10, 20],
        "alignment_response": {"words": [{}]},
        "modified_intervals_samples": [[11, 19]],
        "delivery": [{"sample_start": 0, "sample_end": 100, "received": True}],
    }
    report = {
        "status": "complete",
        "events": [],
        "conversation": {"status": "complete"},
        "acoustic_clarification": {
            "utterances": [target],
            "evidence": {
                "transcripts": [user, answer],
                "context_prefixes": {"a1": [user]},
                "errors": [],
            },
        },
    }

    def send(client, request, **kwargs):
        body = json.loads(request.content)
        prompt = body["messages"][0]["content"]
        if "json" not in prompt.lower():
            return httpx.Response(
                400,
                request=request,
                json={
                    "error": {
                        "message": "messages must contain json",
                        "code": "invalid_request_error",
                        "param": "messages",
                    }
                },
            )
        schema = json.loads(prompt.split("\n")[-1])["response_schema"]["title"]
        content = (
            {
                "request": None,
                "answer": {"segment_id": "a1", "quote": "Try a pocket bird guide."},
                "no_query_certain": True,
                "reason": "No clarification was requested.",
            }
            if schema == "Discovery"
            else {
                "result": "supported_inference",
                "reason": "Interest is in received ASR.",
                "evidence": [{"id": "u1", "quote": "birdwatching"}],
            }
        )
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "chat-judge",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-5.6-luna-2026-07-09",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(content),
                        },
                    }
                ],
            },
        )

    monkeypatch.setattr(httpx.Client, "send", send)
    scenario = DuplexScenario(
        selected_events=[], acoustic_clarification={"mode": "dropchunk"}
    )
    result = judge_report(
        scenario, assess_report(scenario, report, 0.2), "byted/gpt-5.6-luna"
    )
    assert result["acoustic_clarification"]["assessment"]["verdict"] == "pass"
