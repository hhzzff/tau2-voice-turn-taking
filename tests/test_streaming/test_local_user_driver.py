"""Continuous timing, native PCM and observation isolation regressions."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from websockets.protocol import State

from tau2.orchestrator.continuous_voice import ContinuousVoiceDriver
from tau2.user.local.config import LocalUserConfig
from tau2.user.local.observation import HeardItem
from tau2.user.local.recording import Recorder, now
from tau2.user.local.scenario import ScenarioRuntime
from tau2.user.local.scenarios import load_scenes
from tau2.user.local_streaming_user import LocalStreamingUser
from tau2.voice.audio_native.audio_converter import StreamingPCMConverter
from tau2.voice.audio_native.openai.continuous import ContinuousOpenAITarget
from tau2.voice.audio_native.openai.provider import (
    OpenAIRealtimeProvider,
    OpenAIVADConfig,
)


class FakeASR:
    def __init__(self):
        self.frames = 0
        self.segment = 0
        self.q = asyncio.Queue()

    async def reset(self, **kwargs):
        self.segment += 1

    async def feed(self, frame):
        self.frames += 1

    async def events(self):
        while True:
            yield await self.q.get()

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_slow_b_and_tts_do_not_freeze_send_receive_playback_or_asr(tmp_path):
    rec = Recorder(tmp_path / "continuous")
    asr = FakeASR()
    async with httpx.AsyncClient() as client:
        runtime = ScenarioRuntime(
            LocalUserConfig(asr={"model_path": tmp_path}),
            load_scenes()[0],
            rec,
            client,
            asr=asr,
        )
        observed = []

        async def slow_inference():
            for label in ("B", "TTS"):
                before = dict(driver.tick_counts)
                await asyncio.sleep(1.1)
                observed.append((label, before, dict(driver.tick_counts)))

        runtime.control = slow_inference
        user = LocalStreamingUser(runtime)

        class Target:
            def __init__(self):
                self.received = 0
                self.item = HeardItem("audible", "response", 0)

            async def receive(self):
                while True:
                    self.received += 1
                    await asyncio.sleep(0.02)

            def play_frame(self, at):
                if not self.item.started_ns:
                    self.item.started_ns = at
                    runtime.begin_agent(self.item)
                self.item.played += 320
                runtime.agent_frame(self.item, b"\x00\x10" * 320, at)

            async def send_packet(self, packet):
                return now()

        target = Target()
        runtime.begin_live(now())
        driver = ContinuousVoiceDriver(user, target, rec)
        try:
            assert await driver.run(4) == "ok"
            assert target.received > 75 and asr.frames > 75
            for _, before, after in observed:
                assert after["send"] - before["send"] > 35
                assert after["playback"] - before["playback"] > 35
        finally:
            await runtime.close()
    await rec.close()


def test_resampling_preserves_phase_and_duration_across_chunk_boundaries():
    source = (np.sin(np.arange(16000) * 0.2) * 10000).astype("<i2").tobytes()
    whole = StreamingPCMConverter().convert_input(source)
    chunks = StreamingPCMConverter()
    split = b"".join(
        chunks.convert_input(source[n : n + 640]) for n in range(0, len(source), 640)
    )
    assert split == whole
    assert abs(len(split) // 2 - 24000) <= 1
    a, b = StreamingPCMConverter(), StreamingPCMConverter()
    down_a = a.convert_output(whole, final=True)
    down_b = b"".join(
        b.convert_output(whole[n : n + 960]) for n in range(0, len(whole), 960)
    ) + b.convert_output(final=True)
    assert down_a == down_b and abs(len(down_a) // 2 - 16000) <= 1


@pytest.mark.asyncio
async def test_provider_native_options_are_acknowledged_and_raw_events_retained():
    class WS:
        state = State.OPEN

        async def send(self, data):
            self.config = json.loads(data)["session"]

        async def recv(self):
            return json.dumps({"type": "session.updated", "session": self.config})

    provider = OpenAIRealtimeProvider(api_key="unit-test", model="unit-test")
    provider.ws = WS()
    await provider.configure_session(
        "中文",
        [],
        OpenAIVADConfig(),
        modality="audio",
        voice="marin",
        transcription_language=None,
        max_output_tokens=2048,
    )
    assert provider.session_config["audio"]["output"]["voice"] == "marin"
    assert provider.session_config["audio"]["input"]["transcription"] is None
    assert provider.session_config["max_output_tokens"] == 2048


@pytest.mark.asyncio
async def test_future_transcript_is_audit_only_and_sessions_are_isolated(tmp_path):
    records = [Recorder(tmp_path / str(n)) for n in range(2)]
    async with httpx.AsyncClient() as client:
        users = [
            LocalStreamingUser(
                ScenarioRuntime(
                    LocalUserConfig(asr={"model_path": tmp_path}),
                    load_scenes()[0],
                    rec,
                    client,
                )
            )
            for rec in records
        ]
        targets = [
            ContinuousOpenAITarget(SimpleNamespace(model="test"), user, rec)
            for user, rec in zip(users, records)
        ]
        await targets[0]._process_event(
            {
                "type": "response.output_audio_transcript.done",
                "item_id": "future",
                "transcript": "尚未播放的答案",
            }
        )
        assert targets[0].transcripts["future"] == "尚未播放的答案"
        assert users[0].runtime.snapshot()["currently_heard_agent"] == ""
        assert not users[0].runtime.history and not targets[1].transcripts
    for rec in records:
        await rec.close()
