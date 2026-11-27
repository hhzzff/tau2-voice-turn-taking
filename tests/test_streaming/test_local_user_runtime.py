"""Streaming, observation and delivery contracts for the migrated user."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("soundfile", reason="requires the local-user extra")

from experiments.tau_voice.local_user.config import LocalUserConfig
from experiments.tau_voice.local_user.local_asr import XASR
from experiments.tau_voice.local_user.observation import HeardItem
from experiments.tau_voice.local_user.recording import Recorder, Track, now
from experiments.tau_voice.local_user.scenario import ScenarioRuntime
from experiments.tau_voice.local_user.scenarios import load_scenes
from experiments.tau_voice.local_user.streaming import SpeechStream


def config():
    return LocalUserConfig(asr={"model_path": Path("unused-in-unit-tests")})


@pytest.mark.asyncio
async def test_first_audio_does_not_wait_for_whole_generation(tmp_path):
    release = asyncio.Event()

    class LLM:
        async def segments(self, *args, **kwargs):
            yield "先说这一句。"
            await release.wait()
            yield "后面继续。"

    class TTS:
        async def stream(self, text):
            yield b"\x00\x10" * 320

    events = []
    speech = SpeechStream(
        config(), LLM(), TTS(), lambda kind, **values: events.append(kind)
    )
    stream = speech.audio()
    try:
        first = await asyncio.wait_for(anext(stream), 1)
        assert first == b"\x00\x10" * 320
        assert not release.is_set()
        assert "llm_complete" not in events
        release.set()
        remaining = [pcm async for pcm in stream]
        assert len(remaining) == 1
        assert "llm_complete" in events
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_audio_is_not_delivered_when_it_is_only_dequeued(tmp_path):
    rec = Recorder(tmp_path / "delivery")
    async with httpx.AsyncClient() as client:
        user = ScenarioRuntime(config(), load_scenes()[0], rec, client)
        user.origin = now()
        user.user_track = Track(user.origin, 16000)
        state = user.snapshot()
        turn = {
            "index": 1,
            "marks": {},
            "pending_intent": state["pending_intent"],
            "speech_gate_state": state,
            "action": "RESPOND",
            "sent_pcm": bytearray(),
            "sent_source_samples": 0,
            "done": asyncio.Event(),
        }
        user.user_queue.put_nowait((turn, b"\x00\x10" * 320))
        packet = user.poll_output()
        assert turn["marks"] == {}
        assert user.sent_turn == 0
        user.acknowledge_delivery(packet["sequence"], now())
        assert user.sent_turn == 1
        assert turn["sent_source_samples"] == 320
        assert "user_first_voice_tx" in turn["marks"]
        assert "user_audio_complete" not in turn["marks"]
        user.acknowledge_delivery(packet["sequence"], now())
        assert turn["sent_source_samples"] == 320
        user.user_queue.put_nowait((turn, None))
        for _ in range(4):
            packet = user.poll_output()
            user.acknowledge_delivery(packet["sequence"], now())
        assert turn["done"].is_set()
        assert "user_audio_complete" in turn["marks"]
    await rec.close()


@pytest.mark.asyncio
async def test_stale_output_is_silenced_before_transport(tmp_path):
    rec = Recorder(tmp_path / "stale")
    async with httpx.AsyncClient() as client:
        user = ScenarioRuntime(config(), load_scenes()[0], rec, client)
        state = user.snapshot()
        user.progress.version += 1
        turn = {
            "index": 1,
            "marks": {},
            "speech_gate_state": state,
            "action": "RESPOND",
            "sent_pcm": bytearray(),
        }
        user.user_queue.put_nowait((turn, b"\x00\x10" * 320))
        packet = user.poll_output()
        assert packet["pcm"] == bytes(640)
        assert packet["source_samples"] == 0
        assert turn["abort_before_voice"]
    await rec.close()


@pytest.mark.asyncio
async def test_only_played_audio_enters_the_asr_queue(tmp_path):
    rec = Recorder(tmp_path / "observation")
    async with httpx.AsyncClient() as client:
        user = ScenarioRuntime(config(), load_scenes()[0], rec, client)
        user.agent_track = Track(now(), 16000)
        item = HeardItem(
            item_id="item", response_id="response", for_turn=0, buffered_samples=16000
        )
        user.playback.items[item.item_id] = item
        assert user.asr_queue.empty()
        assert user.snapshot()["currently_heard_agent"] == ""
        user.begin_agent(item)
        user.agent_frame(item, b"\x00\x10" * 320, now())
        assert user.asr_queue.get_nowait()[0] == "begin"
        kind, _, frame = user.asr_queue.get_nowait()
        assert kind == "frame" and len(frame.pcm) == 640
        assert user.asr_queue.empty()
    await rec.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("active", [True, False])
async def test_reserved_feedback_checks_current_voice_not_used_quota(tmp_path, active):
    rec = Recorder(tmp_path / "reserved-feedback")
    async with httpx.AsyncClient() as client:
        user = ScenarioRuntime(config(), load_scenes()[0], rec, client)
        state = {
            "floor_permission": {"kind": "LISTEN"},
            "audio_state": {"voice_active": active},
            "observed_item_id": "same-utterance",
            "backchannel": {"available": False, "reason": "segment_limit"},
        }
        user.snapshot = lambda: state
        turn = {
            "index": 1,
            "kind": "backchannel",
            "marks": {},
            "speech_gate_state": state,
            "sent_pcm": bytearray(),
        }
        user.user_queue.put_nowait((turn, b"\x00\x10" * 320))
        packet = user.poll_output()
        assert packet["voice"] is active
        assert packet["source_samples"] == (320 if active else 0)
    await rec.close()


@pytest.mark.asyncio
async def test_asr_same_length_revision_is_published():
    rec = SimpleNamespace(run_id="asr-test", emit=lambda *a, **kw: None)
    asr = XASR(config(), rec)
    asr._publish(("周五", 1, 10, 20))
    asr._publish(("周六", 1, 30, 40))
    first, second = asr.q.get_nowait(), asr.q.get_nowait()
    assert first.text == "周五" and second.text == "周六"
    assert second.revision > first.revision
