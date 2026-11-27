"""Native ASR workers must outlive cancelled coroutine wrappers safely."""

import asyncio
import threading
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from experiments.tau_voice.local_user.config import LocalUserConfig
from experiments.tau_voice.local_user.local_asr import XASR
from experiments.tau_voice.local_user.observation import HeardItem, PlaybackObservation
from experiments.tau_voice.local_user.recording import Frame


def test_playback_state_keeps_live_item_identity_and_frames_are_immutable():
    item = HeardItem(item_id="heard", response_id="response", for_turn=1)
    playback = PlaybackObservation(items={item.item_id: item})
    item.started_ns = 1
    item.played += 320
    assert playback.audible_item is item
    assert playback.items["heard"].played == 320
    other = PlaybackObservation()
    assert not other.items
    frame = Frame(pcm=bytes(640), sample_index=320, received_mono_ns=1)
    with pytest.raises(ValidationError):
        frame.sample_index = 640
    with pytest.raises(ValidationError):
        Frame(pcm="not PCM bytes", sample_index=0, received_mono_ns=0)


@pytest.mark.asyncio
async def test_cancelled_decode_finishes_before_asr_releases_native_handles(tmp_path):
    entered, release = threading.Event(), threading.Event()
    results = []

    class Recognizer:
        def is_ready(self, stream):
            entered.set()
            assert release.wait(5), "test did not release ASR worker"
            return False

        def get_result(self, stream):
            results.append(stream)
            return "识别完成"

    asr = XASR(
        LocalUserConfig(asr={"model_path": tmp_path}),
        SimpleNamespace(run_id="cancel-test", emit=lambda *a, **kw: None),
    )
    stream = SimpleNamespace(accept_waveform=lambda *args: None)
    asr.recognizer, asr.stream = Recognizer(), stream
    feed = asyncio.create_task(
        asr.feed(Frame(pcm=bytes(640), sample_index=0, received_mono_ns=0))
    )
    closing = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        feed.cancel()
        with pytest.raises(asyncio.CancelledError):
            await feed
        closing = asyncio.create_task(asr.close())
        await asyncio.sleep(0)
        assert not closing.done(), (
            "close released native handles while decode still ran"
        )
    finally:
        release.set()
        await asyncio.gather(feed, return_exceptions=True)
        if closing is not None:
            await closing
    assert results == [stream]
    assert asr.recognizer is asr.stream is None
