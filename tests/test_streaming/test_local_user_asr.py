"""Native ASR workers must outlive cancelled coroutine wrappers safely."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from experiments.tau_voice.local_user.config import LocalUserConfig
from experiments.tau_voice.local_user.local_asr import XASR
from experiments.tau_voice.local_user.recording import Frame


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
    feed = asyncio.create_task(asr.feed(Frame(bytes(640), 0, 0)))
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
