"""Opt-in live shutdown regression using local backends and OpenAI Realtime.

Enable LOCAL_USER_LIVE_TEST_ENABLED=1 and set LOCAL_USER_ASR_MODEL_PATH.
The normal voice tier deselects these tests via full_duplex_integration.
"""

import asyncio
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("soundfile", reason="requires the local-user extra")
pytest.importorskip("soxr", reason="requires the local-user extra")

from experiments.tau_voice import local_user_bench as bench
from experiments.tau_voice.local_user.config import LocalUserConfig
from experiments.tau_voice.local_user.scenarios import load_scenes
from tau2.config import DEFAULT_LOCAL_USER_TARGET_MODEL

pytestmark = [
    pytest.mark.full_duplex_integration,
    pytest.mark.skipif(
        os.environ.get("LOCAL_USER_LIVE_TEST_ENABLED") != "1",
        reason="set LOCAL_USER_LIVE_TEST_ENABLED=1 for live local services",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["cancel", "disconnect"])
async def test_live_failure_keeps_partial_audio_and_releases_resources(
    monkeypatch, tmp_path, fault
):
    """Inject a fault after real playback and retain evidence without resource leaks."""
    cfg = LocalUserConfig(
        asr={"model_path": Path(os.environ["LOCAL_USER_ASR_MODEL_PATH"])}
    )
    output = Path(os.environ.get("LOCAL_USER_LIVE_OUTPUT", str(tmp_path))) / fault
    captured = {}
    for name in ("Recorder", "ScenarioRuntime", "OpenAIRealtimeProvider"):
        constructor = getattr(bench, name)

        def capture(*args, _name=name, _constructor=constructor, **kwargs):
            instance = _constructor(*args, **kwargs)
            captured[_name] = instance
            return instance

        monkeypatch.setattr(bench, name, capture)
    case = next(case for case in load_scenes() if case["id"] == "I01")
    task = asyncio.create_task(
        bench.run_case(cfg, case, output, DEFAULT_LOCAL_USER_TARGET_MODEL)
    )

    async def wait_for_playback():
        while not task.done():
            recorder = captured.get("Recorder")
            if recorder and any(
                e["kind"] == "target_first_play" for e in recorder.events
            ):
                return
            await asyncio.sleep(0.05)
        pytest.fail(f"trial ended before fault injection: {await task}")

    try:
        await asyncio.wait_for(wait_for_playback(), 90)
        if fault == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 30)
        else:
            await captured["OpenAIRealtimeProvider"].ws.close()
            result = await asyncio.wait_for(task, 30)
            assert result["status"] == "error"
        summary = json.loads((output / "summary.json").read_text())
        assert summary["status"] == ("cancelled" if fault == "cancel" else "error")
        assert summary["duration_s"] > 0
        assert summary["observations_drained"]
        assert (output / "stereo.flac").stat().st_size > 0
        assert not captured["Recorder"].worker.is_alive()
        assert not captured["OpenAIRealtimeProvider"].is_connected
        runtime = captured["ScenarioRuntime"]
        assert all(worker.done() for worker in runtime.tasks)
        assert runtime.asr.recognizer is None
        assert runtime.asr.stream is None
        assert not runtime.asr._workers
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
