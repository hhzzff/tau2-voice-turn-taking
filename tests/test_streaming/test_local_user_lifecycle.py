"""Failed and cancelled trials preserve evidence and release every resource."""

import asyncio
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("soundfile", reason="requires the local-user extra")

from experiments.tau_voice import local_user_bench as bench
from tau2.user.local.recording import Recorder, Track


@pytest.fixture
def trial(monkeypatch, tmp_path):
    calls = []
    state = SimpleNamespace(fail=None, recorder=None)

    def step(name):
        calls.append(name)
        if state.fail == name or name in getattr(state, "failures", ()):
            raise ValueError(name)
        if state.fail == "cancel" and name == "driver":
            raise asyncio.CancelledError()

    class Journal(Recorder):
        def __init__(self, output):
            super().__init__(output)
            state.recorder = self

        async def close(self):
            calls.append("journal_close")
            await super().close()

    class Runtime:
        def __init__(self, *args):
            self.turns, self.backchannels, self.aborted_turns, self.history = (
                [],
                [],
                [],
                [],
            )
            self.progress = SimpleNamespace(error=None, manifest=lambda: {})
            self.playback = SimpleNamespace(responses={})

        async def start(self):
            step("runtime_start")

        def begin_live(self, origin):
            self.user_track, self.agent_track = (
                Track(origin, 16000),
                Track(origin, 16000),
            )

        async def close(self):
            step("runtime_close")

        def additional_summary(self):
            return {}

    class Provider:
        def __init__(self, **kwargs):
            pass

        async def disconnect(self):
            step("disconnect")

    class Target:
        def __init__(self, *args):
            self.transcripts = {}

        async def start(self, *args, **kwargs):
            step("target_start")

        def finish_playback(self):
            step("playback")

    class Driver:
        def __init__(self, *args):
            pass

        async def run(self, *args):
            step("driver")
            return "ok"

    def provenance(*args):
        step("provenance")
        return {}

    monkeypatch.setattr(bench, "Recorder", Journal)
    monkeypatch.setattr(bench, "ScenarioRuntime", Runtime)
    monkeypatch.setattr(bench, "LocalStreamingUser", lambda runtime: runtime)
    monkeypatch.setattr(bench, "OpenAIRealtimeProvider", Provider)
    monkeypatch.setattr(bench, "ContinuousOpenAITarget", Target)
    monkeypatch.setattr(bench, "ContinuousVoiceDriver", Driver)
    monkeypatch.setattr(bench, "provenance", provenance)
    monkeypatch.setattr(bench, "load_clip", lambda ident: None)
    monkeypatch.setattr(bench, "save_tracks", lambda *args: step("save_tracks"))
    case = {"id": "P03", "title": "Failure injection", "max_seconds": 1}
    state.run = lambda: bench.run_case(
        SimpleNamespace(api_timeout_seconds=1), case, tmp_path / "trial", "fake-model"
    )
    state.calls = calls
    state.output = tmp_path / "trial"
    return state


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        "target_start",
        "driver",
        "playback",
        "runtime_close",
        "disconnect",
        "save_tracks",
    ],
)
async def test_failure_keeps_summary_and_closes_remaining_resources(trial, failure):
    trial.fail = failure
    summary = await trial.run()
    assert summary["status"] == "error"
    assert summary["error_class"] == "ValueError"
    assert {"runtime_close", "disconnect", "journal_close"} <= set(trial.calls)
    assert not trial.recorder.worker.is_alive()
    assert json.loads((trial.output / "summary.json").read_text())["status"] == "error"


@pytest.mark.asyncio
async def test_provenance_failure_closes_journal_and_records_failure(trial):
    trial.fail = "provenance"
    result = await trial.run()
    assert result["status"] == "error"
    assert not trial.recorder.worker.is_alive()
    assert (trial.output / "summary.json").is_file()


@pytest.mark.asyncio
async def test_cancellation_keeps_partial_evidence_and_propagates(trial):
    trial.fail = "cancel"
    with pytest.raises(asyncio.CancelledError):
        await trial.run()
    assert {"runtime_close", "disconnect", "save_tracks", "journal_close"} <= set(
        trial.calls
    )
    assert not trial.recorder.worker.is_alive()
    summary = json.loads((trial.output / "summary.json").read_text())
    assert summary["status"] == "cancelled"


@pytest.mark.asyncio
async def test_all_cleanup_errors_are_recorded_without_hiding_first_failure(trial):
    trial.failures = {
        "driver",
        "playback",
        "runtime_close",
        "disconnect",
        "save_tracks",
    }
    summary = await trial.run()
    assert summary["status"] == "error"
    assert [error["message"] for error in summary["failures"]] == [
        "driver",
        "playback",
        "runtime_close",
        "disconnect",
        "save_tracks",
    ]
    assert not trial.recorder.worker.is_alive()


@pytest.mark.asyncio
async def test_journal_error_during_observation_drain_still_stops_runtime(tmp_path):
    from tau2.user.local.config import LocalUserConfig
    from tau2.user.local.runtime import UserRuntime
    from tau2.user.local.scenarios import load_scenes

    closed = []

    async def close_asr():
        closed.append(True)

    def emit(*args, **kwargs):
        raise OSError("journal unavailable")

    runtime = UserRuntime(
        LocalUserConfig(asr={"model_path": tmp_path}, backchannel={"enabled": False}),
        load_scenes()[0],
        SimpleNamespace(path=tmp_path, emit=lambda *args, **kwargs: None),
        None,
        asr=SimpleNamespace(close=close_asr),
    )
    runtime.rec.emit = emit

    async def decode():
        _, _, barrier = await runtime.asr_queue.get()
        barrier.set_result(None)
        await asyncio.Future()

    runtime.decode_task = asyncio.create_task(decode())
    runtime.tasks = [runtime.decode_task]
    try:
        with pytest.raises(OSError, match="journal unavailable"):
            await runtime.close()
        assert closed == [True]
        assert runtime.decode_task.done()
    finally:
        runtime.decode_task.cancel()
        await asyncio.gather(runtime.decode_task, return_exceptions=True)
