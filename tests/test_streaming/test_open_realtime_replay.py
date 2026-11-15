"""Realtime native replay must advance while inference is pending."""

import base64
import importlib.util
import threading
import time
from pathlib import Path
from types import SimpleNamespace


def test_tick_returns_before_native_generation_finishes():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/control_signal_probes/realtime_adapter.py"
    )
    spec = importlib.util.spec_from_file_location("realtime_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    release = threading.Event()
    started = threading.Event()
    inputs = []

    class Runtime:
        def request(self, op, **payload):
            if op == "connect":
                return {"sample_rate": 16000}
            if op == "step":
                inputs.append(base64.b64decode(payload["audio_pcm16_b64"]))
                started.set()
                assert release.wait(5)
                return {
                    "sample_rate": 16000,
                    "audio_pcm16_b64": base64.b64encode(b"\x00\x10" * 2560).decode(),
                    "events": [],
                    "text": "",
                }
            return {}

        def close(self):
            pass

    adapter = module.RealtimeOpenModelAdapter(
        SimpleNamespace(model="freeze_omni", tick_duration_ms=100, pacing="realtime"),
        {},
        Runtime(),
        native_width_ms=160,
    )
    adapter.connect("", [], {})
    try:
        adapter.run_tick(b"\x80" * 800)
        assert not started.is_set()  # Only 100ms, below native 160ms.
        begin = time.monotonic()
        result = adapter.run_tick(b"\x90" * 800)
        assert started.is_set()
        assert time.monotonic() - begin < 0.5
        assert result.agent_audio_bytes == 0
        assert len(inputs) == 1 and len(inputs[0]) == 5120
        release.set()
        result = adapter.run_tick(b"\xff" * 800)
        assert result.agent_audio_bytes > 0
    finally:
        release.set()
        adapter.disconnect()


def test_realtime_replay_uses_production_step(monkeypatch):
    directory = Path(__file__).resolve().parents[2] / "scripts/control_signal_probes"
    monkeypatch.syspath_prepend(str(directory))
    from open_equivalence import build_scheduler

    requests = []

    class Runtime:
        def request(self, op, **payload):
            if op == "step":
                requests.append(payload)
            return {"sample_rate": 16000, "audio_pcm16_b64": "", "events": []}

        def close(self):
            pass

    scheduler, agent = build_scheduler(
        "freeze_omni", b"\xff" * 1600, {}, Runtime(), realtime=True
    )
    try:
        scheduler.initialize()
        scheduler.step()
        scheduler.step()
        assert len(requests) == 1
        assert len(scheduler.get_ticks()) == 3
        assert (
            len(base64.b64decode(scheduler.get_ticks()[-1].agent_chunk.audio_content))
            == 800
        )
    finally:
        agent.stop()


def test_tick_deadlines_do_not_accumulate_caller_overhead():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/control_signal_probes/realtime_adapter.py"
    )
    spec = importlib.util.spec_from_file_location("realtime_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Runtime:
        def request(self, op, **payload):
            return {"sample_rate": 16000, "audio_pcm16_b64": "", "events": []}

        def close(self):
            pass

    adapter = module.RealtimeOpenModelAdapter(
        SimpleNamespace(model="freeze_omni", tick_duration_ms=100, pacing="realtime"),
        {},
        Runtime(),
        native_width_ms=160,
    )
    adapter.connect("", [], {})
    try:
        start = time.monotonic()
        for index in range(5):
            if index:
                time.sleep(0.08)  # Caller bookkeeping between ticks.
            adapter.run_tick(b"\xff" * 800)
        assert time.monotonic() - start < 0.65
    finally:
        adapter.disconnect()


def test_native_window_keeps_advancing_while_inference_is_blocked(monkeypatch):
    directory = Path(__file__).resolve().parents[2] / "scripts/control_signal_probes"
    monkeypatch.syspath_prepend(str(directory))
    from open_equivalence import build_scheduler

    release = threading.Event()
    entered = threading.Event()

    class Runtime:
        def request(self, op, **payload):
            if op == "step":
                assert len(base64.b64decode(payload["audio_pcm16_b64"])) == 5120
                entered.set()
                assert release.wait(5)
            return {"sample_rate": 16000, "audio_pcm16_b64": "", "events": []}

        def close(self):
            pass

    scheduler, agent = build_scheduler(
        "freeze_omni",
        b"\xff" * 3840,
        {},
        Runtime(),
        realtime=True,
        window_ms=160,
    )
    try:
        scheduler.initialize()
        start = time.monotonic()
        scheduler.step()
        scheduler.step()
        assert entered.is_set()
        assert time.monotonic() - start < 0.7
        assert (
            len(base64.b64decode(scheduler.get_ticks()[-1].agent_chunk.audio_content))
            == 1280
        )
    finally:
        release.set()
        agent.stop()


def test_reference_stop_tts_cancels_queued_audio_but_natural_end_drains(monkeypatch):
    directory = Path(__file__).resolve().parents[2] / "scripts/control_signal_probes"
    monkeypatch.syspath_prepend(str(directory))
    from realtime_adapter import RealtimeOpenModelAdapter

    for signal, expects_audio in [
        ("controller_stop_tts", False),
        ("native_generate_stat", True),
    ]:

        class Runtime:
            steps = 0

            def request(self, op, **payload):
                event = {"sample_rate": 16000, "audio_pcm16_b64": "", "events": []}
                if op == "step":
                    self.steps += 1
                    if self.steps == 1:
                        event["audio_pcm16_b64"] = base64.b64encode(
                            b"\x00\x10" * 12800
                        ).decode()
                    elif self.steps == 2:
                        event["events"] = [{"type": signal, "stat": "sl"}]
                return event

            def close(self):
                pass

        adapter = RealtimeOpenModelAdapter(
            SimpleNamespace(
                model="freeze_omni", tick_duration_ms=100, pacing="realtime"
            ),
            {},
            Runtime(),
            native_width_ms=160,
        )
        adapter.connect("", [], {})
        try:
            for _ in range(4):
                result = adapter.run_tick(b"\xff" * 800)
            assert (result.agent_audio_bytes > 0) is expects_audio
        finally:
            adapter.disconnect()


def test_registered_open_models_default_to_100ms_realtime():
    from tau2.agent.open_model_agent import create_open_model_agent
    from tau2.data_model.simulation import OpenModelConfig
    from tau2.voice.audio_native.open_model.realtime_adapter import (
        RealtimeOpenModelAdapter,
    )

    for model, native_width in [("salmonn", 80), ("minicpm", 1000)]:
        config = OpenModelConfig(model=model)
        assert config.tick_duration_ms == 100
        assert config.pacing == "realtime"
        agent = create_open_model_agent(
            [], "", model=model, runtime_config={}, transport=object()
        )
        assert isinstance(agent.adapter, RealtimeOpenModelAdapter)
        assert agent.adapter.pacing == "realtime"
        assert agent.adapter.native_width_ms == native_width
        assert agent.adapter.bytes_per_tick == 800


def test_reconnect_cannot_reuse_transport_with_old_inference_in_flight(monkeypatch):
    import pytest

    from tau2.data_model.simulation import OpenModelConfig
    from tau2.voice.audio_native.open_model import realtime_adapter

    monkeypatch.setattr(realtime_adapter, "_SHUTDOWN_WAIT_SECONDS", 0.05, raising=False)
    started = threading.Event()
    release = threading.Event()
    returned = threading.Event()

    class Runtime:
        connects = 0

        def request(self, op, **payload):
            if op == "connect":
                self.connects += 1
                return {"sample_rate": 16000}
            if op == "step":
                label = "old" if self.connects == 1 else "new"
                if label == "old":
                    started.set()
                    assert release.wait(25)
                    returned.set()
                return {
                    "sample_rate": 16000,
                    "audio_pcm16_b64": "",
                    "events": [{"session_label": label}],
                }
            return {}

        def close(self):
            pass  # A runtime that cannot immediately cancel the old request.

    runtime = Runtime()
    adapter = realtime_adapter.RealtimeOpenModelAdapter(
        OpenModelConfig(model="salmonn"), {}, runtime, native_width_ms=80
    )
    adapter.connect("old", [], {})
    try:
        adapter.run_tick(b"\xff" * 800)
        assert started.wait(2)
        adapter.disconnect()
        with pytest.raises(RuntimeError, match="previous session"):
            adapter.connect("new", [], {})
        assert runtime.connects == 1

        release.set()
        assert returned.wait(2)
        deadline = time.monotonic() + 2
        while True:
            try:
                adapter.connect("new", [], {})
                break
            except RuntimeError as exc:
                assert "previous session" in str(exc)
                assert time.monotonic() < deadline
                time.sleep(0.01)
        labels = []
        for _ in range(3):
            result = adapter.run_tick(b"\xff" * 800)
            labels.extend(
                e["session_label"] for e in result.events if "session_label" in e
            )
        assert labels and set(labels) == {"new"}
    finally:
        release.set()
        adapter.disconnect()


def test_registered_minicpm_does_not_forward_forced_listening():
    import pytest

    from tau2.registry import registry

    requests = []
    submitted = threading.Event()

    class Runtime:
        def request(self, op, **payload):
            if op == "connect":
                return {"sample_rate": 24000}
            if op == "step":
                requests.append(payload)
                submitted.set()
                return {
                    "sample_rate": 24000,
                    "audio_pcm16_b64": "",
                    "is_listen": True,
                }
            return {}

        def close(self):
            pass

    factory = registry.get_agent_factory("open_model_agent")
    adapter = factory(
        [], "", model="minicpm", runtime_config={}, transport=Runtime()
    ).adapter
    adapter.connect("", [], {})
    try:
        audio = b"\xff" * adapter.bytes_per_tick
        for controls in ({"interrupt": True}, {"force_listen": True}):
            with pytest.raises(ValueError, match="native autonomous control only"):
                adapter.run_tick(audio, **controls)
        assert not requests

        # Ten accepted 100 ms windows complete the native 1000 ms input block.
        for _ in range(10):
            adapter.run_tick(audio)
        assert submitted.wait(2)
        assert len(requests) == 1
        assert set(requests[0]) == {"audio_pcm16_b64"}
        assert len(base64.b64decode(requests[0]["audio_pcm16_b64"])) == 32000
    finally:
        adapter.disconnect()
