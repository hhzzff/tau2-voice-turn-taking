"""Public adapter behavior using an external runtime boundary double."""

import base64
from types import SimpleNamespace

from tau2.voice.audio_native.open_model.adapter import OpenModelAdapter


def _native_step_config():
    """Drive one synchronous native request per tick for playback unit tests."""
    return SimpleNamespace(
        pacing="frozen",
        model="minicpm",
        tick_duration_ms=1000,
        model_identity="MiniCPM-o-4.5",
    )


class Runtime:
    def request(self, op, **payload):
        if op == "connect":
            self.count = 0
            return {"checkpoint": "test-only", "sample_rate": 24000}
        if op == "step":
            assert len(base64.b64decode(payload["audio_pcm16_b64"])) == 32000
            self.count += 1
            return {
                "audio_pcm16_b64": base64.b64encode(
                    b"\x00\x10" * 48000 if self.count == 1 else b""
                ).decode(),
                "sample_rate": 24000,
                "text": "hello",
                "events": [{"input_count": self.count}],
            }
        return {}

    def close(self):
        pass


def test_buffered_output_still_consumes_each_input_and_reconnect_resets():
    adapter = OpenModelAdapter(
        _native_step_config(),
        runtime_config={},
        transport=Runtime(),
    )
    adapter.connect("hello", [], None)
    first = adapter.run_tick(b"\xff" * 8000)
    second = adapter.run_tick(b"\xff" * 8000)
    assert len(first.get_played_agent_audio()) == 8000
    assert len(second.agent_audio_data) == 8000
    assert second.events[-1]["input_count"] == 2
    assert second.model_dump()["serializable_events"][-1] == {"input_count": 2}
    adapter.disconnect()
    adapter.connect("fresh", [], None)
    assert adapter.run_tick(b"\xff" * 8000).events[-1]["input_count"] == 1
    adapter.disconnect()


def test_real_subprocess_protocol_and_failure_are_visible(tmp_path):
    import sys

    import pytest

    from tau2.voice.audio_native.open_model.transport import SubprocessTransport

    worker = tmp_path / "runtime.py"
    worker.write_text("""import sys,json
for line in sys.stdin:
 r=json.loads(line)
 if r['op']=='connect': out={'result': {'sample_rate':24000}}
 elif r['op']=='disconnect': out={'result':{}}
 else: out={'error':'inference failed explicitly'}
 print(json.dumps(dict(out,session_id=r['session_id'],sequence=r['sequence'])),flush=True)
""")
    adapter = OpenModelAdapter(
        _native_step_config(),
        runtime_config={},
        transport=SubprocessTransport(
            [sys.executable, str(worker)], timeout_seconds=10
        ),
    )
    adapter.connect("hello", [], None)
    with pytest.raises(RuntimeError, match="inference failed explicitly"):
        adapter.run_tick(b"\xff" * 8000)
    adapter.disconnect()
    assert not adapter.is_connected


def test_mailbox_reconnect_isolates_sessions_and_keeps_tick_audio(tmp_path):
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    from tau2.voice.audio_native.open_model.transport import MailboxTransport

    source = (
        Path(__file__).resolve().parents[4] / "scripts/open_model_workers/worker.py"
    )
    shutil.copy(source, tmp_path / "worker.py")
    (tmp_path / "minicpm_worker.py").write_text("""import base64
class MiniCPMWorker:
 def connect(self, config, system_prompt):
  self.count=0
  return {'sample_rate':24000}
 def step(self, pcm16, **controls):
  self.count+=1
  return {'audio_pcm16_b64':base64.b64encode(b'\\x00\\x10'*24000).decode(),'sample_rate':24000,'text':'hello','events':[{'count':self.count, **({'controls':controls} if controls else {})}]}
 def disconnect(self): pass
""")
    mailbox = tmp_path / "mailbox"
    server = subprocess.Popen(
        [
            sys.executable,
            str(tmp_path / "worker.py"),
            "--model",
            "minicpm",
            "--mailbox",
            str(mailbox),
            "--idle-timeout",
            "10",
        ]
    )
    transport = MailboxTransport(mailbox, timeout_seconds=5)
    adapter = OpenModelAdapter(
        _native_step_config(),
        runtime_config={},
        transport=transport,
    )
    try:
        for _ in range(2):
            adapter.connect("new session", [], None)
            result = adapter.run_tick(b"\xff" * 8000)
            assert result.events[-1] == {"count": 1}
            assert len(result.agent_audio_data) == 8000
            interrupted = adapter.run_tick(b"\xff" * 8000, interrupt=True)
            assert interrupted.events[-1] == {
                "count": 2,
                "controls": {"interrupt": True},
            }
            held = adapter.run_tick(b"\xff" * 8000, force_listen=True)
            assert held.events[-1] == {"count": 3, "controls": {"force_listen": True}}
            adapter.disconnect()
    finally:
        (mailbox / "STOP").touch()
        server.wait(timeout=5)


def test_registered_factory_defaults_to_100ms_without_loading_weights():
    from tau2.registry import registry

    factory = registry.get_agent_factory("open_model_agent")
    agent = factory(
        [], "Have a natural conversation.", runtime_config={}, transport=Runtime()
    )
    assert agent.tick_duration_ms == 100
    assert agent.adapter.native_width_ms == 1000
    assert not agent.adapter.is_connected


def test_native_listen_state_is_preserved_alongside_delivered_audio():
    class ListeningRuntime(Runtime):
        def request(self, op, **payload):
            result = super().request(op, **payload)
            if op == "step":
                result["is_listen"] = self.count == 2
                result["end_of_turn"] = False
                if self.count == 1:
                    result["audio_pcm16_b64"] = base64.b64encode(
                        b"\x00\x10" * 27840
                    ).decode()
                elif self.count == 2:
                    result["audio_pcm16_b64"] = base64.b64encode(
                        b"\x00\x00" * 24000
                    ).decode()
                if self.count == 3:
                    result["audio_pcm16_b64"] = base64.b64encode(
                        b"\x00\x20" * 24000
                    ).decode()
            return result

    adapter = OpenModelAdapter(
        _native_step_config(),
        runtime_config={},
        transport=ListeningRuntime(),
    )
    adapter.connect("Have a conversation.", [], None)
    try:
        first = adapter.run_tick(b"\xff" * 8000)
        listening = adapter.run_tick(b"\x00" * 8000)
        resumed = adapter.run_tick(b"\xff" * 8000)
        assert any(value != 0xFF for value in first.get_played_agent_audio())
        state = next(
            e for e in listening.serializable_events if e.get("type") == "native_state"
        )
        assert state["is_listen"] is True
        assert state["end_of_turn"] is False
        assert "interrupted" not in listening.vad_events
        assert any(value != 0xFF for value in resumed.get_played_agent_audio())
        assert listening.events[-1]["input_count"] == 2
    finally:
        adapter.disconnect()


def test_default_salmonn_runtime_uses_selected_checkpoint_and_training_audio_contract():
    from tau2.voice.audio_native.open_model.adapter import load_runtime_config

    config = load_runtime_config("salmonn")
    selected = "/mnt/bn/audio-visual-llm-data2/yuwenyi/playground/exp_results/cu12_stage2-0519_fc-180_aes_er0_chunk30_llama31/checkpoint-44000"
    assert config["checkpoint"] == selected
    assert (
        config["assets"]["training_config"]
        == selected.rsplit("/", 1)[0] + "/config.json"
    )
    assert config["settings"]["echo_ratio"] == 0.0


def test_native_listen_cancels_unplayed_tail_without_claiming_user_interruption():
    class ListeningRuntime(Runtime):
        def request(self, op, **payload):
            result = super().request(op, **payload)
            if op == "step":
                result.update(is_listen=self.count >= 2, end_of_turn=False)
                result["audio_pcm16_b64"] = base64.b64encode(
                    b"\x00\x10" * 27840 if self.count == 1 else b""
                ).decode()
            return result

    adapter = OpenModelAdapter(
        _native_step_config(),
        runtime_config={},
        transport=ListeningRuntime(),
    )
    adapter.connect("test", [], None)
    try:
        adapter.run_tick(b"\xff" * 8000)
        listening = adapter.run_tick(b"\xff" * 8000)
        assert listening.agent_audio_data == b""
        assert listening.truncated_audio_bytes == 1280
        assert "interrupted" not in listening.vad_events
        assert adapter.run_tick(b"\xff" * 8000).agent_audio_data == b""
    finally:
        adapter.disconnect()


def test_natural_turn_end_drains_tail_even_when_next_tick_reports_listening():
    class FinishedRuntime(Runtime):
        def request(self, op, **payload):
            result = super().request(op, **payload)
            if op == "step":
                result.update(is_listen=self.count >= 2, end_of_turn=self.count == 1)
                result["audio_pcm16_b64"] = base64.b64encode(
                    b"\x00\x10" * 51840 if self.count == 1 else b""
                ).decode()
            return result

    adapter = OpenModelAdapter(
        _native_step_config(),
        runtime_config={},
        transport=FinishedRuntime(),
    )
    adapter.connect("test", [], None)
    try:
        first = adapter.run_tick(b"\xff" * 8000)
        second = adapter.run_tick(b"\xff" * 8000)
        third = adapter.run_tick(b"\xff" * 8000)
        assert len(first.agent_audio_data) == len(second.agent_audio_data) == 8000
        assert len(third.agent_audio_data) == 1280
        assert not second.was_truncated
        assert not third.was_truncated
    finally:
        adapter.disconnect()


def test_explicit_interrupt_is_forwarded_and_keeps_replacement_response_tail():
    class InterruptedRuntime(Runtime):
        def request(self, op, **payload):
            result = super().request(op, **payload)
            if op == "step":
                result.update(is_listen=False, end_of_turn=self.count == 3)
                if self.count == 2:
                    assert payload["interrupt"] is True
                else:
                    assert "interrupt" not in payload
                result["audio_pcm16_b64"] = base64.b64encode(
                    b"\x00\x10" * 27840 if self.count <= 2 else b""
                ).decode()
            return result

    adapter = OpenModelAdapter(
        _native_step_config(),
        runtime_config={},
        transport=InterruptedRuntime(),
    )
    adapter.connect("test", [], None)
    try:
        adapter.run_tick(b"\xff" * 8000)
        interrupted = adapter.run_tick(b"\xff" * 8000, interrupt=True)
        tail = adapter.run_tick(b"\xff" * 8000)
        assert interrupted.truncated_audio_bytes == 1280
        assert interrupted.vad_events == ["interrupted"]
        assert len(interrupted.agent_audio_data) == 8000
        assert len(tail.agent_audio_data) == 1280
        control = next(
            e
            for e in interrupted.serializable_events
            if e["type"] == "playback_control"
        )
        assert control["source"] == "external_policy"
        assert control["signal"] == "external_interrupt"
    finally:
        adapter.disconnect()


def test_mailbox_close_unblocks_pending_request_without_waiting_for_response(tmp_path):
    import time
    from concurrent.futures import ThreadPoolExecutor

    import pytest

    from tau2.voice.audio_native.open_model.transport import MailboxTransport

    transport = MailboxTransport(tmp_path, timeout_seconds=3)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(transport.request, "step")
        deadline = time.monotonic() + 1
        while not list(tmp_path.glob("*/*.request.json")):
            assert time.monotonic() < deadline, "request was not published"
            time.sleep(0.01)
        transport.close()
        with pytest.raises(RuntimeError, match="is closed"):
            pending.result(timeout=1)
