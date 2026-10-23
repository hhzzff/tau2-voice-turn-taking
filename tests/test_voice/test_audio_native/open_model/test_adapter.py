"""Public adapter behavior using an external runtime boundary double."""

import base64

from tau2.data_model.simulation import OpenModelConfig
from tau2.voice.audio_native.open_model.adapter import OpenModelAdapter


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
        OpenModelConfig(model="minicpm"), runtime_config={}, transport=Runtime()
    )
    adapter.connect("hello", [], None)
    first = adapter.run_tick(b"\xff" * 8000)
    second = adapter.run_tick(b"\xff" * 8000)
    assert len(first.get_played_agent_audio()) == 8000
    assert len(second.agent_audio_data) == 8000
    assert second.events[-1]["input_count"] == 2
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
        OpenModelConfig(model="minicpm"),
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
 def step(self, pcm16):
  self.count+=1
  return {'audio_pcm16_b64':base64.b64encode(b'\\x00\\x10'*24000).decode(),'sample_rate':24000,'text':'hello','events':[{'count':self.count}]}
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
        OpenModelConfig(model="minicpm"), runtime_config={}, transport=transport
    )
    try:
        for _ in range(2):
            adapter.connect("new session", [], None)
            result = adapter.run_tick(b"\xff" * 8000)
            assert result.events[-1] == {"count": 1}
            assert len(result.agent_audio_data) == 8000
            adapter.disconnect()
    finally:
        (mailbox / "STOP").touch()
        server.wait(timeout=5)


def test_registered_factory_defaults_to_native_tick_without_loading_weights():
    from tau2.registry import registry

    factory = registry.get_agent_factory("open_model_agent")
    agent = factory(
        [], "Have a natural conversation.", runtime_config={}, transport=Runtime()
    )
    assert agent.tick_duration_ms == 1000
    assert not agent.adapter.is_connected
