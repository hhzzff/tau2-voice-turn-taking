"""SALMONN's existing public adapter seam against an external runtime double."""

import base64

from tau2.data_model.simulation import OpenModelConfig
from tau2.voice.audio_native.open_model.adapter import OpenModelAdapter


class Runtime:
    """Model boundary with one frontend startup tick and known audible output."""

    def request(self, op, **payload):
        if op == "connect":
            self.tick = 0
            return {"checkpoint_variant": "before_sft", "sample_rate": 16000}
        if op == "disconnect":
            return {}
        assert len(base64.b64decode(payload["audio_pcm16_b64"])) == 2560
        self.tick += 1
        return {
            "audio_pcm16_b64": ""
            if self.tick == 1
            else base64.b64encode(b"\x00\x20" * 1280).decode(),
            "sample_rate": 16000,
            "text": "",
            "events": [{"tick": self.tick, "input_samples_total": self.tick * 1280}],
        }

    def close(self):
        pass


def test_salmonn_startup_does_not_consume_future_tick_and_reconnect_is_fresh():
    adapter = OpenModelAdapter(
        OpenModelConfig(model="salmonn"), runtime_config={}, transport=Runtime()
    )
    for _ in range(2):
        adapter.connect("Listen naturally.", [], None)
        first = adapter.run_tick(b"\xff" * 640)
        assert not first.agent_audio_data
        assert len(first.get_played_agent_audio()) == 640
        assert first.events[-1]["input_samples_total"] == 1280
        second = adapter.run_tick(b"\xff" * 640)
        assert len(second.agent_audio_data) == 640
        assert second.events[-1]["input_samples_total"] == 2560
        adapter.disconnect()
