"""Observable playback clock: never backdate model output or retain cancelled tail."""

import importlib.util
from pathlib import Path


def load_probe():
    path = (
        Path(__file__).resolve().parents[4]
        / "scripts/control_signal_probes/equivalence.py"
    )
    spec = importlib.util.spec_from_file_location("equivalence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_completion_at_1401ms_is_released_at_1500ms_not_input_start():
    probe = load_probe()
    pcm = b"\x00\x10" * 16000
    events = [
        {
            "kind": "audio",
            "at_ms": 1401,
            "response_id": "a",
            "raw_offset_bytes": 0,
            "audio_bytes": len(pcm),
        }
    ]
    direct, direct_spans = probe.render(
        events, pcm, rate=16000, horizon_ms=3000, quantum_ms=0
    )
    scheduled, spans = probe.render(
        events, pcm, rate=16000, horizon_ms=3000, quantum_ms=100
    )
    assert direct_spans[0]["start_ms"] == 1401
    assert spans[0]["start_ms"] == 1500
    assert direct[1401 * 32 : 2401 * 32] == pcm
    assert scheduled[: 1500 * 32] == bytes(1500 * 32)
    assert scheduled[1500 * 32 : 2500 * 32] == pcm


def test_cancel_removes_old_tail_and_late_packets_without_discarding_new_reply():
    probe = load_probe()
    old = b"\x00\x10" * 16000
    new = b"\x00\x20" * 1600
    events = [
        {
            "kind": "audio",
            "at_ms": 1000,
            "response_id": "old",
            "raw_offset_bytes": 0,
            "audio_bytes": len(old),
        },
        {"kind": "cancel", "at_ms": 1240, "response_id": "old"},
        {
            "kind": "audio",
            "at_ms": 1250,
            "response_id": "old",
            "raw_offset_bytes": 0,
            "audio_bytes": len(old),
        },
        {
            "kind": "audio",
            "at_ms": 1310,
            "response_id": "new",
            "raw_offset_bytes": len(old),
            "audio_bytes": len(new),
        },
    ]
    played, spans = probe.render(
        events, old + new, rate=16000, horizon_ms=2000, quantum_ms=100
    )
    assert played[1000 * 32 : 1300 * 32] == old[: 300 * 32]
    assert played[1300 * 32 : 1400 * 32] == bytes(100 * 32)
    assert played[1400 * 32 : 1500 * 32] == new
    assert [s["response_id"] for s in spans] == ["old", "new"]
