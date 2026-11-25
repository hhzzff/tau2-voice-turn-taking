"""The report must not substitute provider text for unfinished local ASR."""

from experiments.tau_voice.local_user_report import heard_utterances


def test_unfinished_asr_remains_partial_in_listening_report():
    events = [
        {"kind": "target_first_play", "item_id": "a", "mono_ns": 1_000_000_000},
        {
            "kind": "heard_partial",
            "item_id": "a",
            "mono_ns": 2_000_000_000,
            "text": "实际已听到的开头",
        },
    ]
    rows = heard_utterances(events, {"a": "包含尚未播放的服务端完整回复"}, 0)
    assert rows[0]["text"] == "实际已听到的开头"
    assert rows[0]["asr_finalized"] is False
    assert rows[0]["at"] == 1 and rows[0]["text_available_at"] == 2
    events.append(
        {
            "kind": "heard_agent_complete",
            "item_id": "a",
            "mono_ns": 3_000_000_000,
            "text": "完整实际听到的回复",
        }
    )
    row = heard_utterances(events, {}, 0)[0]
    assert row["asr_finalized"] and row["text"] == "完整实际听到的回复"
