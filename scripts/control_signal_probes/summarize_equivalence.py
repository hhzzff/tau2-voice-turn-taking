"""Summarize preserved A/B evidence without making new model requests."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def read_rows(path):
    """Read an optional event stream."""
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def summarize(root):
    """Retain per-run timings, input identity, partial text, and paired differences."""
    runs = {}
    for path in sorted(root.glob("*/*/result.json")):
        result = json.loads(path.read_text())
        directory = path.parent
        trace = read_rows(directory / "scheduler-events.jsonl")
        native = read_rows(directory / "native.jsonl")
        sends = [r for r in trace if r["kind"] == "send"]
        provider = [r["event"] for r in trace if r["kind"] == "provider_event"]
        text = {}
        for event in provider:
            typ = event.get("type", "")
            identity = event.get("response_id", "unknown")
            if typ in {
                "response.output_audio_transcript.delta",
                "response.audio_transcript.delta",
            }:
                text[identity] = text.get(identity, "") + event.get("delta", "")
            elif typ in {
                "response.output_audio_transcript.done",
                "response.audio_transcript.done",
            }:
                text[identity] = event.get("transcript", text.get(identity, ""))
        events = json.loads((directory / "events.json").read_text())
        cancellations = [
            e for e in events if e["kind"] == "cancel" and e["at_ms"] > 10000
        ]
        onset = result["first_voice_ms"]
        reply = result["post_interruption_voice_ms"]
        origin = result.get("input_origin_wall_ms", 0)
        row = {
            "model": result["model"],
            "mode": result["mode"],
            "first_voice_ms": onset,
            "first_response_latency_ms": None if onset is None else onset - 3412.6875,
            "post_interruption_voice_ms": reply,
            "reply_latency_ms": None
            if reply is None
            else reply - result["interruption_end_ms"],
            "input_sha256": result["input_sha256"],
            "text": text or result["text"],
            "provider_event_counts": dict(
                Counter(e.get("type", "gemini") for e in provider)
            ),
            "cancel_arrival_ms": [e["at_ms"] for e in cancellations],
            "cancel_delivery_ms": [
                math.ceil(e["at_ms"] / 100) * 100
                if result["mode"] == "scheduled"
                else e["at_ms"]
                for e in cancellations
            ],
            "send_packet_count": len(sends),
            "send_sequence_sha256": hashlib.sha256(
                "".join(r["pcm_sha256"] for r in sends).encode()
            ).hexdigest()
            if sends
            else None,
            "max_send_lateness_ms": max(
                (r["wall_ms"] - origin - r["input_end_ms"] for r in sends), default=None
            ),
            "last_input_end_ms": sends[-1]["input_end_ms"] if sends else None,
            "same_trace_first_voice_quantization_ms": onset
            - result["arrival_speech_regions_ms"][0][0]
            if onset is not None
            else None,
            "native_steps": native,
        }
        runs[str(directory.relative_to(root))] = row
    pairs = {}
    for family in sorted({key.split("/")[0] for key in runs}):
        for left, right in [
            ("0-direct", "1-scheduled"),
            ("3-direct", "2-scheduled"),
            ("0-direct", "3-direct"),
        ]:
            a, b = runs.get(f"{family}/{left}"), runs.get(f"{family}/{right}")
            if not a or not b:
                continue
            pair = {"same_input": a["input_sha256"] == b["input_sha256"]}
            for metric in ("first_voice_ms", "post_interruption_voice_ms"):
                pair[metric + "_difference"] = (
                    None
                    if a[metric] is None or b[metric] is None
                    else b[metric] - a[metric]
                )
            if a["native_steps"] and b["native_steps"]:
                x, y = a["native_steps"], b["native_steps"]
                pair["native_step_counts"] = [len(x), len(y)]
                for key in [
                    "input_sha256",
                    "audio_sha256",
                    "text",
                    "is_listen",
                    "force_listen",
                    "end_of_turn",
                ]:
                    pair[key + "_equal_steps"] = sum(
                        p[key] == q[key] for p, q in zip(x, y)
                    )
                pair["max_abs_audio_ready_difference_ms"] = max(
                    abs(p["audio_ready_ms"] - q["audio_ready_ms"]) for p, q in zip(x, y)
                )
            pairs[f"{family}/{left} -> {right}"] = pair
    output = {
        "scope": "prototype; not production FullDuplexOrchestrator",
        "runs": runs,
        "pairs": pairs,
    }
    (root / "analysis.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"root": str(root), "pairs": pairs}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    for root in parser.parse_args().roots:
        summarize(root)
