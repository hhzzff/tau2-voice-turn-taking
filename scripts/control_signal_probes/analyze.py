"""Audit actual event/PCM traces; presence of an opening VAD is not a barge-in."""

import argparse
import audioop
import hashlib
import json
import wave
from collections import Counter
from pathlib import Path


def read_wave(path):
    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 1 and handle.getsampwidth() == 2
        return handle.readframes(handle.getnframes()), handle.getframerate()


def audio_stats(path):
    raw, rate = read_wave(path)
    size = rate * 2 // 10
    run = longest = silent = leading = 0
    first_voice = None
    for i in range(0, len(raw), size):
        frame = raw[i : i + size]
        quiet = audioop.rms(frame, 2) <= 100
        if quiet:
            silent += len(frame) / rate / 2
            run += len(frame) / rate / 2
            longest = max(run, longest)
            if first_voice is None:
                leading += len(frame) / rate / 2
        else:
            if first_voice is None:
                first_voice = i / rate / 2
            run = 0
    return {
        "seconds": len(raw) / rate / 2,
        "rms_le_100_seconds": silent,
        "longest_quiet_seconds": longest,
        "leading_quiet_seconds": leading,
        "first_voice_seconds": first_voice,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def analyze_closed(path):
    report = json.loads(path.read_text())
    trace = path.with_name("scheduler-events.jsonl")
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    trigger = next((r for r in rows if r["kind"] == "probe_interjection"), None)
    controls = [r for r in rows if r["kind"] == "interruption"]
    after = [
        r
        for r in controls
        if trigger
        and r["wall_ms"] >= trigger["wall_ms"]
        and r.get("old_response_id") is not None
    ]
    stale_played = []
    for control in after:
        for row in rows:
            if row["kind"] == "tick" and row["wall_ms"] > control["wall_ms"]:
                for span in row.get("response_spans", []):
                    if span["response_id"] == control["old_response_id"]:
                        stale_played.append({"tick": row["tick_id"], "span": span})
    done_statuses = Counter()
    content_flags = Counter()
    for row in rows:
        event = row.get("event", {})
        if event.get("type") == "response.done":
            done_statuses[(event.get("response") or {}).get("status", "missing")] += 1
        content = event.get("serverContent") or event.get("server_content") or {}
        for flag in (
            "interrupted",
            "turnComplete",
            "turn_complete",
            "generationComplete",
            "generation_complete",
        ):
            if content.get(flag):
                content_flags[flag] += 1
    return {
        "artifact": str(path.parent),
        "model": report["model"],
        "model_id": report["model_id"],
        "scenario": report["scenario"],
        "execution_status": report["status"],
        "execution_error": report["error"],
        "endpoint": report["endpoint"],
        "input_seconds": report["input_seconds"],
        "trigger": trigger,
        "opening_controls_not_counted_as_barge_in": len(controls) - len(after),
        "post_injection_controls": after,
        "cancelled_pending_bytes": sum(r.get("discarded_bytes", 0) for r in after),
        "old_audio_played_after_cancel": stale_played,
        "response_done_statuses": dict(done_statuses),
        "gemini_content_flags": dict(content_flags),
        "raw_audio": audio_stats(path.with_name("response-native.wav")),
        "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
    }


def analyze_native(path):
    report = json.loads(path.read_text())
    trace = path.with_name("native-events.jsonl")
    rows = (
        [json.loads(x) for x in trace.read_text().splitlines()]
        if trace.exists()
        else []
    )
    flags = Counter()
    controls = []
    states = Counter()
    for row in rows:
        event = row["native"]
        if "is_listen" in event:
            flags[
                f"is_listen={event['is_listen']},end_of_turn={event['end_of_turn']}"
            ] += 1
        if event.get("control", {}).get("source") == "external_policy":
            controls.append({"tick": row["tick"], "control": event["control"]})
        for item in event.get("events", []):
            name = item.get("type", item.get("event", "unknown"))
            states[name + ("=" + str(item["stat"]) if "stat" in item else "")] += 1
    result = {
        "artifact": str(path.parent),
        "model": report["model"],
        "scenario": report["scenario"],
        "execution_status": report["status"],
        "execution_error": report.get("error"),
        "trigger": report.get("trigger"),
        "native_flags": dict(flags),
        "native_states": dict(states),
        "externally_forced_steps": controls,
        "cancelled_pending_bytes": sum(
            x.get("cancelled_pending_bytes", 0) for x in rows
        ),
        "input_ticks": len(rows),
        "nonempty_output_ticks": sum(x["native_output_samples"] > 0 for x in rows),
        "output_samples": sum(x["native_output_samples"] for x in rows),
        "provenance": report.get("provenance"),
        "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest()
        if trace.exists()
        else None,
    }
    audio = path.with_name("output-native.wav")
    if audio.exists():
        result["raw_audio"] = audio_stats(audio)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for path in sorted(args.root.glob("20261112-*/*/*/probe.json")):
        results.append(
            analyze_closed(path)
            if path.with_name("scheduler-events.jsonl").exists()
            else analyze_native(path)
        )
    args.output.write_text(
        json.dumps(
            {
                "scope": "actual fixed synthetic stimulus probes, not population-level capability proof",
                "runs": results,
            },
            indent=2,
        )
        + "\n"
    )
    for row in results:
        print(
            row["model"],
            row.get("model_id", "native"),
            row["scenario"],
            row["execution_status"],
            "post_controls",
            len(row.get("post_injection_controls", [])),
            "cancel_bytes",
            row["cancelled_pending_bytes"],
        )


if __name__ == "__main__":
    main()
