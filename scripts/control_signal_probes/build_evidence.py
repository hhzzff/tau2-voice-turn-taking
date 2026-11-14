"""Build an auditable admission ledger from actual completed native/API probes."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from analyze import analyze_closed, analyze_native

PROFILES = {
    "gpt_live": "openai_live",
    "gpt_realtime": "openai",
    "gemini_live": "gemini",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    signals = {}
    continuous = {}
    cases = []
    jobs = []

    def add(profile, signal, case, example, count=1):
        entry = signals.setdefault(profile, {}).setdefault(signal, {"witnesses": []})
        entry["witnesses"].append(
            {
                "artifact": case["artifact"],
                "scenario": case["scenario"],
                "model_id": case.get("model_id"),
                "checkpoint": (case.get("provenance") or {}).get("checkpoint"),
                "trace_sha256": case["trace_sha256"],
                "matched_occurrences": count,
                "first_example": example,
            }
        )

    for path in sorted(args.root.glob("20261112-*/*/*/probe.json")):
        is_closed = path.with_name("scheduler-events.jsonl").exists()
        case = analyze_closed(path) if is_closed else analyze_native(path)
        case["run_root"] = str(path.parents[2])
        snapshot = path.parents[2] / "source/native.py"
        if snapshot.exists():
            case["native_probe_source_sha256"] = digest(snapshot)
            source = path.parents[2] / "source/workers/salmonn_session.py"
            if source.exists():
                case["salmonn_worker_source_sha256"] = digest(source)
        cases.append(case)
        if case["execution_status"] != "observed":
            continue
        profile = PROFILES.get(case["model"], case["model"])
        if is_closed:
            trace = [
                json.loads(x)
                for x in path.with_name("scheduler-events.jsonl")
                .read_text()
                .splitlines()
            ]
            if case["old_audio_played_after_cancel"]:
                raise AssertionError(
                    "Stale response audio played after cancellation: " + str(path)
                )
            if profile in {"openai", "doubao_s2s"}:
                name = (
                    "input_audio_buffer.speech_started"
                    if profile == "openai"
                    else "conversation.item.input_audio_transcription.started"
                )
                matches = [
                    r
                    for r in case["post_injection_controls"]
                    if r.get("event_type") == name
                ]
                if matches:
                    add(profile, name, case, matches[0], len(matches))
            if profile == "openai":
                done = [
                    r["event"]
                    for r in trace
                    if r.get("event", {}).get("type") == "response.done"
                ]
                done = [
                    {
                        "type": e["type"],
                        "response_id": (e.get("response") or {}).get("id"),
                        "status": (e.get("response") or {}).get("status"),
                    }
                    for e in done
                ]
                if done:
                    add(profile, "response.done", case, done, len(done))
            if profile == "doubao_s2s":
                done = [
                    r["event"]
                    for r in trace
                    if r.get("event", {}).get("type") == "response.output_audio.done"
                ]
                if done:
                    add(profile, "response.output_audio.done", case, done[0], len(done))
            if profile == "gemini":
                flags = case["gemini_content_flags"]
                for signal, field in [
                    ("interruption", "interrupted"),
                    ("turn.complete", "turnComplete"),
                ]:
                    count = flags.get(field, 0) + flags.get("turn_complete", 0) * (
                        field == "turnComplete"
                    )
                    if count:
                        add(
                            profile,
                            signal,
                            case,
                            {"native_field": field, "true_count": count},
                            count,
                        )
            if profile == "openai_live":
                if case["post_injection_controls"]:
                    raise AssertionError(
                        "Unexpected GPT Live cancellation requires contract review"
                    )
                continuous.setdefault(profile, []).append(
                    {
                        "artifact": case["artifact"],
                        "model_id": case["model_id"],
                        "input_seconds": case["input_seconds"],
                        "output_seconds": case["raw_audio"]["seconds"],
                        "quiet_seconds_rms_le_100": case["raw_audio"][
                            "rms_le_100_seconds"
                        ],
                        "longest_quiet_seconds": case["raw_audio"][
                            "longest_quiet_seconds"
                        ],
                        "trace_sha256": case["trace_sha256"],
                        "audio_sha256": case["raw_audio"]["sha256"],
                    }
                )
        else:
            rows = [
                json.loads(x)
                for x in path.with_name("native-events.jsonl").read_text().splitlines()
            ]
            if profile == "minicpm":
                for signal in [
                    "is_listen",
                    "end_of_turn",
                    "external_interrupt",
                    "forced_listen",
                ]:
                    matches = []
                    for r in rows:
                        e = r["native"]
                        control = e.get("control", {})
                        ok = (
                            signal == "is_listen"
                            and e.get("is_listen") is True
                            and control.get("source") == "native_runtime"
                            or signal == "end_of_turn"
                            and e.get("end_of_turn") is True
                            or signal == "external_interrupt"
                            and control.get("interrupt_requested") is True
                            and e.get("is_listen") is True
                            or signal == "forced_listen"
                            and control.get("force_listen") is True
                            and not control.get("interrupt_requested")
                            and e.get("is_listen") is True
                        )
                        if ok:
                            matches.append(
                                {
                                    "tick": r["tick"],
                                    "is_listen": e["is_listen"],
                                    "end_of_turn": e["end_of_turn"],
                                    "control": control,
                                    "output_rms": r["output_rms"],
                                }
                            )
                    if matches:
                        add(profile, signal, case, matches[0], len(matches))
            elif profile == "salmonn":
                for signal in [
                    "native_assistant_start",
                    "native_listening_transition",
                    "playback_turn_ended",
                    "listening_after_playback",
                ]:
                    matches = [
                        {"tick": r["tick"], **e}
                        for r in rows
                        for e in r["native"].get("events", [])
                        if e.get("event") == signal
                    ]
                    if signal == "playback_turn_ended":
                        for match in matches:
                            playback = match["playback"]
                            assert match["source"] == "playback_controller"
                            assert playback["text_closed"]
                            assert not playback["tts_errors"]
                            assert playback["cancelled_samples_16k"] == 0
                            assert (
                                playback["rendered_samples_16k"]
                                == playback["played_samples_16k"]
                            )
                    if matches:
                        add(profile, signal, case, matches[0], len(matches))
            elif profile == "freeze_omni":
                for phase, key in [
                    ("listening", "native_listen_stat"),
                    ("generation", "native_generate_stat"),
                ]:
                    states = Counter(
                        e["stat"]
                        for r in rows
                        for e in r["native"].get("events", [])
                        if e.get("type") == key
                    )
                    for state, count in states.items():
                        add(
                            profile,
                            f"{phase}.stat={state}",
                            case,
                            {"phase": phase, "stat": state},
                            count,
                        )
            elif profile == "moshi":
                if case["nonempty_output_ticks"] != case["input_ticks"]:
                    raise AssertionError(
                        "Moshi packet continuity changed; review warmup contract"
                    )
                continuous.setdefault(profile, []).append(
                    {
                        "artifact": case["artifact"],
                        "checkpoint": case["provenance"]["checkpoint"],
                        "input_ticks": case["input_ticks"],
                        "nonempty_output_ticks": case["nonempty_output_ticks"],
                        "frame_size_samples": case["provenance"]["frame_size"],
                        "sample_rate": case["provenance"]["sample_rate"],
                        "quiet_seconds_rms_le_100": case["raw_audio"][
                            "rms_le_100_seconds"
                        ],
                        "trace_sha256": case["trace_sha256"],
                        "audio_sha256": case["raw_audio"]["sha256"],
                    }
                )
    for path in sorted(args.root.glob("20261112-*/jobs/*-created.json")):
        if not path.stat().st_size:
            continue
        created = json.loads(path.read_text())
        jobs.append({"artifact": str(path), "sid": created.get("sid")})
    ledger = {
        "schema_version": 1,
        "date": "2026-11-12",
        "scope": "Real endpoint/native-weight signal and PCM observations; not autonomous interruption success or complete tau adapter live acceptance",
        "input_data": "Generic synthetic repository test fixtures and newly generated fixed synthetic text; no historical user conversations uploaded",
        "audio_activity_thresholds": {
            "trace_pcm16_rms": 100,
            "production_shared_rms": 80,
        },
        "signals": signals,
        "continuous_media": continuous,
        "cases": cases,
        "jobs": jobs,
        "rejected_contracts": {
            "salmonn/native_speech_end": "Upstream injects EOS into model INPUT after text EOT and real playback drain; do not wait for a model-predicted EOS output"
        },
        "failure_notes": {
            "gpt-realtime-1.5": "Current tested gateway closed before a usable audio session; do not transfer gpt-realtime-2.1 evidence",
            "initial_salmonn_sample_rate": "Initial probe read the wrong provenance key; excluded from signal admission; rerun with output_sample_rate fallback",
        },
    }
    args.output.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n")
    print(
        "cases",
        len(cases),
        "observed",
        sum(c["execution_status"] == "observed" for c in cases),
        "jobs",
        len(jobs),
    )
    for profile, items in signals.items():
        print(
            profile,
            {
                name: sum(w["matched_occurrences"] for w in item["witnesses"])
                for name, item in items.items()
            },
        )


if __name__ == "__main__":
    main()
