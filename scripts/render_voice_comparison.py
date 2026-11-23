"""Build a local comparison page for completed full-duplex runs."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

MODEL_NAMES = {
    "realtime": "GPT Realtime",
    "gpt-realtime": "GPT Realtime",
    "doubao": "Doubao Seeduplex",
    "doubao_complete": "Doubao Seeduplex",
    "grok": "Grok Voice",
    "xai": "Grok Voice",
    "gemini": "Gemini Live",
    "gemini-live": "Gemini Live",
    "gemini_live": "Gemini Live",
}


def read_json(path: Path, fallback=None):
    if not path.exists():
        return {} if fallback is None else fallback
    return json.loads(path.read_text(encoding="utf-8"))


def is_gpt_live(entry: dict) -> bool:
    value = f"{entry.get('directory', '')} {entry.get('name', '')}".lower()
    return "gpt-live" in value or "gptlive" in value


def sample_for(batch: Path, entry: dict) -> dict | None:
    folder = batch / entry["directory"]
    audio_path = folder / "audio" / "both.wav"
    assessment_path = folder / "assessment.json"
    if not audio_path.exists() or not assessment_path.exists():
        return None
    with wave.open(str(audio_path)) as audio:
        duration = audio.getnframes() / audio.getframerate()
    labels = []
    for role in ("user", "third_party", "assistant"):
        path = folder / "audio" / f"{role}_labels.txt"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t", 2)
            if len(parts) != 3:
                continue
            try:
                start, end = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            labels.append({"start": start, "end": end, "text": parts[2], "role": role})
    assessment = read_json(assessment_path)
    fields = (
        "event_id",
        "status",
        "start_seconds",
        "end_seconds",
        "stimulus",
        "verdict",
        "reason",
        "event_judgment",
        "judge_model",
        "timing_evidence",
        "metrics",
        "memory_assessment",
        "pause_audio_evidence",
        "asr_observations",
    )
    events = []
    for event in assessment.get("events", []):
        normalized = {key: event.get(key) for key in fields}
        if not normalized.get("event_judgment") and event.get("verdict"):
            normalized["event_judgment"] = {
                "verdict": event["verdict"],
                "reason": event.get("reason", ""),
                "model": event.get("judge_model"),
            }
        events.append(normalized)
    snapshots = read_json(folder / "perception.json", {"snapshots": []}).get(
        "snapshots", []
    )
    pause_values = [
        float(
            (event.get("stimulus") or {}).get("pause_seconds")
            or event.get("pause_seconds")
            or 0
        )
        for event in events
        if event.get("event_id") in {"rejection.pause", "proactive.completion"}
    ]
    return {
        **entry,
        "name": entry.get("name")
        or MODEL_NAMES.get(entry["directory"], entry["directory"]),
        "status": "complete",
        "duration": duration,
        "summary": assessment.get("summary", {}),
        "events": events,
        "pause_seconds": pause_values[0] if pause_values else None,
        "labels": sorted(labels, key=lambda row: row["start"]),
        "asr": snapshots,
        "review_notes": read_json(batch / "review_notes.json").get(
            entry["directory"], {}
        ),
        "observed_event_order": read_json(folder / "events.json").get(
            "observed_event_order", []
        ),
        "judge_models": sorted(
            {
                event.get("judge_model") or event["event_judgment"]["model"]
                for event in events
                if event.get("judge_model")
                or (event.get("event_judgment") or {}).get("model")
            }
        ),
    }


def build(batch: Path) -> Path:
    """Render available non-GPT-Live samples without changing evidence."""
    manifest = read_json(
        batch / "manifest.json", {"samples": [], "notes": [], "excluded": []}
    )
    entries = [entry for entry in manifest.get("samples", []) if not is_gpt_live(entry)]
    known = {entry["directory"] for entry in entries}
    for directory in ("grok", "xai", "gemini", "gemini-live", "gemini_live"):
        if directory not in known and (batch / directory).is_dir():
            entries.append(
                {
                    "directory": directory,
                    "name": MODEL_NAMES[directory],
                    "description": "Local live voice test",
                }
            )
    samples, failures = [], []
    for entry in entries:
        sample = sample_for(batch, entry)
        if sample:
            samples.append(sample)
            continue
        error_path = batch / entry["directory"] / "error.json"
        if error_path.exists():
            error = read_json(error_path)
            failures.append(
                {
                    "name": entry.get("name")
                    or MODEL_NAMES.get(entry["directory"], entry["directory"]),
                    "directory": entry["directory"],
                    "error": error.get("error_type", "run failed"),
                }
            )
    manifest["samples"] = [
        {key: entry.get(key) for key in ("name", "directory", "description")}
        for entry in entries
    ]
    manifest["failures"] = failures
    payload = json.dumps(
        {"manifest": manifest, "samples": samples}, ensure_ascii=False
    ).replace("<", r"\u003c")
    output = batch / "comparison.html"
    template = (
        Path(__file__).parent / "templates" / "voice_comparison.html"
    ).read_text(encoding="utf-8")
    output.write_text(template.replace("__DATA__", payload), encoding="utf-8")
    print(output.resolve())
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    build(parser.parse_args().batch)
