"""Measure saved SSML gaps and optionally repair event markers before rejudging."""

import argparse
import json
import wave
from pathlib import Path

import numpy as np

from tau2.voice.utils.pause_evidence import locate_synthesized_pause


def audit(folder: Path, apply: bool = False) -> list[dict]:
    """Audit original TTS and delivered user-channel PCM without changing audio."""
    event_path = folder / "events.json"
    original = event_path.read_text(encoding="utf-8")
    events = json.loads(original)
    utterances = json.loads((folder / "synthesis.json").read_text(encoding="utf-8"))[
        "utterances"
    ]
    rows = []
    for event in events["events"]:
        stimulus = event.get("stimulus") or {}
        requested = stimulus.get("pause_seconds", 0)
        if not requested:
            continue
        matches = [
            u
            for u in utterances
            if u.get("ssml") and u.get("script_text") == stimulus.get("text")
        ]
        if len(matches) != 1:
            rows.append(
                {"event_id": event["event_id"], "status": "no_unique_ssml_record"}
            )
            continue
        utterance = matches[0]
        with wave.open(str(folder / utterance["audio_file"]), "rb") as wav:
            rate = wav.getframerate()
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        evidence = locate_synthesized_pause(samples, rate, requested)
        evidence["audio_file"] = utterance["audio_file"]
        if evidence["status"] == "measured":
            absolute_start = event["start_seconds"] + evidence["start_seconds"]
            absolute_end = event["start_seconds"] + evidence["end_seconds"]
            # The saved stereo file is user-left / assistant-right. Verify that
            # the quiet interval was delivered, even when the assistant overlaps.
            with wave.open(str(folder / "audio" / "both.wav"), "rb") as wav:
                delivered_rate = wav.getframerate()
                wav.setpos(round(absolute_start * delivered_rate))
                data = (
                    np.frombuffer(
                        wav.readframes(
                            round((absolute_end - absolute_start) * delivered_rate)
                        ),
                        dtype="<i2",
                    )
                    .reshape(-1, wav.getnchannels())[:, 0]
                    .astype(float)
                )
            evidence["delivered_user_channel_rms"] = float(np.sqrt(np.mean(data**2)))
            evidence["absolute_start_seconds"] = absolute_start
            evidence["absolute_end_seconds"] = absolute_end
            if apply:
                event.setdefault("original_marker_seconds", event["marker_seconds"])
                event["marker_seconds"] = absolute_start
                event["pause_audio_evidence"] = evidence
        rows.append({"event_id": event["event_id"], **evidence})
    if apply:
        backup = folder / "events.before_pause_alignment.json"
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        event_path.write_text(
            json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (folder / "pause_audit.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", type=Path, nargs="+")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    for folder in args.folders:
        print(json.dumps({"folder": str(folder), "pauses": audit(folder, args.apply)}))
