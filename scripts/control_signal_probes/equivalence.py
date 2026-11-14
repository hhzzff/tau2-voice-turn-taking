"""Controlled real-time replay and an independently clocked 100ms prototype.

This diagnostic does not claim to exercise FullDuplexOrchestrator. Receive
timestamps remain high resolution; only prototype delivery is quantized.
"""

import argparse
import audioop
import base64
import hashlib
import json
import math
import os
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PI = Path("/mnt/bn/audio-visual-llm-data6/terumi/pi-bench")
PROMPT = (
    "Speak English. Follow the spoken instructions exactly. When asked to count, "
    "count slowly from one to one hundred without commentary. When the user says "
    "stop, stop counting and answer the new arithmetic question with just the number."
)


def read_wav(path, rate):
    """Load mono PCM16 once so both paths receive identical samples."""
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError("Expected mono PCM16")
        pcm = handle.readframes(handle.getnframes())
        return audioop.ratecv(pcm, 2, 1, handle.getframerate(), rate, None)[0]


def write_wav(path, pcm, rate):
    """Write a playable evidence track."""
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm)


def render(events, raw, *, rate, horizon_ms, quantum_ms):
    """Replay arrival/cancel events causally, optionally at scheduler boundaries.

    Natural completion preserves queued tails. Cancellation truncates the old
    response at delivery time and rejects its later packets. Audio cannot be
    placed earlier than its observed availability. No global time realignment.
    """
    spans, blocked = [], set()
    cursor = 0
    for event in events:
        at = max(0, event["at_ms"])
        if quantum_ms:
            at = math.ceil(at / quantum_ms) * quantum_ms
        sample = math.ceil(at * rate / 1000)
        identity = event.get("response_id")
        if event["kind"] == "cancel":
            identities = (
                {identity}
                if identity is not None
                else {s["response_id"] for s in spans if s["end"] > sample}
            )
            blocked.update(identities)
            retained = []
            for span in spans:
                if span["response_id"] in identities:
                    span = {**span, "end": min(span["end"], sample)}
                if span["end"] > span["start"]:
                    retained.append(span)
            spans = retained
            cursor = max([sample] + [s["end"] for s in spans])
        elif event["kind"] == "audio" and identity not in blocked:
            length = event["audio_bytes"] // 2
            if length:
                start = max(sample, cursor)
                spans.append(
                    {
                        "start": start,
                        "end": start + length,
                        "offset": event["raw_offset_bytes"],
                        "response_id": identity,
                    }
                )
                cursor = start + length
    end = math.ceil(horizon_ms * rate / 1000)
    output = bytearray(end * 2)
    result = []
    for span in spans:
        start, stop = span["start"], min(span["end"], end)
        if start >= stop:
            continue
        count = (stop - start) * 2
        pcm = raw[span["offset"] : span["offset"] + count]
        if len(pcm) != count:
            raise ValueError("Trace points outside raw PCM evidence")
        output[start * 2 : stop * 2] = pcm
        result.append(
            {
                "response_id": span["response_id"],
                "start_ms": start * 1000 / rate,
                "end_ms": stop * 1000 / rate,
            }
        )
    return bytes(output), result


def activity(pcm, rate):
    """10ms RMS windows, joining gaps shorter than 200ms into speech regions."""
    size = rate * 2 // 100
    active = [
        i * 10
        for i, start in enumerate(range(0, len(pcm), size))
        if pcm[start : start + size] and audioop.rms(pcm[start : start + size], 2) > 100
    ]
    regions = []
    for at in active:
        if regions and at - regions[-1][1] < 200:
            regions[-1][1] = at + 10
        else:
            regions.append([at, at + 10])
    return regions


def finish(directory, events, raw, *, rate, horizon_ms, quantum_ms, metadata):
    """Save unshifted arrival and playback timelines plus readable metrics."""
    (directory / "events.json").write_text(json.dumps(events, indent=2))
    (directory / "raw.pcm16").write_bytes(raw)
    direct, _ = render(events, raw, rate=rate, horizon_ms=horizon_ms, quantum_ms=0)
    played, spans = render(
        events, raw, rate=rate, horizon_ms=horizon_ms, quantum_ms=quantum_ms
    )
    write_wav(directory / "arrival-playback.wav", direct, rate)
    write_wav(directory / "played.wav", played, rate)
    regions = activity(played, rate)
    result = {
        **metadata,
        "playback_quantum_ms": quantum_ms,
        "raw_audio_sha256": hashlib.sha256(raw).hexdigest(),
        "speech_regions_ms": regions,
        "first_voice_ms": regions[0][0] if regions else None,
        "post_interruption_voice_ms": next(
            (a for a, b in regions if a >= metadata["interruption_end_ms"]), None
        ),
        "response_spans": spans,
        "cancellations": [e for e in events if e["kind"] == "cancel"],
        "arrival_speech_regions_ms": activity(direct, rate),
        "tau_production_orchestrator": False,
    }
    (directory / "result.json").write_text(json.dumps(result, indent=2))
    return result


def build_input(root):
    """One 40-second file: count request at 0s, correction at exactly 10s."""
    pcm = bytearray(40 * 32000)
    utterances = []
    for start, name in [(0, "greeting"), (10, "interruption")]:
        audio = read_wav(root / "stimuli" / f"{name}.wav", 16000)
        offset = start * 32000
        if offset + len(audio) > len(pcm):
            raise ValueError("Input horizon too short")
        pcm[offset : offset + len(audio)] = audio
        utterances.append(
            {
                "name": name,
                "start_ms": start * 1000,
                "end_ms": start * 1000 + len(audio) / 32,
            }
        )
    write_wav(root / "input.wav", pcm, 16000)
    (root / "input.json").write_text(
        json.dumps(
            {
                "utterances": utterances,
                "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                "seconds": 40,
            },
            indent=2,
        )
    )


def closed_case(family, mode, run, root, scheduled_transport="paced"):
    """Compare continuous 20ms upload with causal 100ms input batching."""
    sys.path.insert(0, str(PI))
    from inference.realtime_tick import TickConfig, TickSession
    from inference.realtime_tick_backend import FAMILIES

    model, endpoint, rate, names = FAMILIES[family]
    key = next((os.environ[n] for n in names if os.environ.get(n)), "")
    if not key:
        raise RuntimeError(f"Missing credential for {family}")
    directory = root / family / f"{run}-{mode}"
    config = TickConfig(
        family,
        model,
        endpoint,
        input_rate=rate,
        output_rate=24000,
        tick_ms=100,
        transport_ms=20,
        realtime_upload=True,
    )
    session = TickSession(config, key, PROMPT, directory)
    pcm = read_wav(root / "input.wav", rate)
    status, error = "failed", None
    origin = None
    try:
        session.connect()
        origin = time.monotonic()
        origin_ms = (origin - session.epoch) * 1000
        session.emit("input_origin", origin_ms=origin_ms, mode=mode)
        block_ms = 20 if mode == "direct" or scheduled_transport == "paced" else 100
        block = rate * 2 * block_ms // 1000
        packet = rate * 2 * 20 // 1000
        for offset in range(0, len(pcm), block):
            end = min(offset + block, len(pcm))
            deadline = origin + end / (rate * 2)
            time.sleep(max(0, deadline - time.monotonic()))
            session._check()
            for pos in range(offset, end, packet):
                part = pcm[pos : min(pos + packet, end)]
                audio = base64.b64encode(part).decode()
                event = (
                    {
                        "realtime_input": {
                            "audio": {
                                "mime_type": f"audio/pcm;rate={rate}",
                                "data": audio,
                            }
                        }
                    }
                    if family == "gemini"
                    else {"type": "input_audio_buffer.append", "audio": audio}
                )
                with session.cv:
                    session._send(event)
                    session.stream_ms += len(part) * 500 / rate
                    session.source_ms = session.stream_ms
                    session.emit(
                        "send",
                        input_end_ms=session.stream_ms,
                        pcm_sha256=hashlib.sha256(part).hexdigest(),
                    )
        time.sleep(1)
        session._check()
        status = "observed" if session.first_voice else "no_audio"
    except Exception as exc:
        error = str(exc).replace(key, "[REDACTED]")
    finally:
        session.closing = True
        if session.ws is not None and session.ws.connected:
            try:
                if family == "doubao":
                    session._send({"type": "session.close"})
                    session.close_ack.wait(3)
                session.ws.close(1000, "equivalence replay complete")
            except Exception:
                pass
        session.closed.wait(2)
        session._events.close()
    events = []
    if origin is not None:
        for line in (directory / "scheduler-events.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["kind"] == "audio":
                events.append({**row, "at_ms": row["wall_ms"] - origin_ms})
            elif row["kind"] == "interruption":
                events.append(
                    {
                        "kind": "cancel",
                        "at_ms": row["wall_ms"] - origin_ms,
                        "response_id": row["old_response_id"],
                        "event_type": row["event_type"],
                    }
                )
    result = finish(
        directory,
        events,
        bytes(session.raw),
        rate=24000,
        horizon_ms=41000,
        quantum_ms=0 if mode == "direct" else 100,
        metadata={
            "model": model,
            "family": family,
            "mode": mode,
            "run": run,
            "status": status,
            "error": error,
            "input_sha256": hashlib.sha256(pcm).hexdigest(),
            "text": session.response_text,
            "prompt": PROMPT,
            "input_origin_wall_ms": origin_ms if origin is not None else None,
            "scheduled_transport": scheduled_transport,
            "interruption_end_ms": json.loads((root / "input.json").read_text())[
                "utterances"
            ][1]["end_ms"],
        },
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("model", "mode", "run", "status", "first_voice_ms", "error")
            }
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--build-input", action="store_true")
    parser.add_argument("--models", nargs="+", default=["gpt", "gemini", "doubao"])
    parser.add_argument(
        "--scheduled-transport", choices=["paced", "batched"], default="paced"
    )
    args = parser.parse_args()
    if args.build_input:
        build_input(args.root)
        return
    sys.path.insert(0, str(PI))
    from inference.repo_credentials import load_repo_credentials

    load_repo_credentials()

    def provider(family):
        return [
            closed_case(family, mode, n, args.root, args.scheduled_transport)
            for n, mode in enumerate(("direct", "scheduled", "scheduled", "direct"))
        ]

    with ThreadPoolExecutor(max_workers=len(args.models)) as executor:
        results = list(executor.map(provider, args.models))
    (args.root / "closed-results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
