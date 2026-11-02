"""Collect final ASR evidence from consumed audio, never from intended scripts."""

import copy
import hashlib
import time
import wave
from pathlib import Path

from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat
from tau2.data_model.voice import TranscriptionConfig


def _read_wav(path):
    with wave.open(str(path), "rb") as stream:
        if stream.getsampwidth() != 2 or stream.getnchannels() != 1:
            raise ValueError("Evaluation evidence requires mono PCM16 WAV")
        return stream.readframes(stream.getnframes()), stream.getframerate()


def _write_wav(path, data, rate):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(data)


def finalize_acoustic_evidence(
    plan,
    report: dict,
    output_dir: Path,
    tick_seconds: float,
    ticks: list,
    transcription_config: TranscriptionConfig | None = None,
    transcribe: bool = True,
) -> dict:
    """Finalize actual consumed audio and prefix ASR after the conversation.

    Tick n user output is consumed in tick n+1. The final emitted frame without
    a receiving tick is excluded. Provider-input tap sample zero belongs to the
    first receiving tick (1), not the initial user tick (0). Offline assessment
    reads these saved results and never invokes this collection stage.
    """
    from tau2.config import DEFAULT_WHISPER_MODEL
    from tau2.voice.transcription.transcribe import transcribe_audio

    result = copy.deepcopy(report)
    if "acoustic_clarification" not in result:
        return result
    started = time.perf_counter()
    config = transcription_config or TranscriptionConfig(
        model=DEFAULT_WHISPER_MODEL,
        language="zh" if plan.language == "Chinese" else "en",
    )
    acoustic = result["acoustic_clarification"]
    evidence = {
        "transcripts": [],
        "context_prefixes": {},
        "errors": [],
        "asr_config": config.model_dump(),
        "note": "ASR of actual consumed audio is fallible evidence, not verified literal speech. Prefixes exclude audio after the earliest possible semantic request onset. No word onset is inferred from ASR arrival time.",
    }
    acoustic["evidence"] = evidence
    receiving = {
        item["tick_id"] if isinstance(item, dict) else item.tick_id for item in ticks
    }
    provider_path = output_dir / "agent_audio/agent_input.wav"
    provider = None
    if provider_path.exists():
        try:
            provider = _read_wav(provider_path)
            evidence["provider_input"] = {
                "audio_file": "agent_audio/agent_input.wav",
                "first_receiver_tick": 1,
                "audio_origin_seconds": tick_seconds,
                "sha256": hashlib.sha256(provider[0]).hexdigest(),
                "sample_rate": provider[1],
            }
        except Exception as exc:
            evidence["errors"].append(
                {"stage": "provider_input", "error": type(exc).__name__}
            )
    buffers = {}

    def recognize(record, data, rate):
        relative = f"acoustic_evaluation_audio/{record['id']}.wav"
        _write_wav(output_dir / relative, data, rate)
        record.update(
            audio_file=relative,
            sha256=hashlib.sha256(data).hexdigest(),
            asr_model=config.model,
            sample_rate=rate,
            audio_bytes=len(data),
        )
        if not transcribe:
            record.update(transcript="", error="asr_not_requested")
            return record
        try:
            recognized = transcribe_audio(
                AudioData(
                    data=data,
                    format=AudioFormat(
                        encoding=AudioEncoding.PCM_S16LE, sample_rate=rate
                    ),
                ),
                config,
            )
            record["transcript"] = recognized.transcript
            if recognized.error:
                record["error"] = "transcription_failed"
        except Exception as exc:
            record["transcript"] = ""
            record["error"] = type(exc).__name__
        return record

    for utterance in acoustic.get("user_utterances", []):
        received_data = []
        rate = utterance["sample_rate"]
        source = "source_pcm_consumed_by_tick"
        try:
            emitted, emitted_rate = _read_wav(
                output_dir / utterance["delivered_audio_file"]
            )
            offset = 0
            confirmed = []
            for delivery in utterance.get("delivery", []):
                width = delivery["sample_end"] - delivery["sample_start"]
                padded = width + delivery.get("padding_samples", 0)
                receiver_tick = round(delivery["start_seconds"] / tick_seconds) + 1
                delivery["receiver_tick"] = receiver_tick
                delivery["received"] = receiver_tick in receiving
                delivery["timing_source"] = (
                    "user_emission_tick_n_consumed_in_agent_tick_n_plus_1"
                )
                delivery["received_start_seconds"] = receiver_tick * tick_seconds
                delivery["received_end_seconds"] = (
                    receiver_tick * tick_seconds + width / utterance["sample_rate"]
                )
                if delivery["received"]:
                    if provider:
                        raw, rate = provider
                        a = round((receiver_tick - 1) * tick_seconds * rate) * 2
                        b = a + round(width / utterance["sample_rate"] * rate) * 2
                        if b > len(raw):
                            raise ValueError(
                                "Provider tap is shorter than receiving-tick evidence"
                            )
                        received_data.append(raw[a:b])
                        source = "decoded_provider_input"
                    else:
                        received_data.append(emitted[offset * 2 : (offset + width) * 2])
                        rate = emitted_rate
                    confirmed.append(delivery)
                offset += padded
            utterance["received_modified_samples"] = sum(
                max(0, min(item["sample_end"], b) - max(item["sample_start"], a))
                for item in confirmed
                for a, b in utterance.get("modified_intervals_samples", [])
            )
            utterance["received_natural_end_seconds"] = (
                utterance["natural_end_seconds"] + tick_seconds
            )
            if not received_data:
                continue
            data = b"".join(received_data)
            row = {
                "id": utterance["id"],
                "role": "user",
                "source": source,
                "start_seconds": confirmed[0]["received_start_seconds"],
                "end_seconds": confirmed[-1]["received_end_seconds"],
                "utterance_id": utterance["id"],
            }
            recognize(row, data, rate)
            evidence["transcripts"].append(row)
            buffers[row["id"]] = (data, rate)
        except Exception as exc:
            evidence["errors"].append(
                {
                    "stage": "user_audio",
                    "utterance_id": utterance["id"],
                    "error": type(exc).__name__,
                }
            )

    # Targets and all-user rows share IDs, even in artifacts whose JSON decoding
    # no longer preserves Python object identity.
    users = {item["id"]: item for item in acoustic.get("user_utterances", [])}
    for target in acoustic.get("utterances", []):
        if target["id"] in users:
            target.update(users[target["id"]])

    groups = []
    for observation in acoustic.get("assistant_observations", []):
        if not observation.get("audio_file") or not (
            observation.get("audible") or observation.get("utterance_ids")
        ):
            continue
        ids = observation.get("utterance_ids", [])
        if (
            groups
            and groups[-1]["utterance_ids"] == ids
            and abs(groups[-1]["end_seconds"] - observation["start_seconds"]) <= 1e-6
        ):
            groups[-1]["observations"].append(observation)
            groups[-1]["end_seconds"] = observation["end_seconds"]
        else:
            groups.append(
                {
                    "id": f"assistant-{len(groups) + 1:04d}",
                    "role": "assistant",
                    "start_seconds": observation["start_seconds"],
                    "end_seconds": observation["end_seconds"],
                    "utterance_ids": ids,
                    "observations": [observation],
                }
            )
    for group in groups:
        try:
            parts = [
                _read_wav(output_dir / item["audio_file"])
                for item in group["observations"]
            ]
            rate = parts[0][1]
            if any(part[1] != rate for part in parts):
                raise ValueError("Assistant format changed inside an utterance")
            data = b"".join(part[0] for part in parts)
            row = {key: value for key, value in group.items() if key != "observations"}
            row["source"] = "actual_delivered_assistant_audio"
            recognize(row, data, rate)
            evidence["transcripts"].append(row)
        except Exception as exc:
            evidence["errors"].append(
                {
                    "stage": "assistant_audio",
                    "segment_id": group["id"],
                    "error": type(exc).__name__,
                }
            )
    evidence["transcripts"].sort(key=lambda row: row["start_seconds"])
    for assistant in [
        row for row in evidence["transcripts"] if row["role"] == "assistant"
    ]:
        cutoff = assistant["start_seconds"]
        context = []
        for row in evidence["transcripts"]:
            if row["end_seconds"] <= cutoff:
                context.append(row)
            elif (
                row["role"] == "user"
                and row["start_seconds"] < cutoff
                and row["id"] in buffers
            ):
                data, rate = buffers[row["id"]]
                data = data[: max(0, int((cutoff - row["start_seconds"]) * rate)) * 2]
                if data:
                    prefix = {
                        "id": f"{row['id']}-prefix-{assistant['id']}",
                        "role": "user",
                        "start_seconds": row["start_seconds"],
                        "end_seconds": cutoff,
                        "source": row["source"],
                        "cutoff_seconds": cutoff,
                    }
                    recognize(prefix, data, rate)
                    if prefix.get("error"):
                        evidence["errors"].append(
                            {
                                "stage": "prefix_asr",
                                "segment_id": prefix["id"],
                                "error": prefix["error"],
                            }
                        )
                    context.append(prefix)
        evidence["context_prefixes"][assistant["id"]] = context
    evidence["collection_seconds"] = time.perf_counter() - started
    return result
