#!/usr/bin/env python3
"""Verify ByteDance ModelHub ElevenLabs TTS, ASR, and forced alignment.

Run from the repository root with .venv/bin/python. Requires requests and
python-dotenv (both core dependencies). Outputs contain no API key.
"""

import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
import time
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import dotenv_values

BASE_URL = "https://aidp.bytedance.net/api/modelhub/online/multimodal/crawl/11labs/v1"
DEFAULT_TEXT = "Good morning everyone, thank you for joining the meeting today."
SAMPLE_RATE = 16000


def write_json(path: Path, value: object) -> None:
    """Write a readable, UTF-8 evidence artifact."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def normalize(text: str) -> str:
    """Ignore English punctuation and case for this short smoke test."""
    return " ".join(re.findall(r"\w+", text.casefold()))


class Gateway:
    """Minimal gateway client; never log an authenticated request URL."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.calls: list[dict] = []

    def request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Send one request, without automatic retries or redirects."""
        params = {"ak": self.api_key, **kwargs.pop("params", {})}
        log_id = uuid.uuid4().hex
        started = time.perf_counter()
        record = {"method": method, "endpoint": endpoint, "log_id": log_id}
        try:
            response = self.session.request(
                method,
                BASE_URL + endpoint,
                params=params,
                headers={"X-TT-LOGID": log_id},
                timeout=(10, 120),
                allow_redirects=False,
                **kwargs,
            )
        except requests.RequestException as exc:
            record.update(error_type=type(exc).__name__)
            # requests exception messages can include the URL and its AK.
            raise RuntimeError(
                f"{method} {endpoint}: {type(exc).__name__}; log_id={log_id}"
            ) from None
        else:
            record.update(
                status=response.status_code,
                content_type=response.headers.get("Content-Type", ""),
                response_bytes=len(response.content),
            )
        finally:
            record["wall_seconds"] = round(time.perf_counter() - started, 3)
            self.calls.append(record)
            print(json.dumps(record), flush=True)
        if not 200 <= response.status_code < 300:
            # Persist a redacted body for diagnosis, without a URL/headers dump.
            body = response.text.replace(self.api_key, "<REDACTED_AK>")
            raise RuntimeError(
                f"{method} {endpoint}: HTTP {response.status_code}; "
                f"log_id={log_id}; body={body[:1000]}"
            )
        return response


def run(args: argparse.Namespace, api: Gateway, report: dict) -> None:
    """Generate one utterance and submit it once to each audio endpoint."""
    output = args.output_dir
    if args.voice_id:
        voice_id = args.voice_id
        voice_name = "explicit --voice-id"
    else:
        voices = api.request("GET", "/voices").json().get("voices", [])
        if not voices:
            raise RuntimeError("The gateway returned no accessible voices")
        voice = next((v for v in voices if v.get("name") == "Rachel"), voices[0])
        voice_id, voice_name = voice["voice_id"], voice.get("name", "")
    report.update(
        voice_id=voice_id,
        voice_name=voice_name,
        tts_model="eleven_multilingual_v2",
        asr_model="scribe_v1",
        output_format="pcm_16000",
        text=args.text,
    )
    (output / "input.txt").write_text(args.text + "\n")

    audio = api.request(
        "POST",
        f"/text-to-speech/{voice_id}",
        params={"output_format": "pcm_16000"},
        json={
            "text": args.text,
            "model_id": "eleven_multilingual_v2",
            "model": "elevenlabs",
            "stream": False,
        },
    )
    pcm = audio.content
    if not audio.headers.get("Content-Type", "").lower().startswith("audio/"):
        raise RuntimeError("TTS did not return an audio content type")
    if not pcm or len(pcm) % 2:
        raise RuntimeError("TTS returned empty or incomplete PCM16 samples")
    if pcm.startswith((b"{", b"RIFF", b"ID3", b"OggS")):
        raise RuntimeError("TTS response is not the requested raw PCM format")
    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    rms = math.sqrt(sum(value * value for value in samples) / len(samples))
    if rms <= 1:
        raise RuntimeError("TTS returned silent PCM")
    wav_path = output / "tts.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    duration = len(samples) / SAMPLE_RATE
    report["audio"] = {
        "file": "tts.wav",
        "sha256": hashlib.sha256(wav_path.read_bytes()).hexdigest(),
        "pcm_bytes": len(pcm),
        "duration_seconds": duration,
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "sample_width_bytes": 2,
        "rms_pcm16": round(rms, 3),
    }

    # requests generates multipart Content-Type and its boundary automatically.
    with wav_path.open("rb") as file:
        asr = api.request(
            "POST",
            "/speech-to-text",
            files={"file": (wav_path.name, file, "audio/wav")},
            data={"model_id": "scribe_v1", "language_code": "en"},
        ).json()
    write_json(output / "asr.json", asr)
    transcript = asr.get("text", "")
    report["asr"] = {
        "text": transcript,
        "language_code": asr.get("language_code"),
        "word_entries": len(asr.get("words", [])),
        "nonspace_word_entries": sum(
            bool(word.get("text", "").strip()) for word in asr.get("words", [])
        ),
        "normalized_exact_match": normalize(transcript) == normalize(args.text),
    }

    with wav_path.open("rb") as file:
        alignment = api.request(
            "POST",
            "/forced-alignment",
            files={"file": (wav_path.name, file, "audio/wav")},
            data={"text": args.text},
        ).json()
    write_json(output / "alignment.json", alignment)
    words = alignment.get("words", [])
    timing_valid = bool(words)
    previous_end = 0.0
    for word in words:
        start, end = word.get("start"), word.get("end")
        if (
            not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not 0 <= start <= end <= duration + 0.1
            or start < previous_end - 0.001
        ):
            timing_valid = False
            break
        previous_end = end
    aligned_text = " ".join(w.get("text", "") for w in words)
    report["alignment"] = {
        "word_entries": len(words),
        "nonspace_word_entries": sum(bool(w.get("text", "").strip()) for w in words),
        "character_entries": len(alignment.get("characters", [])),
        "loss": alignment.get("loss"),
        "timestamps_valid": timing_valid,
        "normalized_exact_match": normalize(aligned_text) == normalize(args.text),
        "first_word": words[0] if words else None,
        "last_word": words[-1] if words else None,
    }
    if not report["asr"]["normalized_exact_match"]:
        raise RuntimeError("ASR transcript did not match the smoke-test text")
    if not timing_valid or not report["alignment"]["normalized_exact_match"]:
        raise RuntimeError("Alignment failed timestamp or text coverage checks")
    report["success"] = True


def main() -> int:
    """Load the gateway AK and save the result even if a provider call fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--voice-id", help="Skip voice discovery with a known ID")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/tmp/byted-elevenlabs-smoke")
    )
    args = parser.parse_args()
    api_key = os.getenv("TERUMI_BT_AK") or dotenv_values(args.env_file).get(
        "TERUMI_BT_AK"
    )
    if not api_key:
        print("Missing TERUMI_BT_AK in environment or --env-file", file=sys.stderr)
        return 1
    if args.output_dir.exists():
        print("Choose a new --output-dir to preserve earlier evidence", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True)
    api = Gateway(api_key)
    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": BASE_URL,
        "success": False,
        "calls": api.calls,
    }
    try:
        run(args, api, report)
    except (RuntimeError, ValueError, OSError, KeyError) as exc:
        report["error"] = str(exc).replace(api_key, "<REDACTED_AK>")
        print(report["error"], file=sys.stderr)
    finally:
        api.session.close()
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output_dir / "report.json", report)
    print(f"success={report['success']}; report={args.output_dir / 'report.json'}")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
