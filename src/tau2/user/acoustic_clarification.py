"""Private construction of aligned acoustic clarification inputs.

The user language model sees only neutral keyword annotations. All execution
state belongs to the audio executor and the post-conversation evaluator.
"""

import hashlib
import io
import math
import os
import time
import wave

import httpx
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tau2.config import (
    DEFAULT_ACOUSTIC_ALIGNMENT_ATTEMPTS,
    DEFAULT_ACOUSTIC_ALIGNMENT_GATEWAY_URL,
    DEFAULT_ACOUSTIC_ALIGNMENT_TIMEOUT_SECONDS,
    DEFAULT_ACOUSTIC_ALIGNMENT_URL,
)


class AcousticClarificationConfig(BaseModel):
    """Opt-in whole-utterance degradation; severity is always hard."""

    model_config = ConfigDict(extra="forbid")
    mode: str = Field(default="dropchunk", pattern="^dropchunk$")


class KeywordAnnotation(BaseModel):
    """A semantic keyword and its exact Python character slice in spoken text."""

    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)


class AlignmentError(ValueError):
    """An input cannot be grounded to an unambiguous audio span."""


def pcm_wav(samples, sample_rate):
    """Encode complete mono PCM as a WAV upload without a temporary file."""
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.astype("<i2", copy=False).tobytes())
    return output.getvalue()


def forced_alignment(samples, sample_rate, text):
    """Align clean audio using ElevenLabs, with bounded transient retries.

    Credentials and provider error bodies are never copied into evidence or the
    simulator's execution feedback.
    """
    gateway_credential = os.environ.get("TERUMI_BT_AK")
    credential = gateway_credential or os.environ.get("ELEVENLABS_API")
    if not credential:
        raise AlignmentError("missing_alignment_credential")
    audio = pcm_wav(samples, sample_rate)
    for attempt in range(DEFAULT_ACOUSTIC_ALIGNMENT_ATTEMPTS):
        try:
            response = httpx.post(
                DEFAULT_ACOUSTIC_ALIGNMENT_GATEWAY_URL
                if gateway_credential
                else DEFAULT_ACOUSTIC_ALIGNMENT_URL,
                headers={} if gateway_credential else {"xi-api-key": credential},
                params={"ak": gateway_credential} if gateway_credential else None,
                files={"file": ("utterance.wav", audio, "audio/wav")},
                data={"text": text},
                timeout=DEFAULT_ACOUSTIC_ALIGNMENT_TIMEOUT_SECONDS,
            )
        except httpx.HTTPError:
            if attempt + 1 < DEFAULT_ACOUSTIC_ALIGNMENT_ATTEMPTS:
                continue
            raise AlignmentError("alignment_transport_failure") from None
        if response.status_code == 429 or response.status_code >= 500:
            if attempt + 1 < DEFAULT_ACOUSTIC_ALIGNMENT_ATTEMPTS:
                continue
        if response.status_code != 200:
            raise AlignmentError(f"alignment_http_{response.status_code}")
        try:
            return response.json()
        except ValueError:
            raise AlignmentError("alignment_invalid_json") from None
    raise AlignmentError("alignment_retries_exhausted")


def aligned_samples(response, text, keyword, sample_rate, sample_count):
    """Map a validated textual occurrence using actual character/word times.

    No estimates are made inside an aligned token. Word-only responses must
    contain exact keyword boundaries. Text mismatch or omitted non-whitespace
    characters fail closed rather than selecting another occurrence.
    """
    if text[keyword.start_char : keyword.end_char] != keyword.text:
        raise AlignmentError("keyword_text_position_mismatch")
    rows = response.get("characters") or response.get("words")
    if not isinstance(rows, list) or not rows:
        raise AlignmentError("alignment_has_no_tokens")
    tokens, cursor, previous = [], 0, 0.0
    for row in rows:
        token = row.get("text")
        if not isinstance(token, str) or not token:
            raise AlignmentError("alignment_invalid_token")
        position = text.find(token, cursor)
        if position < 0 or text[cursor:position].strip():
            raise AlignmentError("alignment_text_mismatch")
        try:
            start, end = float(row["start"]), float(row["end"])
        except (KeyError, TypeError, ValueError):
            raise AlignmentError("alignment_invalid_time") from None
        if (
            not (math.isfinite(start) and math.isfinite(end))
            or not (0 <= start <= end <= sample_count / sample_rate + 1 / sample_rate)
            or start < previous - 1 / sample_rate
        ):
            raise AlignmentError("alignment_invalid_time")
        tokens.append((position, position + len(token), start, end))
        cursor, previous = position + len(token), end
    if text[cursor:].strip():
        raise AlignmentError("alignment_text_mismatch")
    selected = [
        row
        for row in tokens
        if row[0] >= keyword.start_char and row[1] <= keyword.end_char
    ]
    if (
        not selected
        or selected[0][0] != keyword.start_char
        or selected[-1][1] != keyword.end_char
    ):
        raise AlignmentError("keyword_not_on_aligned_boundaries")
    start = max(0, round(selected[0][2] * sample_rate))
    end = min(sample_count, round(selected[-1][3] * sample_rate))
    if end <= start:
        raise AlignmentError("alignment_empty_target")
    return start, end


def hard_dropchunk(samples, target, seed):
    """Silence two locally sampled 65–90 percent gaps, allowing overlap.

    Preserves the source recipe's integer length clamps and full-gap placement
    inside the target. Local RNG avoids global Torch/NumPy random side effects.
    """
    start, end = target
    length = end - start
    maximum = max(1, length - 1)
    low = max(1, min(int(0.65 * length), maximum))
    high = max(low, min(int(0.90 * length), maximum))
    rng = np.random.default_rng(seed)
    intervals = []
    result = samples.copy()
    for _ in range(2):
        count = int(rng.integers(low, high + 1))
        onset = int(rng.integers(start, end - count + 1))
        intervals.append([onset, onset + count])
        result[onset : onset + count] = 0
    union = []
    for onset, stop in sorted(intervals):
        if union and onset <= union[-1][1]:
            union[-1][1] = max(union[-1][1], stop)
        else:
            union.append([onset, stop])
    return result, {
        "drop_intervals_samples": intervals,
        "drop_union_samples": union,
        "drop_length_low": low,
        "drop_length_high": high,
    }


def degrade_utterance(samples, text, keyword, sample_rate, seed, record):
    """Construct one private evidence record and a whole degraded waveform."""
    started = time.perf_counter()
    record.update(
        {
            "sample_rate": sample_rate,
            "sample_count": len(samples),
            "script_text": text,
            "keyword": keyword.model_dump() if keyword else None,
            "seed": seed,
            "clean_sha256": hashlib.sha256(samples.tobytes()).hexdigest(),
            "timing": {},
            "status": "failed",
            "delivery": [],
            "delivered_samples": 0,
            "delivered_target_samples": 0,
        }
    )
    try:
        if keyword is None:
            raise AlignmentError("missing_keyword_annotation")
        if (
            not (0 <= keyword.start_char < keyword.end_char <= len(text))
            or text[keyword.start_char : keyword.end_char] != keyword.text
        ):
            raise AlignmentError("keyword_text_position_mismatch")
        alignment_start = time.perf_counter()
        try:
            response = forced_alignment(samples, sample_rate, text)
        finally:
            record["timing"]["alignment_seconds"] = (
                time.perf_counter() - alignment_start
            )
        record["alignment_response"] = response
        target = aligned_samples(response, text, keyword, sample_rate, len(samples))
        record["target_samples"] = list(target)
        operation_start = time.perf_counter()
        samples, details = hard_dropchunk(samples, target, seed)
        record["timing"]["degradation_seconds"] = time.perf_counter() - operation_start
        record.update(details)
        record["status"] = "prepared"
    except (AlignmentError, TypeError, AttributeError) as exc:
        record["failure_reason"] = (
            str(exc) if isinstance(exc, AlignmentError) else "alignment_invalid_schema"
        )
    record["timing"].setdefault("alignment_seconds", 0.0)
    record["timing"].setdefault("degradation_seconds", 0.0)
    record["timing"]["tts_postprocess_seconds"] = time.perf_counter() - started
    record["degraded_sha256"] = hashlib.sha256(samples.tobytes()).hexdigest()
    return samples
