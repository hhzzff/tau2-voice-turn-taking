"""Private construction of aligned acoustic clarification inputs.

The user language model sees only neutral keyword annotations. All execution
state belongs to the audio executor and the post-conversation evaluator.
"""

import hashlib
import io
import math
import os
import struct
import time
import wave
from pathlib import Path
from typing import Literal

import httpx
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tau2.config import (
    DEFAULT_ACOUSTIC_ALIGNMENT_ATTEMPTS,
    DEFAULT_ACOUSTIC_ALIGNMENT_GATEWAY_URL,
    DEFAULT_ACOUSTIC_ALIGNMENT_TIMEOUT_SECONDS,
    DEFAULT_ACOUSTIC_ALIGNMENT_URL,
    DEFAULT_ACOUSTIC_DNS_NOISE_PATH,
    DEFAULT_ACOUSTIC_KEYWORD_LIMIT,
)


class AcousticClarificationConfig(BaseModel):
    """Opt-in whole-utterance degradation; severity is always hard."""

    model_config = ConfigDict(extra="forbid")
    mode: Literal["dropchunk", "dns_noise"] = "dropchunk"
    noise_path: str = DEFAULT_ACOUSTIC_DNS_NOISE_PATH


class KeywordAnnotation(BaseModel):
    """A semantic keyword and its exact Python character slice in spoken text."""

    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, gt=0)
    occurrence: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def one_location(self):
        """Require one explicit location representation; never guess a match."""
        coordinates = self.start_char is not None or self.end_char is not None
        if coordinates:
            if (
                self.start_char is None
                or self.end_char is None
                or self.occurrence is not None
            ):
                raise ValueError(
                    "Use either a complete character range or an occurrence index"
                )
        elif self.occurrence is None:
            raise ValueError(
                "An explicit occurrence index or character range is required"
            )
        return self


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


def hard_dropchunk(samples, target, seed, *, output=None):
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
    result = samples.copy() if output is None else output
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


class PreparedNoise(BaseModel):
    """One immutable, predecoded asset and private preparation diagnostics."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    samples: np.ndarray | None = Field(default=None, exclude=True)
    source: str | None = None
    sha256: str | None = None
    sample_rate: int
    preparation_seconds: float = 0.0
    failure_reason: str | None = None


def prepare_noise(config, sample_rate, seed):
    """Load one deterministically selected WAV before any utterance is timed.

    A directory is scanned once. Decoding, mono conversion, resampling and asset
    hashing happen here, never inside the local degradation operation.
    """
    if config is None or config.mode != "dns_noise":
        return None
    started = time.perf_counter()
    prepared = PreparedNoise(sample_rate=sample_rate)
    try:
        from scipy.io import wavfile
        from scipy.signal import resample_poly

        path = Path(config.noise_path)
        if path.is_dir():
            candidates = sorted(
                p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".wav"
            )
            if not candidates:
                candidates = sorted(path.rglob("*.wav"))
            if not candidates:
                raise AlignmentError("noise_directory_has_no_wavs")
            path = candidates[
                int(np.random.default_rng(seed).integers(len(candidates)))
            ]
        prepared.source = str(path)
        raw = path.read_bytes()
        rate, values = wavfile.read(io.BytesIO(raw))
        prepared.sha256 = hashlib.sha256(raw).hexdigest()
        if values.dtype.kind == "i":
            values = values.astype(np.float32) / (2 ** (8 * values.dtype.itemsize - 1))
        elif values.dtype == np.uint8:
            values = (values.astype(np.float32) - 128) / 128
        elif values.dtype.kind == "f":
            values = values.astype(np.float32)
        else:
            raise AlignmentError("noise_unsupported_encoding")
        if values.ndim == 2:
            values = values.mean(axis=1)
        if values.ndim != 1 or not values.size or not np.isfinite(values).all():
            raise AlignmentError("noise_invalid_samples")
        if rate != sample_rate:
            factor = math.gcd(int(rate), sample_rate)
            values = resample_poly(
                values, sample_rate // factor, int(rate) // factor
            ).astype(np.float32)
        values = np.ascontiguousarray(values, dtype=np.float32)
        if not values.size or np.max(np.abs(values)) < 0.01:
            raise AlignmentError("noise_below_rms_threshold")
        values.flags.writeable = False
        prepared.samples = values
    except AlignmentError as exc:
        prepared.failure_reason = str(exc)
    except (OSError, ValueError, EOFError, struct.error):
        prepared.failure_reason = "noise_asset_unavailable_or_invalid"
    prepared.preparation_seconds = time.perf_counter() - started
    return prepared


def hard_dns_noise(samples, target, seed, noise, *, output=None):
    """Mix source hard (-20 dB) noise locally, then clip and quantize PCM16.

    The source recipe's epsilon, threshold and 12-offset selection are retained.
    Ineligible fragments explicitly fail instead of using a quiet fallback.
    """
    if noise is None or noise.failure_reason or noise.samples is None:
        raise AlignmentError(noise.failure_reason if noise else "noise_not_prepared")
    start, end = target
    length = end - start
    values = noise.samples
    rng = np.random.default_rng(seed)
    repetitions = max(1, math.ceil(length / len(values)))
    extended_count = len(values) * repetitions
    max_offset = extended_count - length
    segment = None
    for _ in range(12):
        offset = int(rng.integers(0, max_offset + 1)) if max_offset else 0
        if repetitions == 1:
            candidate = values[offset : offset + length]
        else:
            candidate = values[(offset + np.arange(length)) % len(values)]
        rms = float(np.sqrt(np.mean(candidate**2)))
        if rms >= 0.01:
            segment = candidate
            break
    if segment is None:
        raise AlignmentError("noise_no_valid_fragment")
    clean = samples[start:end].astype(np.float32) / 32768
    clean_rms = float(np.sqrt(np.mean(clean**2)))
    scale = (clean_rms + 1e-8) * 10 / (rms + 1e-8)
    scaled = segment * scale
    mixed = clean + scaled
    result = samples.copy() if output is None else output
    result[start:end] = np.clip(
        np.rint(np.clip(mixed, -1.0, 1.0) * 32768), -32768, 32767
    ).astype("<i2")
    return result, {
        "noise_source": noise.source,
        "noise_sha256": noise.sha256,
        "noise_offset_samples": offset,
        "noise_segment_samples": length,
        "noise_repetitions": repetitions,
        "noise_sample_rate": noise.sample_rate,
        "noise_scale": scale,
        "noise_segment_rms": rms,
        "clean_target_rms": clean_rms,
        "preclip_noise_rms": float(np.sqrt(np.mean(scaled**2))),
        "target_snr_db": -20,
        "selected_by_threshold": True,
        "clipped_samples": int(np.count_nonzero((mixed > 1.0) | (mixed < -1.0))),
        "clipping_range": [-1.0, 1.0],
        "applied_range_samples": [start, end],
    }


def degrade_utterance(
    samples, text, keyword, sample_rate, seed, record, *, mode="dropchunk", noise=None
):
    """Construct one private evidence record and a whole degraded waveform."""
    started = time.perf_counter()
    record.update(
        {
            "sample_rate": sample_rate,
            "sample_count": len(samples),
            "script_text": text,
            "keyword": keyword.model_dump()
            if isinstance(keyword, KeywordAnnotation)
            else None,
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
        annotations = (
            keyword if isinstance(keyword, list) else [keyword] if keyword else []
        )
        if not annotations:
            raise AlignmentError("missing_keyword_annotation")
        if len(annotations) > DEFAULT_ACOUSTIC_KEYWORD_LIMIT:
            raise AlignmentError("too_many_keyword_annotations")
        record["keyword_annotations"] = [
            item.model_dump(exclude_none=True) for item in annotations
        ]
        resolved = []
        for annotation in annotations:
            original = annotation.model_dump(exclude_none=True)
            if annotation.occurrence is not None:
                cursor = 0
                for _ in range(annotation.occurrence + 1):
                    position = text.find(annotation.text, cursor)
                    if position < 0:
                        raise AlignmentError("keyword_occurrence_not_found")
                    cursor = position + len(annotation.text)
                annotation = KeywordAnnotation(
                    text=annotation.text, start_char=position, end_char=cursor
                )
            if (
                not (0 <= annotation.start_char < annotation.end_char <= len(text))
                or text[annotation.start_char : annotation.end_char] != annotation.text
            ):
                raise AlignmentError("keyword_text_position_mismatch")
            resolved.append((annotation, original))
        resolved.sort(key=lambda item: item[0].start_char)
        if any(
            a.end_char > b.start_char for (a, _), (b, _) in zip(resolved, resolved[1:])
        ):
            raise AlignmentError("overlapping_keyword_annotations")
        record["keyword_annotation"] = resolved[0][1]
        record["keyword"] = resolved[0][0].model_dump(exclude_none=True)
        record["keywords"] = [
            item.model_dump(exclude_none=True) for item, _ in resolved
        ]
        alignment_start = time.perf_counter()
        try:
            response = forced_alignment(samples, sample_rate, text)
        finally:
            record["timing"]["alignment_seconds"] = (
                time.perf_counter() - alignment_start
            )
        record["alignment_response"] = response
        targets = []
        for index, (annotation, original) in enumerate(resolved):
            bounds = aligned_samples(
                response, text, annotation, sample_rate, len(samples)
            )
            if targets and targets[-1]["target_samples"][1] > bounds[0]:
                raise AlignmentError("overlapping_aligned_targets")
            targets.append(
                {
                    "id": f"target-{index + 1:02d}",
                    "keyword": annotation.model_dump(exclude_none=True),
                    "keyword_annotation": original,
                    "target_samples": list(bounds),
                    "seed": seed + index,
                }
            )
        record["targets"] = targets
        record["target_samples"] = targets[0]["target_samples"]
        record["target_ranges_samples"] = [item["target_samples"] for item in targets]
        operation_start = time.perf_counter()
        try:
            # One waveform copy for all disjoint spans. Kernels still read clean PCM.
            result = samples.copy()
            for target in targets:
                bounds = target["target_samples"]
                if mode == "dns_noise":
                    _, details = hard_dns_noise(
                        samples, bounds, target["seed"], noise, output=result
                    )
                else:
                    _, details = hard_dropchunk(
                        samples, bounds, target["seed"], output=result
                    )
                target.update(details)
            # Preserve the historical single-target fields for existing consumers.
            if len(targets) == 1:
                record.update(details)
        finally:
            record["timing"]["degradation_seconds"] = (
                time.perf_counter() - operation_start
            )
        for target in targets:
            a, b = target["target_samples"]
            changed = np.flatnonzero(samples[a:b] != result[a:b]) + a
            if not changed.size:
                raise AlignmentError(
                    "no_effective_modification"
                    if len(targets) == 1
                    else "no_effective_target_modification"
                )
            breaks = np.flatnonzero(np.diff(changed) > 1)
            target["modified_intervals_samples"] = [
                [int(left), int(right) + 1]
                for left, right in zip(
                    np.r_[changed[0], changed[breaks + 1]],
                    np.r_[changed[breaks], changed[-1]],
                )
            ]
        samples = result
        record["status"] = "prepared"
    except (AlignmentError, TypeError, AttributeError) as exc:
        # No partially modified waveform is returned if any target fails.
        for item in record.get("targets", []):
            item["modified_intervals_samples"] = []
        record["failure_reason"] = (
            str(exc) if isinstance(exc, AlignmentError) else "alignment_invalid_schema"
        )
    if noise is not None:
        record["timing"]["noise_preparation_seconds"] = noise.preparation_seconds
        record["noise_preparation"] = noise.model_dump()
    record["timing"].setdefault("alignment_seconds", 0.0)
    record["timing"].setdefault("degradation_seconds", 0.0)
    record["timing"]["tts_postprocess_seconds"] = time.perf_counter() - started
    record["degraded_sha256"] = hashlib.sha256(samples.tobytes()).hexdigest()
    return samples
