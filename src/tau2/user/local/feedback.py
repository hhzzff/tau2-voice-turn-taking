"""Immutable reviewed feedback pool with per-session retrieval counters."""

import hashlib
import json
import re
import wave

from .recording import now
from .scenarios import SCENE_DIR


class FeedbackPool:
    """Retrieve from the frozen enabled pool, without mutating shared assets."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.root = SCENE_DIR / "feedback"
        manifest = json.loads((self.root / "manifest.json").read_text())
        if (
            manifest["speaker"] != cfg.tts.speaker
            or manifest["reference_sha256"] != cfg.tts.reference_sha256
        ):
            raise ValueError("Feedback pool does not match the active TTS reference")
        self.rows = {r["id"]: dict(r) for r in manifest["clips"]}
        self._cache = {}
        for ident in self.rows:
            self._cache[ident] = self.read(ident)

    def retrieve(self, context, recent=()):
        def grams(value):
            value = re.sub(r"\W", "", value)
            return {value[i : i + 2] for i in range(max(0, len(value) - 1))}

        def form(value):
            return value.strip().rstrip("。！.!！ ")

        query, avoid = grams(context), {form(t) for t in recent}
        rows = [
            r
            for r in self.rows.values()
            if r["id"] not in self.cfg.backchannel.excluded_clip_ids
            and form(r["text"]) not in avoid
        ]

        def rank(row):
            previous = grams(row["context"])
            overlap = len(query & previous) / max(1, len(query | previous))
            return (-overlap, row["uses"], row["last_used"], row["id"])

        result, seen = [], set()
        for row in sorted(rows, key=rank):
            if row["text"] in seen:
                continue
            result.append(row)
            seen.add(row["text"])
            if len(result) >= self.cfg.backchannel.retrieval_limit:
                break
        return result

    def read(self, ident):
        if ident in self._cache:
            return self._cache[ident]
        row = self.rows[ident]
        with wave.open(str(self.root / row["file"]), "rb") as audio:
            if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (
                16000,
                1,
                2,
            ):
                raise ValueError("Feedback audio format mismatch")
            pcm = audio.readframes(audio.getnframes())
        if hashlib.sha256(pcm).hexdigest() != row["pcm_sha256"]:
            raise ValueError("Feedback hash mismatch")
        return {**row, "pcm": pcm}

    def used(self, ident):
        self.rows[ident]["uses"] += 1
        self.rows[ident]["last_used"] = now()


def feedback_actions(state):
    control = state.get("feedback_control", {})
    return (
        {"BACKCHANNEL"}
        if control.get("available") and state.get("agent_speaking")
        else set()
    )


def feedback_control(cfg, state, at, last_at, counts):
    item = state["observed_item_id"]
    remaining = max(0, cfg.min_interval_ms - (at - last_at) / 1e6) if last_at else 0
    reason = (
        "agent_not_speaking"
        if not state["agent_speaking"]
        else "cooldown"
        if remaining > 0
        else "segment_limit"
        if counts.get(item, 0) >= cfg.max_per_segment
        else "insufficient_played_audio"
        if state["current_item_played_ms"] < cfg.min_played_ms
        else "silence_or_turn_end"
        if state["agent_silence_ms"] > cfg.max_silence_ms
        else "insufficient_heard_content"
        if len(state["currently_heard_agent"].strip()) < 8
        else None
    )
    return {
        "available": reason is None,
        "unavailable_reason": reason,
        "cooldown_remaining_ms": round(remaining),
        "already_in_this_segment": counts.get(item, 0),
        "scope": "non_floor_taking_continuer_only",
    }
