"""Precise user pauses and auxiliary input over the real duplex PCM clock."""

from __future__ import annotations

import asyncio
import copy
import re

import numpy as np
import soundfile as sf

from .recording import Track, TurnRecorder
from .scenarios import load_clip
from .streaming import update_spoken_text


def environment_trigger_evidence(stage, heard):
    """Frozen environment stimuli may use an explicit audible action anchor."""
    if stage["kind"] != "environment":
        return ""
    for pattern in stage.get("audio_trigger_patterns", []):
        match = re.search(pattern, heard)
        if match:
            return match.group()
    return ""


class ScenarioAudioMixin:
    """Execute prepared stimuli and pauses through the same delivery-receipt path as speech."""

    def input_frame_sent(self, turn, raw, at, source_samples, voice):
        """Account for delivered fixture segments and gaps using the actual send clock."""
        if turn.get("kind") == "environment":
            if getattr(self, "environment_track", None) is None:
                self.environment_track = Track(self.origin, self.user_track.rate)
            self.environment_track.add(at, raw, segment=turn["recording_segment"])
        if "fixture_samples_sent" not in turn:
            return
        before = turn["fixture_samples_sent"]
        turn["fixture_samples_sent"] += source_samples
        after = turn["fixture_samples_sent"]
        for segment in turn["speech_segments"]:
            left, right = segment["output_start_sample"], segment["output_end_sample"]
            if before <= left < after:
                self.rec.emit(
                    "scenario_segment_start",
                    turn=turn["index"],
                    stage_id=turn["pending_intent"]["id"],
                    segment_id=segment["id"],
                    text=segment["text"],
                    asset_id=segment["asset_id"],
                    actor=turn.get("kind", "main"),
                    actual_ns=at,
                )
                segment["first_frame_ns"] = at
            if before < right <= after:
                segment["last_frame_ns"] = at
                segment["delivered"] = True
                self.rec.emit(
                    "scenario_segment_end",
                    turn=turn["index"],
                    stage_id=turn["pending_intent"]["id"],
                    segment_id=segment["id"],
                    text=segment["text"],
                    actual_ns=at + 20_000_000,
                )
                event = turn.get("_segment_done", {}).get(segment["id"])
                if event:
                    event.set()
        for gap in turn.get("scheduled_gaps", []):
            if before <= gap["start_sample"] < after:
                gap["actual_start_ns"] = at
                self.rec.emit("scenario_gap_start", turn=turn["index"], **gap)
            if before < gap["end_sample"] <= after:
                gap["actual_end_ns"] = at + 20_000_000
                self.rec.emit("scenario_gap_end", turn=turn["index"], **gap)
                event = turn.get("_gap_done", {}).get(gap["after_segment"])
                if event:
                    event.set()

    async def play_prepared(self, action, state, decision_at, usage):
        """Enqueue immutable stimulus PCM and track actual delivery across controlled pauses."""
        stage = state["pending_intent"]
        auxiliary = stage["kind"] in {"environment", "side_speech"}
        actor = (
            "environment"
            if stage["kind"] == "environment"
            else "side"
            if auxiliary
            else "main"
        )
        self.speech_attempt += 1
        key = f"scenario-{self.speech_attempt}"
        index = len(self.turns) if auxiliary else len(self.turns) + 1
        parts = stage.get("segments") or [
            {"id": "stimulus", "asset": stage["asset"], "gap_after_ms": 0}
        ]
        chunks, segments, gaps, cursor = [], [], [], 0
        for i, part in enumerate(parts):
            pcm, metadata = load_clip(part["asset"])
            gain = float(stage.get("gain", 1))
            if gain != 1:
                x = np.frombuffer(pcm, dtype="<i2").astype(np.float64) * gain
                if np.max(np.abs(x)) > 32767:
                    raise ValueError("scenario auxiliary gain clips")
                pcm = x.astype("<i2").tobytes()
            n = len(pcm) // 2
            segments.append(
                {
                    "segment": i,
                    "id": part["id"],
                    "text": metadata["text"],
                    "asset_id": part["asset"],
                    "asset_sha256": metadata["sha256"],
                    "output_start_sample": cursor,
                    "output_end_sample": cursor + n,
                    "yield_if_target": part.get("yield_if_target", False),
                }
            )
            chunks.append(pcm)
            cursor += n
            pause = int(part.get("gap_after_ms", 0))
            if pause % 20:
                raise ValueError("scenario gap must align to 20ms clock")
            if pause:
                gaps.append(
                    {
                        "after_segment": part["id"],
                        "planned_ms": pause,
                        "start_sample": cursor,
                        "end_sample": cursor + pause * 16,
                    }
                )
                cursor += pause * 16
        turn = {
            "index": index,
            "action": action,
            "kind": actor,
            "non_main_input": auxiliary,
            "recording_segment": key,
            "audio_file": f"{key}.wav" if auxiliary else f"user-{index:02d}.wav",
            "marks": {"decision": decision_at},
            "decision_usage": usage,
            "generation_usage": {},
            "text": "",
            "source": "frozen_scenario_audio",
            "planned_text": "".join(s["text"] for s in segments),
            "pending_intent": copy.deepcopy(stage),
            "agenda_snapshot": state.get("agenda"),
            "heard_at_decision": state["currently_heard_agent"],
            "speech_gate_state": state,
            "speech_segments": segments,
            "scheduled_gaps": gaps,
            "sent_source_samples": 0,
            "fixture_samples_sent": 0,
            "sent_pcm": bytearray(),
            "done": asyncio.Event(),
            "_segment_done": {s["id"]: asyncio.Event() for s in segments},
            "_gap_done": {g["after_segment"]: asyncio.Event() for g in gaps},
        }
        collection = self.auxiliary_inputs if auxiliary else self.turns
        collection.append(turn)
        self.busy = True
        tagged = TurnRecorder(self, turn)
        tagged.emit(
            "user_action", action=action, source="frozen_scenario_audio", actor=actor
        )
        self.rec.emit(
            "scenario_audio_start",
            stage_id=stage["id"],
            actor=actor,
            turn=index,
            asset_ids=[p["asset"] for p in parts],
            gain=stage.get("gain", 1),
            model_generation_used=False,
            counts_as_formal_turn=not auxiliary,
        )
        try:
            queued_any = False
            for i, (part, pcm) in enumerate(zip(parts, chunks)):
                if part.get("yield_if_target"):
                    evidence_end = next(
                        (
                            s.get("last_frame_ns", 0)
                            for s in segments
                            if s["id"] == stage.get("yield_after_segment")
                        ),
                        0,
                    )
                    if evidence_end and self.playback.last_voice_ns >= evidence_end:
                        turn["yielded_to_target"] = True
                        turn["omitted_segments"] = [p["id"] for p in parts[i:]]
                        self.rec.emit(
                            "scenario_user_yield",
                            stage_id=stage["id"],
                            before_segment=part["id"],
                            target_last_voice_ns=self.playback.last_voice_ns,
                            evidence_complete_ns=evidence_end + 20_000_000,
                        )
                        break
                for offset in range(0, len(pcm), 640):
                    if turn.get("abort_before_voice"):
                        break
                    await self.user_queue.put((turn, pcm[offset : offset + 640]))
                    queued_any = True
                if turn.get("abort_before_voice"):
                    break
                await turn["_segment_done"][part["id"]].wait()
                pause = int(part.get("gap_after_ms", 0))
                for _ in range(pause // 20):
                    await self.user_queue.put((turn, bytes(640)))
                if pause:
                    # The target may respond during the last queued silence frames.
                    # Check yielding only after the pause has actually been sent.
                    await turn["_gap_done"][part["id"]].wait()
            if queued_any:
                await self.user_queue.put((turn, None))
                await turn["done"].wait()
            if turn.get("abort_before_voice"):
                collection.remove(turn)
                self.aborted_turns.append(turn)
                self.rec.emit(
                    "scenario_audio_not_sent",
                    stage_id=stage["id"],
                    reason="floor_changed",
                )
                return False
            update_spoken_text(turn)
            turn["text"] = turn["spoken_text"]
            if not auxiliary:
                self.history.append({"role": "user", "content": turn["text"]})
                self.progress.user_finished(turn)
            self.rec.emit(
                "scenario_audio_end",
                stage_id=stage["id"],
                actor=actor,
                turn=index,
                delivered_segments=[s["id"] for s in segments if s.get("delivered")],
                yielded=turn.get("yielded_to_target", False),
                text=turn["text"],
            )
            return True
        finally:
            self.busy = False
            turn.pop("_segment_done", None)
            turn.pop("_gap_done", None)
            if turn["sent_pcm"]:
                self.save_sent_audio(turn)
            if not turn["marks"].get("user_audio_complete"):
                turn["interrupted"] = True
                update_spoken_text(turn)
                turn["text"] = turn["spoken_text"]

    def additional_summary(self):
        """Export auxiliary-input evidence and separate environment audio after track saving."""
        extra = {"scenario_audio": [self.serial_turn(t) for t in self.auxiliary_inputs]}
        track = getattr(self, "environment_track", None)
        if track is not None:
            path = self.rec.path / "user.wav"
            combined, rate = sf.read(path, dtype="int16")
            environment = track.render(len(combined))[: len(combined)]
            primary = np.clip(
                combined.astype(np.int32) - environment.astype(np.int32), -32768, 32767
            ).astype("<i2")
            sf.write(
                self.rec.path / "environment.wav", environment, rate, subtype="PCM_16"
            )
            sf.write(
                self.rec.path / "primary-user.wav", primary, rate, subtype="PCM_16"
            )
            extra["input_track_note"] = (
                "user.wav is the actual combined input; primary-user.wav and environment.wav separate main voice and recorded/synthesized environment."
            )
            extra["environment_layout"] = track.manifest()
        return extra
