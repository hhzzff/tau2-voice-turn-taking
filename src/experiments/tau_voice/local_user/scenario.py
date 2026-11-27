"""Execute bounded scenario events over the existing audited duplex audio transport."""

from __future__ import annotations

import asyncio
import copy

from .agenda import private_answer_text
from .audio import first_effective
from .feedback import feedback_actions
from .llm import ScenarioSpeaker, requested_private_fields, review
from .policy import FloorController, FloorState, effective_action, evidence_still_valid
from .recording import now
from .runtime import UserRuntime
from .scene_audio import ScenarioAudioMixin, environment_trigger_evidence


class ScenarioRuntime(ScenarioAudioMixin, UserRuntime):
    """Execute frozen stages using audio floor permission and revisable B evidence."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.floor = FloorController()
        self.item_ends, self.item_voice = {}, {}
        self.own_active = False
        self.main_end_ns = 0
        self.reply_after_ns = 0
        self.last_feedback_end = 0
        self._floor_log_key = None
        self._feedback_turns = set()
        self._last_review_key = None
        self._next_review_ns = 0
        self._recoveries = 0
        self.auxiliary_inputs = []
        self._scheduled_release_ns = 0

    def scenario_speaker(self, turn, state):
        """Create an expression source for the accepted stage and reserved turn."""
        return ScenarioSpeaker(self, turn, state)

    def own_audio_event(self, kind, turn, at):
        """Pause or resume the decision clock using actual user voice transmission events."""
        feedback = turn.get("kind") == "backchannel" or turn.get(
            "non_main_input", False
        )
        if kind == "user_first_voice_tx":
            self.own_active = True
            if turn.get("kind") == "backchannel":
                self._feedback_turns.add(self.sent_turn)
            self.rec.emit(
                "decision_clock_paused",
                turn=turn["index"],
                speech_kind="feedback" if feedback else "main",
            )
        else:
            self.own_active = False
            if feedback:
                self.last_feedback_end = at
            else:
                self.main_end_ns = at
                # A reply heard during a held utterance does not answer its final
                # segment. Only actual voice extending beyond our last voice can
                # release the following turn. No task semantics enter this gate.
                self.reply_after_ns = (
                    turn["marks"].get("user_last_voice_tx", at) + 20_000_000
                )
            self._next_review_ns = at + 500_000_000
            self.rec.emit(
                "decision_clock_resumed",
                turn=turn["index"],
                anchor_ns=at,
                speech_kind="feedback" if feedback else "main",
            )
        self.control_wakeup.set()

    def begin_agent(self, item):
        """Queue a newly audible item and wake the scenario controller."""
        super().begin_agent(item)
        self.control_wakeup.set()

    def agent_frame(self, item, pcm, at):
        """Observe played PCM and track the last effective voice sample for floor decisions."""
        super().agent_frame(item, pcm, at)
        if first_effective(pcm, self.cfg.audio.effective_rms) is not None:
            self.item_voice[item.item_id] = at

    def end_agent(self, item, reason):
        """Record playback end, queue its ASR final and wake the controller."""
        self.item_ends[item.item_id] = (now(), reason)
        super().end_agent(item, reason)
        self.control_wakeup.set()

    def floor_state(self, *, ignore_reservation=False):
        """Snapshot audible replies, buffered samples and speech reservations at one instant."""
        at = now()
        items = list(self.playback.items.values())
        current = [
            i for i in items if i.for_turn == self.sent_turn and self.sent_turn > 0
        ]
        heard = [
            i
            for i in current
            if i.played and self.item_voice.get(i.item_id, 0) >= self.reply_after_ns
        ]
        ends = [
            (self.item_ends[i.item_id][0], self.item_ends[i.item_id][1])
            for i in heard
            if i.item_id in self.item_ends
        ]
        end, reason = max(ends, default=(0, "absent"))
        open_items = [i for i in items if not i.ended and not i.cancelled]
        buffered = sum(i.buffered_samples for i in open_items)
        last_voice = max(
            (self.item_voice.get(i.item_id, 0) for i in current), default=0
        )
        any_voice = max(self.item_voice.values(), default=0)
        ambiguous = any(
            i.for_turn == 0 and (i.played or i.buffered_samples) for i in open_items
        )
        return FloorState(
            at_ns=at,
            own_turn=self.sent_turn,
            own_active=self.own_active,
            reserved=self.busy and not ignore_reservation,
            own_end_ns=self.main_end_ns,
            reply_seen=bool(heard),
            voice_active=bool(any_voice and at - any_voice < 60_000_000),
            response_open=bool(open_items),
            buffered_samples=buffered,
            last_voice_ns=last_voice,
            playback_end_ns=end,
            normal_end=reason == "complete",
            ambiguous_audio=ambiguous,
        )

    def snapshot(self):
        """Extend heard context with audio floor permission and the current main-turn reply."""
        state = super().snapshot()
        audio = self.floor_state(ignore_reservation=True)
        permission = self.floor.evaluate(audio)
        current = [
            i
            for i in self.playback.items.values()
            if i.for_turn == self.sent_turn
            and i.played
            and self.item_voice.get(i.item_id, 0) >= self.reply_after_ns
        ]
        texts = [
            self.observations.get(i.item_id, {}).get("text", "")
            for i in current
            if self.playback.responses.get(i.response_id, {}).get("speech_kind")
            not in {"environment", "side", "backchannel"}
        ]
        state.update(
            floor_permission=permission.model_dump(),
            audio_state=audio.model_dump(),
            current_reply_text="\n".join(t for t in texts if t.strip()),
            heard_current_main_reply=audio.reply_seen,
            current_main_turn=self.sent_turn,
            allow_formal_interrupt=permission.kind == "LISTEN",
            recovery_due=permission.kind == "RECOVER",
        )
        if self.sent_turn in self._feedback_turns and state.get("feedback_control"):
            state["feedback_control"].update(
                available=False, unavailable_reason="formal_turn_limit"
            )
        return state

    def speech_valid(self, action, state):
        """Recheck stage version and audible trigger evidence immediately before speech."""
        if state.get("agenda", {}).get("version") != self.progress.version:
            return False
        latest = self.snapshot()
        if (
            state.get("pending_intent", {}).get("trigger") == "scheduled_release"
            and self._scheduled_release_ns
            and now() >= self._scheduled_release_ns
        ):
            return not self.own_active
        permission = latest["floor_permission"]
        if action in {"START", "RESPOND"}:
            return permission["kind"] in {"RESPOND", "RECOVER"}
        if action == "INTERRUPT":
            return (
                permission["kind"] == "LISTEN"
                and latest["audio_state"]["voice_active"]
                and latest["observed_item_id"] == state["observed_item_id"]
                and evidence_still_valid(
                    state.get("trigger_evidence", ""),
                    state["currently_heard_agent"],
                    latest["currently_heard_agent"],
                )
            )
        return False

    async def choose_feedback(self, state, candidates):
        # Freeze this experiment to the reviewed pool: B chooses USE only.
        """Ask B to choose an existing reviewed clip; reject invalid or empty selections."""
        candidates = [
            c
            for c in candidates
            if c["id"] not in self.cfg.backchannel.excluded_clip_ids
        ]
        if not candidates:
            raise ValueError("reviewed feedback pool is empty")
        body = self.body(
            [
                {
                    "role": "system",
                    "content": "你是倾听中的用户，从已有同音色短附和中选最自然的一条。只输出候选编号，不提问或说新内容。",
                },
                {
                    "role": "user",
                    "content": str(
                        {
                            "heard": state["currently_heard_agent"],
                            "candidates": {
                                str(i): c["text"] for i, c in enumerate(candidates)
                            },
                        }
                    ),
                },
            ]
        )
        body["regex"] = "(" + "|".join(str(i) for i in range(len(candidates))) + ")"
        start = self.rec.emit(
            "local_request", phase="backchannel_content", body=body, model_role="B"
        )
        response = await self.client.post(
            self.cfg.llm.base_url + "/chat/completions", json=body
        )
        response.raise_for_status()
        data = response.json()
        raw = data["choices"][0]["message"]["content"].strip()
        self.rec.emit(
            "local_result",
            phase="backchannel_content",
            model_role="B",
            raw=raw,
            usage=data.get("usage", {}),
            duration_ms=(now() - start) / 1e6,
        )
        if raw not in {str(i) for i in range(len(candidates))}:
            raise ValueError("invalid reviewed feedback selection")
        return {"mode": "retrieval", "id": candidates[int(raw)]["id"]}

    async def observe_stage(self, stage):
        """Collect an observation window after a stimulus without declaring semantic success."""
        started = now()
        stimulus_end = max(self.main_end_ns, self.last_feedback_end)
        if stage.get("fixed_seconds") is not None:
            previous = self.turns[-1] if self.turns else None
            stimulus_end = (
                previous["marks"].get("user_last_voice_tx", stimulus_end) + 20_000_000
                if previous
                else stimulus_end
            )
            deadline = stimulus_end + int(stage["fixed_seconds"] * 1e9)
            self.rec.emit(
                "scenario_observation_start",
                stage_id=stage["id"],
                reason=stage["need"],
                deadline_ns=deadline,
                model_wait_permitted=False,
                fixed=True,
            )
            while now() < deadline:
                self.check_target()
                await self.wait_control(min(deadline, now() + 100_000_000))
            self._scheduled_release_ns = deadline
            sample = {
                "stage_id": stage["id"],
                "reason": "fixed_window_complete",
                "duration_ms": (now() - stimulus_end) / 1e6,
                "heard": self.snapshot()["current_reply_text"],
                "outcome": "sampled_pending_review",
            }
            self.progress.samples.append(sample)
            self.rec.emit("scenario_observation_end", **sample)
            self.progress.activate_next(
                "fixed_observation_finished_not_semantic_success"
            )
            return
        deadline = stimulus_end + int(stage.get("max_seconds", 20) * 1e9)
        self.rec.emit(
            "scenario_observation_start",
            stage_id=stage["id"],
            reason=stage["need"],
            deadline_ns=deadline,
            model_wait_permitted=False,
        )
        reason = "observation_deadline"
        while now() < deadline:
            self.check_target()
            state = self.snapshot()
            audio = state["audio_state"]
            ended = audio["playback_end_ns"]
            if (
                ended > stimulus_end
                and not audio["response_open"]
                and not audio["buffered_samples"]
                and now() - ended >= int(stage.get("quiet_after_end_ms", 4000) * 1e6)
            ):
                reason = "response_finished_then_observed"
                break
            await self.wait_control(now() + 100_000_000)
        state = self.snapshot()
        sample = {
            "stage_id": stage["id"],
            "reason": reason,
            "duration_ms": (now() - started) / 1e6,
            "heard": state["current_reply_text"],
            "outcome": "sampled_pending_review",
        }
        self.progress.samples.append(sample)
        self.rec.emit("scenario_observation_end", **sample)
        self.progress.activate_next("observation_finished_not_semantic_success")

    def check_target(self):
        """Propagate a recorded target transport failure to the controller."""
        if self.transport_error is not None:
            raise RuntimeError("Target transport failed") from self.transport_error

    async def control(self):
        """Advance the local scenario while independent audio clocks keep running.

        Sample only permitted actions, recheck asynchronous B results against
        current evidence, and preserve unfinished or invalid stimuli as outcomes.
        Finishing the sampling cursor does not assert semantic task success.
        """
        invalids = 0
        while not self.stop:
            self.check_target()
            if self.progress.active is None:
                # Retain target farewell; never cut it at a fixed 1.5 seconds.
                state = self.snapshot()
                if (
                    state["floor_permission"]["kind"] in {"RESPOND", "RECOVER"}
                    and now() - self.main_end_ns > 1_500_000_000
                ):
                    self.rec.emit("scenario_finished", semantic_success_implied=False)
                    return
                await self.wait_control(now() + 100_000_000)
                continue
            stage = self.progress.active
            if stage["kind"] == "observe":
                await self.observe_stage(stage)
                continue
            state = self.snapshot()
            audio = state["audio_state"]
            permission = self.floor.evaluate(FloorState(**audio))
            key = (permission.kind, permission.reason, self.sent_turn)
            if key != self._floor_log_key:
                self.rec.emit(
                    "floor_permission",
                    permission=permission.kind,
                    reason=permission.reason,
                    epoch=permission.epoch,
                    audio=audio,
                )
                self._floor_log_key = key
            trigger = stage.get("trigger", "handoff")
            overlap = trigger == "overlap"
            if overlap and permission.kind in {"RESPOND", "RECOVER"} and self.sent_turn:
                if stage.get("setup_extension") and not stage.get("extension_used"):
                    stage["extension_used"] = True
                    extension = copy.deepcopy(stage["setup_extension"])
                    extension.update(
                        id=stage["id"] + "_setup_extension",
                        kind="question",
                        trigger="handoff",
                    )
                    state["pending_intent"] = extension
                    self.rec.emit(
                        "scenario_assisted_setup",
                        stage_id=stage["id"],
                        reason="first_reply_ended_without_trigger",
                    )
                    await self.speak("RESPOND", state, None, now(), {})
                    self._last_review_key = None
                    continue
                self.progress.error = "not_triggered:" + stage["id"]
                self.rec.emit(
                    "scenario_not_triggered",
                    stage_id=stage["id"],
                    reason="reply_finished_before_stimulus",
                )
                return
            eligible = (not overlap and permission.kind in {"RESPOND", "RECOVER"}) or (
                overlap
                and permission.kind == "LISTEN"
                and audio["voice_active"]
                and audio["reply_seen"]
                and state["current_item_played_ms"] >= 1600
                and len(state["currently_heard_agent"].strip()) >= 8
            )
            if (
                trigger == "scheduled_release"
                and now() >= self._scheduled_release_ns > 0
            ):
                eligible = True
            if stage["kind"] == "feedback" and not feedback_actions(state):
                eligible = False
            if not eligible:
                await self.wait_control(now() + 50_000_000)
                continue
            # Allow the final local ASR result before selecting private fields.
            if (
                permission.reason == "playback_complete"
                and state["observed_item_id"]
                and not self.observations.get(state["observed_item_id"], {}).get(
                    "final"
                )
                and now() - audio["playback_end_ns"] < 500_000_000
            ):
                await self.wait_control(audio["playback_end_ns"] + 500_000_000)
                continue
            if stage["kind"] == "private_answer" and not stage.get("prepared"):
                asked = await requested_private_fields(self, state, stage)
                self.progress.record_private_question(
                    stage, asked, state["current_reply_text"]
                )
                if self.progress.finish_private_answers_if_ready(stage, asked):
                    self._last_review_key = None
                    continue
                stage["required"] = [
                    stage["private_fields"][k]["answer"] for k in asked
                ]
                if stage.get("other_question"):
                    stage["required"].append(
                        "助手另外问到的‘"
                        + stage["other_question"]
                        + "’没有已定答案，坦诚还没确定，不编造条件"
                    )
                stage["facts"] = {k: stage["private_fields"][k]["value"] for k in asked}
                stage["update"] = copy.deepcopy(stage["facts"])
                stage["example"] = ""
                if stage.get("answer_mode") == "asked_fields_template":
                    stage["prepared_answer"] = private_answer_text(stage, asked)
                stage["prepared"] = True
                state = self.snapshot()
            review_key = (
                self.progress.cursor,
                self.sent_turn,
                state["current_reply_text"],
                state["observed_item_id"],
                permission.kind,
            )
            if now() < self._next_review_ns and permission.kind not in {
                "RESPOND",
                "RECOVER",
            }:
                await self.wait_control(self._next_review_ns)
                continue
            self._next_review_ns = now() + 500_000_000
            if review_key == self._last_review_key and permission.kind == "LISTEN":
                self.rec.emit(
                    "decision_tick", skip_reason="no_new_evidence", stage_id=stage["id"]
                )
                await self.wait_control(self._next_review_ns)
                continue
            self._last_review_key = review_key
            self.rec.emit("decision_tick", skip_reason=None, stage_id=stage["id"])
            allowed = (
                ["WAIT", "BACKCHANNEL" if stage["kind"] == "feedback" else "INTERRUPT"]
                if overlap
                else ["RESPOND"]
            )
            scripted_environment = stage["kind"] == "environment" and bool(
                stage.get("audio_trigger_patterns")
            )
            try:
                if scripted_environment:
                    evidence = environment_trigger_evidence(
                        stage, state["currently_heard_agent"]
                    )
                    if not evidence:
                        continue
                    wire = {
                        "met": {},
                        "action": "INTERRUPT",
                        "ready": True,
                        "evidence": evidence,
                    }
                    rid, usage = "environment-" + stage["id"], {}
                    at = self.rec.emit(
                        "scenario_environment_trigger",
                        stage_id=stage["id"],
                        evidence=evidence,
                        source="actual_played_audio_asr_pattern",
                        model_decision_used=False,
                    )
                else:
                    wire, rid, at, usage = await review(
                        self,
                        state,
                        allowed,
                        stage.get("trigger_condition", "正常交接后开口"),
                    )
            except ValueError as exc:
                invalids += 1
                self.rec.emit(
                    "scenario_invalid_decision", attempt=invalids, reason=str(exc)
                )
                if invalids >= 3:
                    raise
                self._last_review_key = None
                continue
            invalids = 0
            latest = self.snapshot()
            latest_permission = self.floor.evaluate(FloorState(**latest["audio_state"]))
            # Semantic review survives a floor change only when its evidence did not change.
            if state["current_reply_text"] == latest["current_reply_text"]:
                if not scripted_environment:
                    self.progress.update_review(wire["met"], rid)
            else:
                self.rec.emit(
                    "decision_stale", request_id=rid, reason="review_evidence_changed"
                )
                # Ongoing speech may extend unrelated text. Keep a still-valid
                # quoted trigger while discarding the stale semantic judgment.
                if not (
                    overlap
                    and evidence_still_valid(
                        wire["evidence"],
                        state["currently_heard_agent"],
                        latest["currently_heard_agent"],
                    )
                ):
                    self._last_review_key = None
                    continue
            proposed = wire["action"]
            choice = effective_action(proposed, latest_permission)
            if (
                trigger == "scheduled_release"
                and now() >= self._scheduled_release_ns > 0
            ):
                choice = (
                    "INTERRUPT" if latest["audio_state"]["voice_active"] else "RESPOND"
                )
            if overlap:
                if (
                    not wire["ready"]
                    or proposed not in allowed
                    or proposed == "WAIT"
                    or (stage.get("variants") and wire.get("variant") == "none")
                    or not evidence_still_valid(
                        wire["evidence"],
                        state["currently_heard_agent"],
                        latest["currently_heard_agent"],
                    )
                ):
                    choice = "WAIT"
                elif latest_permission.kind != "LISTEN":
                    choice = "WAIT"
            self.rec.emit(
                "decision_result",
                request_id=rid,
                choice=choice,
                model_action=proposed,
                effective_action=choice,
                floor_reason=latest_permission.reason,
                floor_epoch=latest_permission.epoch,
                met=wire["met"],
                ready=wire["ready"],
                evidence=wire["evidence"],
                turn=len(self.turns) + 1,
            )
            if choice == "WAIT":
                continue
            latest.update(decision_request_id=rid, trigger_evidence=wire["evidence"])
            if stage.get("variants") and wire.get("variant") != "none":
                projected = copy.deepcopy(stage)
                projected.update(stage["variants"][wire["variant"]])
                projected["selected_variant"] = wire["variant"]
                latest["pending_intent"] = projected
                self.rec.emit(
                    "scenario_variant_selected",
                    stage_id=stage["id"],
                    variant=wire["variant"],
                )
            if latest_permission.kind == "RECOVER":
                if self._recoveries >= 1 or not self.progress.last_sent:
                    self.progress.error = "no_reply_after_recovery"
                    return
                retry = copy.deepcopy(self.progress.last_sent["stage"])
                retry["need"] = (
                    "对方没有回复，简短询问是否听到并重述上一条重点：" + retry["need"]
                )
                latest["pending_intent"] = retry
                self._recoveries += 1
                self.rec.emit(
                    "scenario_recovery", stage_id=retry["id"], attempt=self._recoveries
                )
                await self.speak("RESPOND", latest, None, at, usage)
                self._last_review_key = None
                continue
            self._recoveries = 0
            if choice == "BACKCHANNEL":
                before = len(self.backchannels)
                await self.backchannel(choice, latest, at, usage)
                if len(self.backchannels) > before:
                    self.progress.activate_next("feedback_sent_not_new_formal_turn")
                else:
                    self._last_review_key = None
            elif stage["kind"] in {"held_speech", "environment", "side_speech"}:
                if await self.play_prepared(choice, latest, at, usage):
                    self.progress.activate_next(
                        "scheduled_audio_delivered_not_semantic_success"
                    )
                else:
                    self._last_review_key = None
            else:
                before = len(self.turns)
                await self.speak(choice, latest, None, at, usage)
                if len(self.turns) > before and self.turns[-1]["marks"].get(
                    "user_audio_complete"
                ):
                    self.progress.activate_next(
                        "planned_stimulus_delivered_not_semantic_success"
                    )
                else:
                    self._last_review_key = None
            await asyncio.sleep(0)
