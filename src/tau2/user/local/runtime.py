"""Transport-independent local user: observation, control, expression, receipts."""

from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import aclosing

from .agenda import ScenarioAgenda
from .audio import first_effective
from .backends import QwenTTS
from .feedback import FeedbackPool, feedback_actions, feedback_control
from .local_asr import XASR
from .observation import PlaybackObservation
from .recording import Frame, Track, TurnRecorder, now
from .streaming import SpeechStream, update_spoken_text


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


class UserRuntime:
    """A user owns its agenda and PCM queue, never a target WebSocket."""

    def __init__(self, cfg, case, rec, client, *, asr=None):
        self.cfg, self.case, self.rec, self.client = cfg, case, rec, client
        self.asr = asr or XASR(cfg, rec)
        self.playback = PlaybackObservation()
        self.transport_error = None
        self.history, self.turns, self.observations, self.segment_items = [], [], {}, {}
        self.asr_queue, self.user_queue = (
            asyncio.Queue(maxsize=150),
            asyncio.Queue(maxsize=32),
        )
        self.sent_turn = self.speech_attempt = 0
        self.aborted_turns, self.backchannels = [], []
        self.responded_turns = set()
        self.last_heard_item = self.current_asr_item = None
        self.busy = self.stop = False
        self.intent_ready = {}
        self.feedback_counts, self.feedback_recent = {}, []
        self.last_feedback_ns = 0
        self.feedback_library = FeedbackPool(cfg) if cfg.backchannel.enabled else None
        self.control_wakeup = asyncio.Event()
        self.progress = ScenarioAgenda(case, cfg.agenda, client, rec)
        self.tasks = []
        self._active = self._ending = None
        self._tail_frames = 0
        self._sequence = self._acked_sequence = 0
        self._pending_packet = None

    async def start(self):
        """Initialize ASR before the live clock starts."""
        await self.asr.start()

    def begin_live(self, origin):
        """Start independent ASR and control tasks after transport readiness."""
        self.origin = origin
        self.user_track = Track(origin, self.cfg.audio.sample_rate)
        self.agent_track = Track(origin, self.cfg.audio.sample_rate)
        self.tasks = [
            asyncio.create_task(coro()) for coro in (self.decode, self.transcripts)
        ]
        self.control_task = asyncio.create_task(self.control())
        self.tasks.append(self.control_task)

    def poll_output(self):
        """Return one ready frame without awaiting inference or transport."""
        if self._pending_packet is not None:
            raise RuntimeError("Previous output frame has no transmission receipt")
        pcm, source_samples = bytes(640), 0
        if not self.user_queue.empty():
            self._active, value = self.user_queue.get_nowait()
            if value is None:
                self._ending, self._tail_frames = self._active, 4
            else:
                source_samples = len(value) // 2
                if not 0 < len(value) <= 640 or len(value) % 2:
                    raise ValueError("Outgoing frame is not PCM16/20ms")
                pcm = value.ljust(640, b"\0")
        turn = self._active
        voice = first_effective(pcm, self.cfg.audio.effective_rms) is not None
        if turn and voice and "user_first_voice_tx" not in turn["marks"]:
            if turn.get("kind") == "backchannel":
                state = turn["speech_gate_state"]
                latest = self.snapshot()
                valid = (
                    latest["floor_permission"]["kind"] == "LISTEN"
                    and latest["audio_state"]["voice_active"]
                    and latest["observed_item_id"] == state["observed_item_id"]
                )
            else:
                valid = self.speech_valid(
                    turn["action"], turn.get("speech_gate_state", {})
                )
            if not valid:
                turn["abort_before_voice"] = True
                self.rec.emit("speech_stale_at_transport", turn=turn["index"])
        if turn and turn.get("abort_before_voice"):
            pcm, source_samples, voice = bytes(640), 0, False
        self._sequence += 1
        packet = {
            "sequence": self._sequence,
            "pcm": pcm,
            "source_samples": source_samples,
            "voice": voice,
            "turn": turn,
            "ending": self._ending,
        }
        self._pending_packet = packet
        return packet

    def acknowledge_delivery(self, sequence, sent_ns):
        """Commit only after the target transport submitted this exact frame."""
        if sequence <= self._acked_sequence:
            return  # Retransmitted receipt is idempotent.
        packet = self._pending_packet
        if (
            packet is None
            or packet["sequence"] != sequence
            or sequence != self._acked_sequence + 1
        ):
            raise ValueError("Out-of-order delivery receipt")
        self._acked_sequence = sequence
        self._pending_packet = None
        turn, pcm, voice = packet["turn"], packet["pcm"], packet["voice"]
        if turn:
            turn["sent_pcm"].extend(pcm)
            turn["sent_source_samples"] = (
                turn.get("sent_source_samples", 0) + packet["source_samples"]
            )
            update_spoken_text(turn)
            self.user_track.add(
                sent_ns, pcm, segment=turn.get("recording_segment", turn["index"])
            )
            if voice:
                if turn.get("kind") != "backchannel" and not turn.get("non_main_input"):
                    self.sent_turn = turn["index"]
                if "user_first_voice_tx" not in turn["marks"]:
                    TurnRecorder(self, turn).emit(
                        "user_first_voice_tx", sent_ns=sent_ns
                    )
                turn["marks"]["user_last_voice_tx"] = sent_ns
            self.rec.emit(
                "user_frame_delivered",
                sequence=sequence,
                turn=turn["index"],
                source_samples=packet["source_samples"],
                total_source_samples=turn["sent_source_samples"],
                pcm_sha256=hashlib.sha256(pcm).hexdigest(),
                voice=voice,
                sent_ns=sent_ns,
                delivery_boundary="client_transport_submit",
            )
            self.input_frame_sent(turn, pcm, sent_ns, packet["source_samples"], voice)
        if packet["ending"] is not None:
            self._tail_frames -= 1
            if self._tail_frames == 0:
                TurnRecorder(self, self._ending).emit(
                    "user_audio_complete", sent_ns=sent_ns
                )
                self._ending["done"].set()
                self._active = self._ending = None

    async def close(self):
        """Cancel generation before a new session can reuse the transport."""
        self.stop = True
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.asr.close()
        for turn in [
            *self.turns,
            *self.backchannels,
            *self.aborted_turns,
            *getattr(self, "auxiliary_inputs", []),
        ]:
            if turn.get("sent_pcm"):
                self.save_sent_audio(turn)

    @staticmethod
    def serial_turn(turn):
        return {
            k: v
            for k, v in turn.items()
            if k not in {"done", "sent_pcm", "speech_gate_state"}
            and not k.startswith("_")
        }

    async def play_feedback(self, selected, state, decision_at, decision_usage):
        """Use the reviewed immutable pool; no online fallback changes the trial."""
        clip = self.feedback_library.read(selected["id"])
        latest = self.snapshot()
        if (
            not feedback_actions(latest)
            or latest["observed_item_id"] != state["observed_item_id"]
        ):
            self.rec.emit(
                "backchannel_skipped", reason="opportunity_expired_before_audio"
            )
            return
        index = len(self.backchannels) + 1
        turn = {
            "id": f"bc-{index:03}",
            "index": len(self.turns),
            "kind": "backchannel",
            "action": "BACKCHANNEL",
            "text": clip["text"],
            "source": "retrieval",
            "recording_segment": f"bc-{index:03}",
            "audio_file": f"backchannel-{index:03}.wav",
            "marks": {"decision": decision_at},
            "decision_usage": decision_usage,
            "heard_at_decision": state["currently_heard_agent"],
            "speech_gate_state": state,
            "cache_entry_id": clip["id"],
            "cache_pcm_sha256": clip["pcm_sha256"],
            "sent_pcm": bytearray(),
            "sent_source_samples": 0,
            "done": asyncio.Event(),
        }
        self.backchannels.append(turn)
        self.last_feedback_ns = now()
        item = state["observed_item_id"]
        self.feedback_counts[item] = self.feedback_counts.get(item, 0) + 1
        for offset in range(0, len(clip["pcm"]), 640):
            await self.user_queue.put((turn, clip["pcm"][offset : offset + 640]))
        await self.user_queue.put((turn, None))
        await turn["done"].wait()
        self.feedback_recent = [*self.feedback_recent, clip["text"]][
            -self.cfg.backchannel.recent_avoid :
        ]
        self.feedback_library.used(clip["id"])
        self.save_sent_audio(turn)

    def body(self, messages, stream=False):
        return {
            "model": self.cfg.llm.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.cfg.llm.max_output_tokens if stream else 8,
            "stream": stream,
            **self.cfg.llm.extra_body,
        }

    def enqueue_asr(self, value):
        if self.asr_queue.full():
            self.rec.emit("audio_drop", stage="live_asr", samples=320)
            raise RuntimeError("ASR did not keep up with live audio")
        self.asr_queue.put_nowait(value)

    def begin_agent(self, item):
        self.enqueue_asr(("begin", item, None))

    def agent_frame(self, item, pcm, at):
        self.agent_track.add(at, pcm, segment=item.item_id)
        self.enqueue_asr(("frame", item, Frame(pcm, item.played, at)))

    def end_agent(self, item, reason):
        self.enqueue_asr(("end", item, reason))

    async def decode(self):
        while True:
            kind, item, value = await self.asr_queue.get()
            if kind == "begin":
                await self.asr.reset(preserve_events=True)
                self.current_asr_item = item.item_id
                self.segment_items[self.asr.segment] = item.item_id
            elif kind == "frame":
                await self.asr.feed(value)
            else:
                await self.asr.finish(
                    padding_ms=self.cfg.asr.endpoint_padding_ms, end_events=False
                )
                text = self.asr.last_text
                self.observations[item.item_id] = {
                    "text": text,
                    "revision": self.asr.revision,
                    "final": True,
                }
                self.last_heard_item = item.item_id
                metadata = self.playback.responses.get(item.response_id, {})
                accepted = value == "complete" and bool(text.strip())
                if getattr(self.playback, "causal_replies", False):
                    accepted = (
                        accepted
                        and item.complete
                        and not item.cancelled
                        and metadata.get("speech_kind") == "main"
                        and (
                            item.for_turn == self.sent_turn
                            or self.cfg.agenda.mode in {"foreground", "single"}
                        )
                        and bool(text.strip())
                    )
                if accepted:
                    self.responded_turns.add(item.for_turn)
                    if getattr(self, "progress", None):
                        self.progress.heard_reply(item.for_turn)
                if text:
                    self.history.append(
                        {
                            "role": "agent",
                            "content": text,
                            "interrupted": value != "complete",
                            "item_id": item.item_id,
                        }
                    )
                self.rec.emit(
                    "heard_agent_complete",
                    item_id=item.item_id,
                    turn=item.for_turn,
                    text=text,
                    interrupted=value != "complete",
                    counts_as_reply=bool(accepted),
                    response_id=item.response_id,
                )
                if accepted and hasattr(
                    getattr(self, "progress", None), "observe_reply"
                ):
                    self.progress.observe_reply(
                        item.item_id, item.for_turn, text, now()
                    )
                elif (
                    text
                    and self.cfg.agenda.enabled
                    and self.cfg.agenda.mode in {"foreground", "single"}
                ):
                    self.progress.observe_reply(
                        item.item_id, item.for_turn, text, now(), complete=False
                    )

    async def transcripts(self):
        async for event in self.asr.events():
            item_id = self.segment_items.get(int(event.event_id.split(":")[-1]))
            self.observations[item_id] = {
                "text": event.text,
                "revision": event.revision,
                "final": event.is_final,
            }
            self.rec.emit(
                "heard_partial",
                item_id=item_id,
                text=event.text,
                revision=event.revision,
                final=event.is_final,
                received_ns=event.received_mono_ns,
            )

    def snapshot(self):
        audible = self.playback.audible_item
        item_id = audible.item_id if audible else self.last_heard_item
        observed = self.observations.get(item_id, {})
        heard_text = observed.get("text", "")
        causal = getattr(self.playback, "causal_replies", False)
        if not audible and not heard_text and not causal:
            # Empty ASR finals/noise segments must not erase speech already heard.
            heard_text = next(
                (
                    h["content"]
                    for h in reversed(self.history)
                    if h["role"] == "agent" and h.get("content")
                ),
                "",
            )
        if causal and not audible:
            item = self.playback.items.get(item_id)
            if item is None or item.for_turn != self.sent_turn:
                heard_text = ""
        previous_spoken = next(
            (
                t
                for t in reversed(self.turns)
                if t["marks"].get("user_first_voice_tx")
                or t["marks"].get("user_audio_complete")
            ),
            None,
        )
        last_user_end = (
            previous_spoken["marks"].get("user_audio_complete", 0)
            if previous_spoken
            else 0
        )
        waiting_ms = (
            max(0, (now() - max(last_user_end, self.playback.last_voice_ns)) / 1e6)
            if last_user_end
            else 0
        )
        state = {
            "heard_history": list(self.history),
            "currently_heard_agent": heard_text,
            "agent_speaking": self.playback.speaking,
            "observed_item_id": item_id,
            "current_item_played_ms": audible.played / 16 if audible else 0,
            "own_utterances_so_far": len(self.turns),
            "max_own_utterances": self.case["max_turns"],
            "instruction": "保持围绕主题自然交流，若助手询问就回答；若需要更新条件可在说话时插入。",
            "causal_replies": causal,
            "current_turn_answered": getattr(self, "sent_turn", 0)
            in getattr(self, "responded_turns", set()),
            "allow_formal_interrupt": not (
                causal
                and previous_spoken
                and previous_spoken["action"] in {"INTERRUPT", "RECOVER"}
                and self.sent_turn not in self.responded_turns
            ),
            "waiting_after_user_ms": waiting_ms,
            "continuous_duplex": getattr(
                self.playback,
                "continuous_duplex",
                self.playback.model.startswith("gpt-live"),
            ),
            "agent_silence_ms": max(0, (now() - self.playback.last_voice_ns) / 1e6)
            if self.playback.last_voice_ns
            else 0,
        }
        if self.case.get("persona"):
            # Only completed user speech releases the next private intention.
            progress = getattr(self, "progress", None)
            facts, pending = progress.state()
            if progress:
                state["settled_exchanges"] = len(progress.exchanges)
            state.update(private_facts=facts, pending_intent=pending)
            if hasattr(progress, "view"):
                state["agenda"] = progress.view()
                if (
                    state["agenda"].get("expressed_turn") is not None
                    and not state["current_turn_answered"]
                ):
                    # The active task is awaiting its answer, not an unexpressed new intent.
                    state["allow_formal_interrupt"] = False
            if pending and pending["id"] not in self.intent_ready:
                self.intent_ready[pending["id"]] = self.rec.emit(
                    "intent_available", intent=pending
                )
        if self.cfg.backchannel.enabled:
            state["feedback_control"] = feedback_control(
                self.cfg.backchannel,
                state,
                now(),
                self.last_feedback_ns,
                self.feedback_counts,
            )
        return state

    async def backchannel(self, action, state, decision_at, decision_usage):
        self.busy = True
        try:
            candidates = self.feedback_library.retrieve(
                state["currently_heard_agent"], self.feedback_recent
            )
            selected = await self.choose_feedback(state, candidates)
            if selected is None:
                self.rec.emit(
                    "backchannel_skipped",
                    reason="invalid_content",
                    decision_ns=decision_at,
                )
                return
            await self.play_feedback(selected, state, decision_at, decision_usage)
        finally:
            self.busy = False

    async def speak(self, action, state, step, decision_at, decision_usage):
        self.speech_attempt += 1
        track_speech = self.cfg.agenda.enabled and self.cfg.agenda.mode == "single"
        evidence_saved = False
        turn = {
            "recording_segment": f"main-{self.speech_attempt}",
            "index": len(self.turns) + 1,
            "action": action,
            "marks": {"decision": decision_at},
            "decision_usage": decision_usage,
            "generation_usage": {},
            "text": "",
            "planned": step,
            "heard_at_decision": state["currently_heard_agent"],
            "agent_speaking_at_decision": state["agent_speaking"],
            "history_chars": len(
                json.dumps(state["heard_history"], ensure_ascii=False)
            ),
            "history_entries": len(state["heard_history"]),
            "pending_intent": state.get("pending_intent"),
            "agenda_snapshot": state.get("agenda"),
            "private_constraints_snapshot": state.get("private_facts")
            if state.get("agenda")
            else None,
            "controller_recovery": bool(state.get("recovery_after_silence")),
            "intent_available_ns": self.intent_ready.get(
                (state.get("pending_intent") or {}).get("id")
            ),
            "agent_played_ms_at_decision": state.get("current_item_played_ms"),
            "sent_pcm": bytearray(),
            "done": asyncio.Event(),
            "speech_gate_state": state,
        }
        if track_speech:
            turn.update(
                source="B",
                sent_source_samples=0,
                decision_request_id=state.get("decision_request_id"),
            )
        self.turns.append(turn)
        self.busy = True
        tagged = TurnRecorder(self, turn)
        tagged.emit(
            "user_action",
            action=action,
            heard=state["currently_heard_agent"],
            planned=step,
        )
        speaker = self.scenario_speaker(turn, state)
        speech = SpeechStream(
            self.cfg, speaker, QwenTTS(self.cfg, self.client, tagged), tagged.emit
        )
        queued = False
        try:
            async with aclosing(speech.audio()) as stream:
                async for pcm in stream:
                    if turn.get("abort_before_voice"):
                        break
                    if not queued and not self.speech_valid(action, state):
                        turn["abort_before_voice"] = True
                        break
                    queued = True
                    await self.user_queue.put((turn, pcm))
            turn["text"] = speech.text
            if queued:
                await self.user_queue.put((turn, None))
                await turn["done"].wait()
            if turn.get("abort_before_voice"):
                self.rec.emit(
                    "speech_stale_before_voice", turn=turn["index"], action=action
                )
                self.turns.remove(turn)
                turn["audio_file"] = f"aborted-{self.speech_attempt:03}.wav"
                self.aborted_turns.append(turn)
                return
            if track_speech:
                turn["generated_text"] = speech.text
                update_spoken_text(turn)
                turn["text"] = turn["spoken_text"]
            self.history.append({"role": "user", "content": turn["text"]})
            if self.progress:
                self.progress.user_finished(turn)
                self.rec.emit("script_progress", **self.progress.manifest())
            evidence_saved = True
        finally:
            if (
                track_speech
                and not evidence_saved
                and not turn.get("abort_before_voice")
            ):
                turn["generated_text"] = speech.text
                turn["interrupted"] = True
                update_spoken_text(turn)
                turn["text"] = turn["spoken_text"]
                if turn["marks"].get("user_first_voice_tx"):
                    if turn["text"]:
                        self.history.append(
                            {
                                "role": "user",
                                "content": turn["text"],
                                "interrupted": True,
                            }
                        )
                    self.progress.user_finished(turn)
            if turn["sent_pcm"]:
                self.save_sent_audio(turn)
            self.busy = False
            self.rec.save_json("turns.json", [self.serial_turn(t) for t in self.turns])

    def save_sent_audio(self, turn):
        filename = turn.get("audio_file", f"user-{turn['index']:02}.wav")
        self.rec.save_audio(filename, turn["sent_pcm"], self.user_track.rate)
        turn["sent_bytes"] = len(turn["sent_pcm"])
        turn["sent_sha256"] = hashlib.sha256(turn["sent_pcm"]).hexdigest()

    async def wait_control(self, deadline):
        event = getattr(self, "control_wakeup", None)
        if event is None:
            return await asyncio.sleep(max(0, (deadline - now()) / 1e9))
        try:
            await asyncio.wait_for(event.wait(), max(0, (deadline - now()) / 1e9))
        except TimeoutError:
            pass
        finally:
            event.clear()
