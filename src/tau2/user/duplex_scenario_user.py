"""A Terra user that chooses duplex actions against the live conversation."""

import hashlib
import json
import time
import wave
from pathlib import Path

import numpy as np

from tau2.data_model.audio import (
    PCM_SAMPLE_RATE,
    AudioData,
    AudioEncoding,
    AudioFormat,
    audio_bytes_to_string,
)
from tau2.data_model.message import TurnTakingAction, UserMessage
from tau2.user.acoustic_clarification import degrade_utterance, pcm_wav, prepare_noise
from tau2.user.duplex_events import (
    EVENT_CATALOG,
    DuplexDecision,
    DuplexScenario,
    terra_llm_args,
)
from tau2.user.user_simulator_streaming import VoiceStreamingUserSimulator
from tau2.utils.llm_utils import generate
from tau2.voice.synthesis.synthesize import synthesize_voice
from tau2.voice.utils.audio_activity import is_audible_chunk
from tau2.voice.utils.audio_preprocessing import convert_to_pcm16, resample_audio

RUNTIME_POLICY = """You are the USER in a live voice conversation, not the assistant.
Speak the configured language and pursue your ordinary conversation goal.
Your pending duplex event TYPES are coverage goals, not a script.
Based on the actual conversation, YOU decide when the required next event is a
natural moment and exactly what to say. If an execution order is provided, trigger
only its next event; you may keep conversing to create an opportunity or wait.
If conversation_outline and current_event_guidance are supplied, follow their
shared story beats to keep cross-model conversations comparable. They constrain
subject matter, not words or timestamps. Adapt to actual heard replies, do not
claim the assistant suggested something it did not, and establish missing shared
facts naturally. You may prepare a later event without triggering it. Avoid
unnecessary tangents and repeated requests once the needed opportunity exists.
Never
pretend an event occurred merely because it was selected. Use what the assistant
actually said; do not invent an assistant mistake. Correction may revise a fact
YOU previously gave. Follow-up adds a question; clarification asks about an
unclear meaning; denial rejects a suggestion; impatience expresses urgency.

Natural interruption timing is mandatory for primary-user interruptions:
- assistant_speaking is only a physical prerequisite, never a reason to interrupt.
  A pending event, its required order, or a need to finish coverage is not a
  conversational reason either. Choose wait until a genuine opportunity arises.
- Hear enough of the current answer to understand a relevant, complete idea,
  recommendation, claim or unclear expression before reacting to it. Greetings,
  acknowledgements, sentence openings and unfinished words are insufficient.
  For example, hearing only "Sure. The fr..." is not a cooling recommendation.
- Follow-up must grow naturally from a specific point already heard. Denial must
  reject an identifiable proposal already heard; clarification must target an
  expression already heard; correction must address an actual claim or clearly
  relevant previously supplied fact. Do not anticipate the rest of a sentence.
- Repeat requests, topic switches, impatience and stop requests also need a
  credible motive in the actual exchange, such as sustained repetition,
  irrelevance, an established urgent constraint or a genuine changed need.
  Never manufacture frustration just because the assistant has begun answering.
- Prefer a local clause boundary after the relevant point, while the assistant
  is still speaking. You need not wait for the entire answer, but must not cut
  off the very explanation needed to justify your interruption. No fixed number
  of elapsed seconds establishes readiness. If the meaning is unclear, wait.
- In private reason, quote the already-heard words that motivate this specific
  interruption and explain why a real user would interrupt NOW rather than keep
  listening. If you cannot provide both, choose wait.
Background speech and incidental noise may overlap independently of the
assistant's meaning; they do not need a semantic reaction to the assistant.
Ordinary bridging turns and recall requests should follow a completed answer
and a natural turn boundary. Do not abandon a newly asked question mid-answer
or say "That makes sense" merely to advance to the next event.

Return one JSON decision with action, event_id (or null), text, continuation, expected_completion,
pause_seconds, observation_seconds, noise and reason. Schema is supplied below.
- wait: listen; do not speak. Choose this if the current content is insufficient.
- speak: an ordinary user turn (no event_id), only when the assistant is quiet.
  Keep the discussion concrete and elicit substantive advice when useful.
- prepare_event: say a natural prerequisite for the named pending event without
  executing it yet. For contextual rejection, establish which person NOT to
  answer, or that your own clearly addressed asides to another person should not
  receive replies. For turn_taking.handoff, naturally state concrete facts or
  constraints to remember, then have at least one intervening exchange before
  the later recall request. For conditional proactive interruption, agree on a unique spoken
  trigger phrase. You will decide whether/when to execute it in a later call.
- trigger_event: actually produce the selected event NOW. Interruption and user
  backchannel require the assistant to be audibly speaking. A background speaker
  may speak over the assistant; make the intended addressee clear in their words.
  Talking to others uses YOUR voice and addresses another person. Noise has no
  spoken text. Contextual rejection uses contextual_speaker=background for the
  named other speaker, or primary for your own aside to someone else; it must
  match the earlier spoken instruction. Choose either variant at runtime.
  turn_taking.handoff is a MEMORY test. Prefer asking after the current answer
  finishes at a natural turn boundary; do not interrupt a useful unfinished
  answer just to start this event. Overlap is not itself scored, but that is
  not permission to rush the conversation. Ask a
  complete request depending on earlier facts WITHOUT restating their values. Cite the earlier
  facts and intervening exchange in your private reason. Never count a generic
  new request as a memory test. After the recall request, listen to the entire
  answer before any further user speech or event. A fixed observation window
  expiring does not mean the assistant has finished its answer. The controller
  protects this listening phase; a response timeout is incomplete evidence.
  Noise has no
  spoken text; choose door_knock, phone_ring or rustle. For a within-turn pause,
  put the incomplete prefix in text, choose a pause, and complete the thought in
  continuation. Observation_seconds is a measurement window, not a future trigger.
- Proactive events: create an opportunity as the USER; never generate the
  assistant's action or assume it happened. The supplemental opportunities are
  additional to the originally selected events, not replacements. Interleave them
  where the actual topic and dialogue make them natural, not in a fixed order.
  For proactive.completion (CTC), use actual shared context to make the missing
  phrase strongly predictable, then get stuck mid-sentence: put ONLY the spoken
  incomplete prefix in text, leave continuation empty, and choose pause_seconds
  for a real hesitation (often 0.5 to 2 seconds). The missing meaning must be
  effectively determined by actual prior spoken facts or a conventional expression,
  allowing synonymous wording. A likely guess from a broad topic is insufficient:
  a picnic does not establish park rather than beach or lake. If multiple
  materially different completions remain natural, keep conversing to establish
  the referent before attempting CTC. Put the supported missing phrase in
  expected_completion; in reason quote the actual shared evidence and explain
  why alternatives are ruled out. Private user_context is not shared evidence.
  expected_completion is PRIVATE evaluator metadata and is never spoken. Do not
  ask the assistant explicitly to complete your sentence. This differs from
  rejection.pause: a non-predictive thinking pause should be waited through,
  whereas CTC deliberately offers a predictable completion.
  For proactive.backchannel, speak a coherent, detailed account long enough for
  at least backchannel_min_speech_seconds of actual speech (several connected
  sentences, typically 45-80 English words). Retain the floor with natural
  clause boundaries; do not fill with repetition, end early with a question,
  ask for agreement, or tell the assistant to say mm-hm. Continue speaking
  through any brief assistant acknowledgement; it is not a request to yield.
  Do not combine the CTC hesitation and the backchannel account in one event.
  For clarification, finish a request whose intent
  is ambiguous in this actual context. For contextual interruption, place the
  agreed phrase AT THE END of text and keep speaking in continuation afterward.
  No direct system prompts or out-of-character scoring instructions to the agent.
- finish: only when no events remain pending and all observations have finished.
  Answer an outstanding assistant question before finishing. With no pending
  event types, first have an ordinary substantive exchange with the assistant.
  Once coverage and observation are complete, finish at the next quiet moment;
  do not introduce new requests just to extend this sample.

For every ordinary primary spoken text, also annotate its essential information
in keywords: a list of up to four {text, occurrence} entries. Prefer 2–4 distinct
specific values needed to fulfil the request when present: names, places, times,
quantities, identifiers or constraints, rather than filler words. Do not invent
facts or extra speech to populate the list. A short acknowledgment may have none.
Each text is an exact substring; occurrence is its zero-based non-overlapping
match index (0 for the first). Mark the intended occurrence when a value repeats.
Entries must not overlap. Do not count character offsets; leave start_char and
end_char null. Leave the legacy single keyword field null.
This is neutral semantic annotation, never extra spoken words. A response to an
assistant question should answer naturally from your original intent and what
was actually asked, without assuming what the assistant understood.

Spoken text must be natural user language with no stage directions, test labels,
sound tags or stop tokens. Do not manufacture a fixed checklist conversation.
reason is private: cite current conversational evidence for timing and content.
When assistant_text_source is asr_of_played_audio, conversation text is recognized
from audio already delivered to you; partial ASR hypotheses can be revised. When
it is proportional_estimate, words may lag or lead the actual speech: an absent
word does not prove the assistant omitted it. Use the explicit
assistant_speaking flag for overlap timing. Without required_next_event, pending
IDs may be triggered in any order. A truncated user utterance may not have been
fully heard.
"""


class DuplexController:
    """Validate physical constraints and record actions, without choosing events."""

    def __init__(self, scenario: DuplexScenario, dt: float):
        self.scenario = scenario
        self.dt = dt
        self.pending = set(scenario.event_ids)
        self.event_order = list(scenario.event_order or scenario.event_ids)
        self.prepared = set()
        self.records = []
        self.decisions = []
        self.history = []
        self.active = None
        self.observing_until = None
        self.finished = False
        self.feedback = ""
        self.heard_assistant_audio = False

    @property
    def next_required_event(self):
        """Return the next event in the enforced sequence, if any."""
        if self.scenario.event_order is None:
            return None
        completed = len(self.records)
        return (
            self.event_order[completed] if completed < len(self.event_order) else None
        )

    def allowed(
        self,
        decision: DuplexDecision,
        assistant_speaking: bool,
    ) -> bool:
        """Reject impossible/unselected actions; never substitute a scripted one."""
        reason = ""
        if (
            self.active
            and decision.action != "wait"
            and not (
                self.scenario.acoustic_clarification is not None
                and decision.action == "speak"
                and self.active["event_id"] != "turn_taking.handoff"
            )
        ):
            reason = "The current event is still being observed; wait before any new user action."
        elif (
            decision.action in {"prepare_event", "trigger_event"}
            and decision.event_id not in self.pending
        ):
            reason = "That event is not pending; choose a selected, unexecuted event."
        elif (
            decision.action == "finish"
            and self.scenario.acoustic_clarification is not None
            and not (
                any(row["role"] == "user" for row in self.history)
                and self.heard_assistant_audio
            )
        ):
            reason = "Have an ordinary exchange with the assistant before finishing."
        elif decision.action == "finish" and (self.pending or self.active):
            reason = "Pending events or an unfinished observation remain."
        elif (
            decision.action == "trigger_event"
            and self.scenario.event_order is not None
            and decision.event_id != self.next_required_event
        ):
            reason = (
                f"Keep the required event order: trigger {self.next_required_event} next; "
                f"{decision.event_id} is later. Continue naturally or wait for the required opportunity."
            )
        elif decision.action in {"speak", "prepare_event"} and assistant_speaking:
            reason = "Ordinary user speech cannot become an unselected interruption; wait or trigger a selected event."
        elif decision.action == "trigger_event":
            trigger = EVENT_CATALOG[decision.event_id].trigger
            if trigger == "assistant_speaking" and not assistant_speaking:
                reason = "The assistant is not audibly speaking; no interruption opportunity yet."
            elif trigger == "assistant_silent" and assistant_speaking:
                reason = "This opportunity requires the assistant to be quiet."
            elif (
                decision.event_id
                in {
                    "rejection.contextual",
                    "proactive.contextual_interruption",
                    "turn_taking.handoff",
                }
                and decision.event_id not in self.prepared
            ):
                reason = "First establish this contextual event's prerequisite with prepare_event."
        self.feedback = reason
        return not reason

    def begin(self, decision, now, duration, marker, assistant_speaking):
        """Commit an event only when its actual audio is ready for playback."""
        speaker = (
            {
                "rejection.third_party": "background",
                "rejection.contextual": "background",
                "rejection.noise": "noise",
            }.get(decision.event_id, "primary")
            if decision.action == "trigger_event"
            else "primary"
        )
        if (
            decision.action == "trigger_event"
            and decision.event_id == "rejection.contextual"
        ):
            speaker = decision.contextual_speaker
        self.history.append(
            {
                "role": "user",
                "source": speaker,
                "seconds": now,
                "text": decision.text,
                "continuation": decision.continuation,
                "pause_seconds": decision.pause_seconds,
                "event_id": decision.event_id,
                "action": decision.action,
                "noise": decision.noise if speaker == "noise" else None,
            }
        )
        if decision.action == "prepare_event":
            self.prepared.add(decision.event_id)
        if decision.action == "trigger_event":
            self.pending.remove(decision.event_id)
            self.active = {
                "event_id": decision.event_id,
                "status": "triggered",
                "start_seconds": now,
                "end_seconds": now + duration,
                "marker_seconds": now + marker,
                "pause_seconds": decision.pause_seconds,
                "observation_seconds": decision.observation_seconds,
                "assistant_was_speaking": assistant_speaking,
                "stimulus": decision.model_dump(),
                "expected_behavior": EVENT_CATALOG[decision.event_id].expected_behavior,
                "prior_user_turns": [
                    dict(entry)
                    for entry in self.history[:-1]
                    if entry["role"] == "user"
                ],
                "prior_conversation": [dict(entry) for entry in self.history[:-1]],
                "observations": [],
                "user_stopped_for_agent": False,
            }
            if decision.event_id == "turn_taking.handoff":
                self.active["memory_response_completion"] = {
                    "status": "waiting",
                    "method": "post_request_speech_then_item_aware_quiet",
                    "answer_start_seconds": None,
                    "last_audio_end_seconds": None,
                    "required_quiet_seconds": self.scenario.response_quiet_seconds,
                }
            self.records.append(self.active)
        return speaker

    def observe(
        self, now, speaking, text, item_bearing_audio=False, user_speaking=None
    ):
        """Accumulate actual incoming chunks and per-event evidence."""
        if speaking or item_bearing_audio:
            self.heard_assistant_audio = True
        if text:
            if self.history and self.history[-1]["role"] == "assistant":
                self.history[-1]["text"] += text
            else:
                self.history.append(
                    {
                        "role": "assistant",
                        "text": text,
                        "seconds": max(0, now - self.dt),
                    }
                )
        if self.active:
            self.active["observations"].append(
                {
                    "seconds": max(0, now - self.dt),
                    "speaking": speaking,
                    "text": text,
                    "user_speaking": user_speaking,
                }
            )
            completion = self.active.get("memory_response_completion")
            # Ignore residual audio from the answer preceding the recall request.
            if completion and now - self.dt >= self.active["end_seconds"] - 1e-6:
                if speaking and completion["answer_start_seconds"] is None:
                    completion["answer_start_seconds"] = max(0, now - self.dt)
                if speaking or item_bearing_audio:
                    completion["last_audio_end_seconds"] = now

    def advance_observation(self, now, output_pending):
        """Protect recall answers beyond the minimum observation window.

        Delivered chunks have no response-completion signal. Infer playback end
        from sustained absence of both audible and item-bearing audio, and retain
        the criterion in the record. A timeout releases the scheduler but leaves
        the event incomplete rather than claiming its answer was observed.
        """
        if not self.active or output_pending:
            return
        if self.observing_until is None:
            self.observing_until = now + self.active["observation_seconds"]
        if now < self.observing_until:
            return
        completion = self.active.get("memory_response_completion")
        status = "observed"
        if completion:
            last_audio = completion["last_audio_end_seconds"]
            quiet = last_audio is not None and (
                now - last_audio >= self.scenario.response_quiet_seconds - 1e-6
            )
            if completion["answer_start_seconds"] is not None and quiet:
                completion["status"] = "completed"
            elif (
                now - self.active["end_seconds"]
                >= self.scenario.memory_response_timeout_seconds
            ):
                completion["status"] = "timeout"
                status = "observation_timeout"
            else:
                return
            completion["released_seconds"] = now
        self.active["status"] = status
        self.active["observation_end_seconds"] = now
        self.active = None
        self.observing_until = None

    def report(self):
        """Keep missing events explicit; decision proposals are not coverage."""
        reached = {row["event_id"] for row in self.records}
        rows = list(self.records) + [
            {"event_id": key, "status": "not_reached"}
            for key in self.scenario.event_ids
            if key not in reached
        ]
        return {
            "status": "complete"
            if not self.pending
            and self.active is None
            and all(row["status"] == "observed" for row in rows)
            else "incomplete",
            "events": rows,
            "selected_events": self.scenario.selected_events,
            "supplemental_events": self.scenario.supplemental_events,
            "required_event_order": self.scenario.event_order,
            "observed_event_order": [row["event_id"] for row in self.records],
            "out_of_order_proposals": sum(
                1
                for row in self.decisions
                if not row.get("accepted")
                and "required event order" in row.get("feedback", "")
            ),
            "decisions": self.decisions,
            "timing_note": "Assistant observations belong to the previous output tick; timing resolution is one tick. Speech activity alone does not prove semantic correctness.",
        }


class DuplexUserSimulator(VoiceStreamingUserSimulator):
    """Use Terra for live timing, event choice and utterance generation."""

    def __init__(
        self,
        *,
        scenario: DuplexScenario,
        output_dir: Path,
        primary_voice="nova",
        background_voice="onyx",
        tts_model=None,
        perception_config=None,
        **kwargs,
    ):
        if primary_voice == background_voice:
            raise ValueError("Primary and background voices must differ")
        self.scenario = scenario
        self.perception_config = perception_config
        self.acoustic_noise = prepare_noise(
            scenario.acoustic_clarification, PCM_SAMPLE_RATE, scenario.seed
        )
        self.output_dir = Path(output_dir)
        self.primary_voice = primary_voice
        self.background_voice = background_voice
        self.tts_model = tts_model
        self.pending_decision = None
        self.synthesis_records = []
        self.active_speaker = "primary"
        self.next_check = 0
        self.consecutive_errors = 0
        self.assistant_speaking = False
        self.last_assistant_audio_seconds = -1.0
        self.last_user_speaking = False
        self.acoustic_records = []
        self.acoustic_user_utterances = []
        self.acoustic_delivery_audio = {}
        self.acoustic_assistant_observations = []
        self.acoustic_current = None
        self.perception = self._new_perception()
        super().__init__(**kwargs)
        synthesis = self.voice_settings.synthesis_config
        provider_config = synthesis.provider_config
        self.tts_model = self.tts_model or (
            provider_config.resource_id
            if synthesis.provider == "volcengine"
            else provider_config.model_id
            if synthesis.provider == "elevenlabs"
            else provider_config.model
        )
        self.controller = DuplexController(scenario, self.tick_duration_seconds)
        self.enable_user_initiated_interruption = False
        self.use_llm_backchannel = False

    def get_init_state(self, message_history=None):
        """Start a fresh conversation without preselected actions or TTS clips."""
        self.controller = DuplexController(self.scenario, self.tick_duration_seconds)
        self.pending_decision = None
        self.synthesis_records = []
        self.acoustic_records = []
        self.acoustic_user_utterances = []
        self.acoustic_delivery_audio = {}
        self.acoustic_assistant_observations = []
        self.acoustic_current = None
        self.next_check = 0
        self.consecutive_errors = 0
        self.last_assistant_audio_seconds = -1.0
        self.last_user_speaking = False
        self.perception = self._new_perception()
        return super().get_init_state(message_history)

    def _new_perception(self):
        if self.scenario.assistant_observation == "played_audio":
            from tau2.user.duplex_perception import PlayedAudioPerception

            return PlayedAudioPerception(
                self.scenario.language,
                self.perception_config,
                model=self.scenario.transcription_model,
            )
        return None

    def speech_detection(self, chunk):
        """Use actual audio activity for the event's overlap constraint."""
        return is_audible_chunk(chunk)

    def _decide(self, now):
        c = self.controller
        perception = getattr(self, "perception", None)
        conversation = (
            perception.history(c.history, now)
            if perception is not None
            else c.history[-60:]
        )
        payload = {
            "language": self.scenario.language,
            "topic": self.scenario.topic,
            "user_context": self.scenario.user_context,
            "conversation_outline": self.scenario.conversation_outline,
            "current_event_guidance": self.scenario.event_guidance.get(
                c.next_required_event, ""
            ),
            "elapsed_seconds": now,
            "assistant_speaking": self.assistant_speaking,
            "assistant_text_source": (
                "asr_of_played_audio"
                if perception is not None
                else "proportional_estimate"
            ),
            "pending_events": [
                EVENT_CATALOG[key].model_dump() for key in sorted(c.pending)
            ],
            "selected_events": self.scenario.selected_events,
            "supplemental_events": self.scenario.supplemental_events,
            "required_event_order": self.scenario.event_order,
            "required_next_event": c.next_required_event,
            "backchannel_min_speech_seconds": self.scenario.backchannel_min_speech_seconds,
            "prepared_events": sorted(c.prepared),
            "executed_events": [row["event_id"] for row in c.records],
            "recent_conversation": conversation,
            "previous_execution_feedback": c.feedback,
            "decision_schema": DuplexDecision.model_json_schema(),
        }
        result = generate(
            model=self.scenario.user_model,
            messages=[
                UserMessage(
                    role="user",
                    content=RUNTIME_POLICY
                    + "\n"
                    + json.dumps(payload, ensure_ascii=False),
                )
            ],
            response_format={"type": "json_object"},
            call_name="duplex_runtime_decision",
            **terra_llm_args(self.scenario.user_model),
        )
        decision = DuplexDecision.model_validate_json(result.content)
        accepted = c.allowed(decision, self.assistant_speaking)
        c.decisions.append(
            {
                "seconds": now,
                "model": self.scenario.user_model,
                "decision": decision.model_dump(),
                "accepted": accepted,
                "feedback": c.feedback,
                "cost": result.cost,
                "generation_time_seconds": result.generation_time_seconds,
            }
        )
        return decision if accepted else None

    def _next_turn_taking_action(self, state):
        now = state.elapsed_samples / PCM_SAMPLE_RATE
        incoming = (
            state.input_turn_taking_buffer[-1]
            if state.input_turn_taking_buffer
            else None
        )
        self.assistant_speaking = is_audible_chunk(incoming)
        if self.scenario.acoustic_clarification is not None and incoming is not None:
            observation = {
                "start_seconds": max(0, now - self.tick_duration_seconds),
                "end_seconds": now,
                "audible": self.assistant_speaking,
                "text": incoming.content or "",
                "text_is_unverified": True,
                "utterance_ids": list(incoming.utterance_ids or []),
            }
            if incoming.audio_content:
                directory = self.output_dir / "acoustic_audio"
                directory.mkdir(parents=True, exist_ok=True)
                relative = f"acoustic_audio/assistant-{len(self.acoustic_assistant_observations):06d}.wav"
                data = incoming.get_audio_bytes()
                observation["sha256"] = hashlib.sha256(data).hexdigest()
                observation["original_format"] = incoming.audio_format.model_dump(
                    mode="json"
                )
                decoded = convert_to_pcm16(
                    AudioData(data=data, format=incoming.audio_format)
                )
                observation["audio_file"] = relative
                observation["saved_pcm_sha256"] = hashlib.sha256(
                    decoded.data
                ).hexdigest()
                with wave.open(str(self.output_dir / relative), "wb") as wav:
                    wav.setnchannels(decoded.format.channels)
                    wav.setsampwidth(2)
                    wav.setframerate(decoded.format.sample_rate)
                    wav.writeframes(decoded.data)
            self.acoustic_assistant_observations.append(observation)
        # A quiet frame within a delivered utterance is not playback completion.
        # Require a full second without item-bearing audio before ending a run.
        if self.assistant_speaking or (incoming is not None and incoming.utterance_ids):
            self.last_assistant_audio_seconds = now
        if self.perception is not None:
            self.perception.observe(incoming, max(0, now - self.tick_duration_seconds))
        c = self.controller
        c.observe(
            now,
            self.assistant_speaking,
            (incoming.content or "") if incoming else "",
            item_bearing_audio=bool(incoming is not None and incoming.utterance_ids),
            user_speaking=self.last_user_speaking,
        )
        if (
            c.active
            and c.active["event_id"] == "proactive.contextual_interruption"
            and now - self.tick_duration_seconds >= c.active["marker_seconds"]
            and self.assistant_speaking
            and state.output_streaming_queue
        ):
            state.output_streaming_queue.clear()
            c.active["user_stopped_for_agent"] = True
            c.active["end_seconds"] = now
            for entry in reversed(c.history):
                if entry["role"] == "user":
                    entry["truncated"] = True
                    break
        if state.output_streaming_queue:
            if self.scenario.acoustic_clarification is not None and not (
                c.active
                and c.active["event_id"]
                in {"turn_taking.handoff", "proactive.backchannel"}
            ):
                # Reuse the shared interruption thresholds and stop_talking queue
                # cancellation instead of adding an acoustic-specific policy.
                action = super()._next_turn_taking_action(state)
                if action.action == "stop_talking":
                    return action
            return TurnTakingAction(action="keep_talking")
        c.advance_observation(now, False)
        protected = c.active and (
            self.scenario.acoustic_clarification is None
            or c.active["event_id"] == "turn_taking.handoff"
        )
        if protected or now < self.next_check:
            return TurnTakingAction(action="wait")
        if (
            self.assistant_speaking
            and c.pending
            and all(
                EVENT_CATALOG[key].trigger == "assistant_silent" for key in c.pending
            )
        ):
            # No pending event can run while the assistant speaks, and ordinary
            # user turns cannot interrupt. Keep listening without repeatedly
            # freezing the audio clock for identical LLM/ASR wait decisions.
            return TurnTakingAction(action="wait")
        if self.assistant_speaking:
            # Ordinary repairs are chosen after listening, using the same model.
            if self.scenario.acoustic_clarification is not None:
                return TurnTakingAction(action="wait")
        if not c.pending and now - self.last_assistant_audio_seconds < 1.0:
            return TurnTakingAction(action="wait")
        self.next_check = now + self.scenario.decision_interval_seconds
        try:
            decision = self._decide(now)
            self.consecutive_errors = 0
        except Exception as exc:
            self.consecutive_errors += 1
            c.feedback = f"Runtime decision failed: {type(exc).__name__}"
            c.decisions.append({"seconds": now, "error_type": type(exc).__name__})
            if self.consecutive_errors >= 3:
                raise RuntimeError(
                    "Three consecutive runtime user decision failures"
                ) from exc
            return TurnTakingAction(action="wait")
        if decision is None or decision.action == "wait":
            return TurnTakingAction(action="wait")
        self.pending_decision = decision
        return TurnTakingAction(
            action="generate_message",
            info=f"duplex_runtime:{decision.event_id or decision.action}",
        )

    def _speech(self, text, background=False, pause_seconds=0.0, continuation=""):
        if not text:
            return np.zeros(0, dtype="<i2")
        voice = self.background_voice if background else self.primary_voice
        synthesis = self.voice_settings.synthesis_config
        provider = synthesis.provider
        provider_config = synthesis.provider_config.model_copy(deep=True)
        if pause_seconds and provider != "volcengine":
            # Providers without SSML retain the explicit PCM pause path. The
            # Volcengine path below still synthesizes the entire pause in one call.
            return np.concatenate(
                [
                    self._speech(text, background),
                    np.zeros(round(pause_seconds * PCM_SAMPLE_RATE), dtype="<i2"),
                    self._speech(continuation, background),
                ]
            )
        ssml = None
        if provider == "volcengine":
            provider_config.speaker = voice
            provider_config.resource_id = self.tts_model
            if pause_seconds:
                ssml = (
                    f'<speak>{text}<break time="{int(pause_seconds * 1000)}ms"/>'
                    f"{continuation}</speak>"
                )
            provider_config.ssml = ssml
        elif provider == "elevenlabs":
            provider_config.voice_id = voice
        elif provider == "openai":
            provider_config.voice = voice
            provider_config.model = self.tts_model
        audio = synthesize_voice(
            text + continuation if ssml else text, provider, provider_config
        )
        self._acoustic_last_tts_return = time.perf_counter()
        # The script is an intention, not an ASR transcript: TTS can omit words.
        # Preserve its response before any resampling, chunking or audio effects.
        directory = self.output_dir / "synthesized_audio"
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"{len(self.synthesis_records) + 1:04d}.wav"
        with wave.open(str(directory / filename), "wb") as wav:
            wav.setnchannels(audio.format.channels)
            wav.setsampwidth(audio.format.sample_width)
            wav.setframerate(audio.format.sample_rate)
            wav.writeframes(audio.data)
        record = {
            "script_text": text,
            "script_is_verified_transcript": False,
            "model": self.tts_model,
            "provider": provider,
            "ssml": ssml,
            "voice": voice,
            "audio_file": f"synthesized_audio/{filename}",
            "format": audio.format.model_dump(mode="json"),
            "raw_bytes": len(audio.data),
            "raw_duration_seconds": audio.duration,
            "raw_sha256": hashlib.sha256(audio.data).hexdigest(),
        }
        self.synthesis_records.append(record)
        self._save_synthesis_records()
        audio = resample_audio(audio, PCM_SAMPLE_RATE)
        record["resampled_bytes"] = len(audio.data)
        record["resampled_duration_seconds"] = audio.duration
        record["resampled_sha256"] = hashlib.sha256(audio.data).hexdigest()
        self._save_synthesis_records()
        samples = np.frombuffer(audio.data, dtype="<i2")
        if not samples.size:
            raise RuntimeError("Synthesis produced no audio")
        return samples

    def _save_synthesis_records(self):
        """Persist synthesis evidence even if later streaming fails."""
        (self.output_dir / "synthesis.json").write_text(
            json.dumps(
                {"utterances": self.synthesis_records}, ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )

    def _noise(self, kind):
        rng = np.random.default_rng(self.scenario.seed + len(self.controller.records))
        count = PCM_SAMPLE_RATE * 2
        if kind == "phone_ring":
            time = np.arange(count) / PCM_SAMPLE_RATE
            return (
                3000 * np.sin(2 * np.pi * 700 * time) * ((time % 0.7) < 0.4)
            ).astype("<i2")
        if kind == "rustle":
            return np.clip(rng.normal(0, 1800, count), -32768, 32767).astype("<i2")
        samples = np.zeros(count)
        width = int(PCM_SAMPLE_RATE * 0.08)
        for onset in (0.1, 0.7, 1.3):
            offset = int(onset * PCM_SAMPLE_RATE)
            samples[offset : offset + width] = rng.normal(0, 7000, width) * np.exp(
                -np.linspace(0, 6, width)
            )
        return np.clip(samples, -32768, 32767).astype("<i2")

    def _generate_full_duplex_voice_message(self, message, state):
        decision = self.pending_decision
        self.pending_decision = None
        if decision.action == "finish":
            self.controller.finished = True
            return UserMessage(role="user", content="###STOP###"), state
        is_event = decision.action == "trigger_event"
        background = is_event and decision.event_id in {
            "rejection.third_party",
        }
        background = background or (
            is_event
            and decision.event_id == "rejection.contextual"
            and decision.contextual_speaker == "background"
        )
        noise = is_event and decision.event_id == "rejection.noise"
        first = (
            self._noise(decision.noise)
            if noise
            else self._speech(
                decision.text,
                background,
                decision.pause_seconds,
                decision.continuation,
            )
        )
        if (
            is_event
            and decision.event_id == "proactive.backchannel"
            and self._audible_seconds(first)
            < self.scenario.backchannel_min_speech_seconds
        ):
            self.controller.feedback = (
                "The synthesized backchannel opportunity was too short: "
                f"{self._audible_seconds(first):.2f} audible seconds. Generate a substantive "
                f"account of at least {self.scenario.backchannel_min_speech_seconds} "
                "seconds; the event remains pending and nothing was played."
            )
            return UserMessage(role="user", content=""), state
        # _speech returns prefix + pause + continuation as one waveform. CTC
        # deliberately has no continuation and ends with the hesitation.
        following = (
            np.zeros(0, dtype="<i2")
            if decision.pause_seconds
            else self._speech(decision.continuation)
        )
        samples = np.concatenate(
            [
                first,
                following,
            ]
        )
        marker = len(first) / PCM_SAMPLE_RATE
        pause_evidence = None
        if decision.pause_seconds:
            from tau2.voice.utils.pause_evidence import locate_synthesized_pause

            pause_evidence = locate_synthesized_pause(
                first, PCM_SAMPLE_RATE, decision.pause_seconds
            )
            if pause_evidence["status"] != "measured":
                self.controller.feedback = (
                    "The requested pause could not be uniquely located in the "
                    "synthesized audio; nothing was played. Try a clearer unfinished "
                    "phrase with only one deliberate pause."
                )
                return UserMessage(role="user", content=""), state
            marker = pause_evidence["start_seconds"]
        now = state.elapsed_samples / PCM_SAMPLE_RATE
        if self.scenario.acoustic_clarification is not None:
            record = {
                "id": f"user-{len(self.acoustic_user_utterances) + 1:04d}",
                "script_text": decision.text + decision.continuation,
                "script_is_verified_transcript": False,
                "sample_rate": PCM_SAMPLE_RATE,
                "sample_count": len(samples),
                "start_seconds": now,
                "natural_end_seconds": now + len(samples) / PCM_SAMPLE_RATE,
                "actual_stop_seconds": None,
                "cancelled_samples": [],
                "delivery": [],
                "delivered_samples": 0,
                "delivered_target_samples": 0,
                "delivered_modified_samples": 0,
                "status": "prepared",
            }
            if not self.acoustic_records and not background and not noise:
                preparation_started = time.perf_counter()
                record.update(
                    {
                        "mode": self.scenario.acoustic_clarification.mode,
                        "severity": "hard",
                    }
                )
                self.acoustic_records.append(record)
                clean = samples
                samples = degrade_utterance(
                    samples,
                    record["script_text"],
                    decision.keywords or decision.keyword,
                    PCM_SAMPLE_RATE,
                    self.scenario.seed,
                    record,
                    mode=self.scenario.acoustic_clarification.mode,
                    noise=self.acoustic_noise,
                )
                directory = self.output_dir / "acoustic_audio"
                directory.mkdir(parents=True, exist_ok=True)
                for name, values in (("clean", clean), ("degraded", samples)):
                    relative = f"acoustic_audio/{record['id']}-{name}.wav"
                    (self.output_dir / relative).write_bytes(
                        pcm_wav(values, PCM_SAMPLE_RATE)
                    )
                    record[f"{name}_audio_file"] = relative
                changed = np.flatnonzero(clean != samples)
                if changed.size:
                    breaks = np.flatnonzero(np.diff(changed) > 1)
                    record["modified_intervals_samples"] = [
                        [int(a), int(b) + 1]
                        for a, b in zip(
                            np.r_[changed[0], changed[breaks + 1]],
                            np.r_[changed[breaks], changed[-1]],
                        )
                    ]
                else:
                    record["modified_intervals_samples"] = []
                    if record["status"] != "failed":
                        record["status"] = "failed"
                        record["failure_reason"] = "no_effective_modification"
                record["timing"]["tts_postprocess_seconds"] = (
                    time.perf_counter() - preparation_started
                )
            self.acoustic_user_utterances.append(record)
            self.acoustic_delivery_audio[record["id"]] = []
            self.acoustic_current = record
        self.active_speaker = self.controller.begin(
            decision,
            now,
            len(samples) / PCM_SAMPLE_RATE,
            marker,
            self.assistant_speaking,
        )
        if pause_evidence and self.controller.active:
            self.controller.active["pause_audio_evidence"] = pause_evidence
        if self.acoustic_current is not None:
            self.acoustic_current["source"] = self.active_speaker
        if self.perception is not None:
            self.perception.start_user_turn()
        text = "" if noise else decision.text + decision.continuation
        return UserMessage(
            role="user",
            content="_noise_" if noise else text,
            audio_script_gold=text,
            is_audio=True,
            audio_content=audio_bytes_to_string(samples.tobytes()),
            audio_format=AudioFormat(
                encoding=AudioEncoding.PCM_S16LE, sample_rate=PCM_SAMPLE_RATE
            ),
        ), state

    def _perform_turn_taking_action(self, state, action):
        now = state.elapsed_samples / PCM_SAMPLE_RATE
        chunk, state = super()._perform_turn_taking_action(state, action)
        record = self.acoustic_current
        if record is not None:
            if action.action == "stop_talking":
                record["actual_stop_seconds"] = now
                record["cancelled_samples"] = [
                    record["delivered_samples"],
                    record["sample_count"],
                ]
                if record["status"] != "failed":
                    record["status"] = "incomplete"
                for entry in reversed(self.controller.history):
                    if entry["role"] == "user":
                        entry["truncated"] = True
                        break
                if self.controller.active is not None:
                    self.controller.active["user_stopped_for_agent"] = True
                    self.controller.active["natural_end_seconds"] = (
                        self.controller.active["end_seconds"]
                    )
                    self.controller.active["end_seconds"] = now
                self.acoustic_current = None
            elif action.action in {"generate_message", "keep_talking"}:
                start = record["delivered_samples"]
                end = min(start + self.chunk_size, record["sample_count"])
                record["delivery"].append(
                    {
                        "start_seconds": now,
                        "end_seconds": now + (end - start) / PCM_SAMPLE_RATE,
                        "sample_start": start,
                        "sample_end": end,
                        "padding_samples": self.chunk_size - (end - start),
                        "utterance_ids": list(chunk.utterance_ids or []),
                        "sha256": hashlib.sha256(chunk.get_audio_bytes()).hexdigest(),
                    }
                )
                record["delivered_samples"] = end
                if action.action == "generate_message" and "timing" in record:
                    record["timing"]["tts_postprocess_seconds"] = (
                        time.perf_counter() - self._acoustic_last_tts_return
                    )
                self.acoustic_delivery_audio[record["id"]].append(
                    chunk.get_audio_bytes()
                )
                record["delivered_format"] = chunk.audio_format.model_dump(mode="json")
                if (
                    chunk.audio_format.encoding != AudioEncoding.PCM_S16LE
                    or chunk.audio_format.sample_rate != PCM_SAMPLE_RATE
                ):
                    record["status"] = "failed"
                    record["failure_reason"] = "unsupported_delivery_format"
                record["delivered_modified_samples"] = sum(
                    max(0, min(end, b) - a)
                    for a, b in record.get("modified_intervals_samples", [])
                )
                ranges = record.get("target_ranges_samples", [])
                if not ranges and record.get("target_samples"):
                    ranges = [record["target_samples"]]
                record["delivered_target_samples"] = sum(
                    max(0, min(end, stop) - start) for start, stop in ranges
                )
                if end == record["sample_count"]:
                    record["actual_stop_seconds"] = (
                        now + (end - start) / PCM_SAMPLE_RATE
                    )
                    if record["status"] != "failed":
                        record["status"] = "delivered"
                    self.acoustic_current = None
        if self.scenario.acoustic_clarification is not None:
            # The agent-facing channel carries actual audio only. Clean scripts
            # and neutral annotations remain private synthesis/evaluation data.
            if not self.is_stop(chunk):
                chunk.content = ""
                chunk.audio_script_gold = None
        if chunk.contains_speech:
            chunk.source = self.active_speaker
            chunk.contains_speech = self.active_speaker != "noise" and is_audible_chunk(
                chunk
            )
        self.last_user_speaking = (
            is_audible_chunk(chunk) and self.active_speaker != "noise"
        )
        return chunk, state

    @staticmethod
    def _audible_seconds(samples):
        """Count actual active 20 ms frames so padded silence cannot satisfy length."""
        from tau2.voice.utils.audio_activity import AUDIO_RMS_THRESHOLD

        width = round(PCM_SAMPLE_RATE * 0.02)
        if not len(samples):
            return 0.0
        padded = np.pad(samples, (0, (-len(samples)) % width)).astype(np.float64)
        frames = padded.reshape(-1, width)
        active = np.sqrt(np.mean(frames**2, axis=1)) >= AUDIO_RMS_THRESHOLD
        return float(np.count_nonzero(active) * 0.02)

    def save_report(self):
        """Persist runtime decisions and coverage even after a failed run."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.perception is not None:
            self.perception.save(self.output_dir)
        report = self.controller.report()
        if self.scenario.acoustic_clarification is not None:
            for utterance in self.acoustic_user_utterances:
                audio = b"".join(self.acoustic_delivery_audio[utterance["id"]])
                utterance["delivered_sha256"] = hashlib.sha256(audio).hexdigest()
                if (
                    audio
                    and utterance.get("delivered_format", {}).get("encoding")
                    == AudioEncoding.PCM_S16LE.value
                ):
                    relative = f"acoustic_audio/{utterance['id']}-delivered.wav"
                    (self.output_dir / relative).write_bytes(
                        pcm_wav(
                            np.frombuffer(audio, dtype="<i2"),
                            utterance["delivered_format"]["sample_rate"],
                        )
                    )
                    utterance["delivered_audio_file"] = relative
            records = self.acoustic_records
            if not records or records[0]["status"] != "delivered":
                report["status"] = "incomplete"
            report["acoustic_clarification"] = {
                "status": records[0]["status"] if records else "not_reached",
                "mode": self.scenario.acoustic_clarification.mode,
                "severity": "hard",
                "utterances": records,
                "user_utterances": self.acoustic_user_utterances,
                "assistant_observations": self.acoustic_assistant_observations,
            }
        (self.output_dir / "events.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return report

    def stop(self, participant_chunk=None, state=None, tool_results=None):
        """Save evidence while the existing pipeline saves its own audio data."""
        self.save_report()
        return super().stop(participant_chunk, state, tool_results)
