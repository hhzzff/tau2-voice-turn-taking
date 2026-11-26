"""Reproducible scheduling of scenarios on the real τ voice user simulator."""

import math
import re
from copy import deepcopy

import numpy as np
from loguru import logger

from tau2.config import (
    OPEN_MODEL_DEMO_ACTIVITY_FRAME_MS,
    OPEN_MODEL_DEMO_ASR_REFRESH_SECONDS,
    OPEN_MODEL_DEMO_ASR_WINDOW_SECONDS,
    OPEN_MODEL_DEMO_TRIGGER_SECONDS,
    OPEN_MODEL_ORDERED_BOUNDARY_CHECK_SECONDS,
)
from tau2.data_model.audio import AudioData
from tau2.data_model.message import SystemMessage, TurnTakingAction, UserMessage
from tau2.data_model.voice import TranscriptionConfig
from tau2.user.user_simulator_streaming import VoiceStreamingUserSimulator
from tau2.utils.llm_utils import generate
from tau2.voice.transcription.transcribe import transcribe_audio
from tau2.voice.utils.audio_activity import active_pcm16_frames
from tau2.voice.utils.audio_preprocessing import convert_to_pcm16, resample_audio
from tau2.voice.utils.incremental_transcript import IncrementalTranscript

SCENARIOS = {
    "ordered_conversation": """English only. Follow the shared conversation protocol: greet and request a THREE-day Beijing itinerary with your mother; after the assistant actually starts speaking, interrupt to change the trip to TWO days; acknowledge a complete useful idea in the subsequent assistant speech with a brief backchannel; then give a longer description of preferences so the assistant has an opportunity to backchannel; finally negotiate the plan dynamically. Your mother tires on stairs, you like local food and history, and your budget is 600 yuan per person per day excluding the hotel. Actual wording and follow-ups must react to already heard audio. Do not invent assistant responses or claim a correction succeeded without hearing it.""",
    "long_conversation": """Have a sustained, natural conversation to plan a two-day Beijing trip with your mother. Your budget is 600 yuan per person per day excluding the hotel. Your mother gets tired on stairs, you arrive on Friday evening, and you enjoy food and history. Reveal these facts gradually as relevant, ask follow-up questions, compare alternatives, and negotiate a useful plan over at least eight substantive user turns. If the assistant contradicts your known constraints or makes a mistake, you may interrupt to correct that specific mistake after hearing it. Acknowledge substantive completed ideas at natural semantic boundaries without taking the floor unnecessarily. During one suitable turn, describe your family preferences and previous travel experience at length (roughly 60-90 words), giving the model opportunities to acknowledge you. Choose your actual words and next topic dynamically from what you have heard. Do not invent assistant responses or assume it followed a correction. Do not end the conversation early just because one question was answered.""",
    "qa": "Ask for three practical tips for a first visit to Beijing. Start with a brief question.",
    "correction": "Ask for a detailed plan for a three-day Beijing visit. When instructed to correct, say that you actually only have one day, and ask to revise the plan.",
    "user_backchannel": "Ask for a detailed explanation of how to plan a first visit to Beijing. Listen and acknowledge without requesting the floor.",
    "model_backchannel": "Describe your plans for a Beijing trip in a continuous 60 to 80 word utterance. Include several details and natural listener acknowledgement opportunities. Do not end with a question until the end.",
}


class OpenModelScenarioUser(VoiceStreamingUserSimulator):
    """Use normal user generation/TTS, with deterministic scenario action timing.

    Incoming native text is discarded. ASR refreshes use at most the last twenty
    seconds of already delivered audio, never queued model output. Each ASR
    annotation states its heard interval; revisions are not claimed to be aligned
    word timestamps. The superclass still owns state, synthesis and audio chunking.
    """

    def __init__(self, *args, scenario="qa", **kwargs):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        super().__init__(*args, **kwargs)
        self.scenario = scenario
        self.events = []
        self.asr_events = []
        self._heard = bytearray()
        self._heard_seconds = 0.0
        self._refresh_seconds = 0.0
        self._heard_speech = False
        self._triggered = False
        self._opened = False
        self._followed_up = False
        self._audible_model_seconds = 0.0
        self._last_model_active = False
        self._incremental_asr = IncrementalTranscript()
        self._substantive_turns = 0
        self._ordered_phase = "greeting"
        self._ordered_after_output = False
        self._ordered_last_check = None
        self._ordered_heard_floor = 0.0

    def get_init_state(self, message_history=None):
        """Reset scenario and ASR state for an independent conversation."""
        self.events = []
        self.asr_events = []
        self._heard = bytearray()
        self._heard_seconds = self._refresh_seconds = 0.0
        self._heard_speech = self._triggered = self._opened = False
        self._followed_up = False
        self._audible_model_seconds = 0.0
        self._last_model_active = False
        self._incremental_asr = IncrementalTranscript()
        self._substantive_turns = 0
        self._ordered_phase = "greeting"
        self._ordered_after_output = False
        self._ordered_last_check = None
        self._ordered_heard_floor = 0.0
        return super().get_init_state(message_history)

    def get_next_chunk(self, state, participant_chunk, tool_results=None):
        """Transcribe only delivered audio before invoking normal streaming flow."""
        incoming = deepcopy(participant_chunk)
        if incoming is not None:
            # Existing audio-native agents store audio even with is_audio=False.
            # Use text-mode history consistently after replacing native text with ASR.
            incoming.is_audio = False
            incoming.content = None
            incoming.audio_script_gold = None
            incoming.raw_data = None
            audio = incoming.get_audio_bytes() if incoming.audio_content else b""
            if audio:
                if self.scenario in {"long_conversation", "ordered_conversation"}:
                    pcm = resample_audio(
                        convert_to_pcm16(
                            AudioData(data=audio, format=incoming.audio_format)
                        ),
                        8000,
                    )
                    samples = np.frombuffer(pcm.data, dtype="<i2").astype(float)
                    active = active_pcm16_frames(samples)
                    self._last_model_active = bool(active.any())
                    incoming.contains_speech = self._last_model_active
                    self._audible_model_seconds += (
                        float(active.sum()) * OPEN_MODEL_DEMO_ACTIVITY_FRAME_MS / 1000
                    )
                self._heard.extend(audio)
                seconds = len(audio) / incoming.audio_format.bytes_per_second
                self._heard_seconds += seconds
                self._refresh_seconds += seconds
                maximum = int(
                    OPEN_MODEL_DEMO_ASR_WINDOW_SECONDS
                    * incoming.audio_format.bytes_per_second
                )
                self._heard = self._heard[-maximum:]
                speech = bool(incoming.contains_speech)
                if self._heard_speech and (
                    self._refresh_seconds >= OPEN_MODEL_DEMO_ASR_REFRESH_SECONDS
                    or not speech
                ):
                    result = transcribe_audio(
                        AudioData(
                            data=bytes(self._heard), format=incoming.audio_format
                        ),
                        TranscriptionConfig(model="doubao-asr"),
                    )
                    if result.error:
                        raise RuntimeError(f"Demo ASR failed: {result.error}")
                    start = max(
                        0,
                        self._heard_seconds
                        - len(self._heard) / incoming.audio_format.bytes_per_second,
                    )
                    transcript = result.transcript.strip()
                    if transcript:
                        delivered_text = transcript
                        if self.scenario in {
                            "long_conversation",
                            "ordered_conversation",
                        }:
                            delivered_text = self._incremental_asr.update(
                                transcript, final=not speech
                            )
                        if delivered_text:
                            incoming.content = delivered_text
                    self.asr_events.append(
                        {
                            "heard_start_seconds": start,
                            "heard_end_seconds": self._heard_seconds,
                            "transcript": transcript,
                            "outcome": "transcript" if transcript else "no_speech",
                            "provider_status": result.provider_status,
                            "tick_index": state.tick_count,
                        }
                    )
                    self._refresh_seconds = 0
                    self._heard_speech = False
                self._heard_speech = self._heard_speech or speech
        return super().get_next_chunk(state, incoming, tool_results)

    def _generate_full_duplex_voice_message(self, message, state):
        if self.scenario == "ordered_conversation":
            state.system_messages.append(
                SystemMessage(
                    role="system",
                    content=self._listener_context(state)
                    + f"\nGenerated substantive user turns: {self._substantive_turns}. Continue relevant English follow-ups until at least eight generated substantive turns; do not end early.",
                )
            )
        if self.scenario == "long_conversation":
            state.system_messages.append(
                SystemMessage(
                    role="system",
                    content=(
                        f"Generated substantive user turns: {self._substantive_turns}. "
                        "Continue exploring the trip plan naturally. Until eight generated substantive "
                        "turns, do not emit a stop marker; ask a relevant follow-up, provide "
                        "new preferences, or correct an actually heard mistake. Use only "
                        "the assistant audio already transcribed, not imagined responses."
                    ),
                )
            )
            state.system_messages.append(
                SystemMessage(role="system", content=self._listener_context(state))
            )
        result = super()._generate_full_duplex_voice_message(message, state)
        if self.scenario == "ordered_conversation" and re.search(
            r"[\u3400-\u9fff]", result[0].content or ""
        ):
            raise ValueError("Ordered demo user must speak English only")
        if (
            self.scenario in {"long_conversation", "ordered_conversation"}
            and result[0].contains_speech
        ):
            self._substantive_turns += 1
        return result

    def _listener_context(self, state):
        if self.scenario not in {"long_conversation", "ordered_conversation"}:
            return super()._listener_context(state)
        snapshot = next((e for e in reversed(self.asr_events) if e["transcript"]), None)
        return (
            SCENARIOS[self.scenario]
            + (
                f"\nCurrent shared phase: {self._ordered_phase}. English only. Current trip duration: {2 if any(e.get('phase') == 'interruption' for e in self.events) else 3} days."
                if self.scenario == "ordered_conversation"
                else ""
            )
            + "\nReact only to actually heard mistakes or completed semantic units. "
            + "Do not interrupt simply because enough words have been spoken. "
            + f"\nLatest authoritative heard ASR snapshot: {snapshot}"
        )

    def _emit_ordered(self, state, phase, action, instruction, next_phase):
        """Request one shared protocol stage through the normal voice pipeline."""
        prefix = "[ordered-demo-stage] "
        state.system_messages = [
            message
            for message in state.system_messages
            if not (message.content or "").startswith(prefix)
        ]
        state.system_messages.append(
            SystemMessage(
                role="system", content=prefix + "English only. " + instruction
            )
        )
        self.events.append(
            {
                "tick_index": state.tick_count - 1,
                "phase": phase,
                "kind": action,
                "intent": instruction,
                "heard_model_active_seconds": self._audible_model_seconds,
                "heard_snapshot": self.asr_events[-1] if self.asr_events else None,
            }
        )
        self._ordered_phase = next_phase
        self._ordered_after_output = True
        return TurnTakingAction(action=action, info=f"ordered stage: {phase}")

    def _ordered_boundary(self, state, instruction):
        """Judge a scheduled semantic boundary on actually heard audio."""
        interval = math.ceil(
            OPEN_MODEL_ORDERED_BOUNDARY_CHECK_SECONDS / self.tick_duration_seconds
        )
        if (
            self._ordered_last_check is not None
            and state.tick_count - self._ordered_last_check < interval
        ):
            return False
        self._ordered_last_check = state.tick_count
        prompt = (
            "Decide whether the current heard audio ends at a completed useful "
            "thought suitable for this scheduled listener action. Respond with ONLY "
            "YES or NO. A recent user turn does not disqualify this scheduled action. "
            "One complete useful clause is enough; two sentences are not required. "
            "Answer NO for an unfinished phrase, no substantive heard content, or "
            "an inappropriate acknowledgement. Do not invent unheard words.\n"
            + instruction
            + "\n"
            + self._listener_context(state)
        )
        try:
            response = generate(
                model=self.decision_model,
                messages=[UserMessage(role="user", content=prompt)],
                call_name="backchannel_decision",
            )
        except Exception as error:
            logger.warning("Ordered listener decision failed: {}", type(error).__name__)
            return False
        return response.content.strip().upper() == "YES"

    def _next_ordered_action(self, state):
        """Follow the same audio-driven stage order at either native tick width.

        Scheduled user utterances finish through the normal synthesis queue. The
        model retains control of its own playback: no user onset forces a cut.
        After these stages, original turn-taking chooses dynamic follow-ups.
        """
        if state.output_streaming_queue:
            return TurnTakingAction(
                action="keep_talking", info="finish ordered user turn"
            )
        if self._ordered_after_output:
            self._ordered_after_output = False
            self._audible_model_seconds = 0.0
            # A brief acknowledgement does not invalidate the completed thought
            # that elicited it. Retain it if the model then becomes silent.
            if self._ordered_phase != "wait_for_long_turn":
                self._ordered_heard_floor = self._heard_seconds
                self._heard = bytearray()
                self._refresh_seconds = 0.0
                self._heard_speech = False
                self._incremental_asr = IncrementalTranscript()
        eligible = (
            self._last_model_active
            and self._audible_model_seconds >= OPEN_MODEL_DEMO_TRIGGER_SECONDS
        )
        heard = bool(
            self.asr_events
            and self.asr_events[-1]["transcript"].strip()
            and self.asr_events[-1]["heard_start_seconds"]
            >= self._ordered_heard_floor - 1e-6
        )
        if self._ordered_phase == "greeting":
            return self._emit_ordered(
                state,
                "greeting",
                "generate_message",
                "Begin with a brief greeting and ask for a relaxed THREE-day Beijing "
                "itinerary with your mother. Say only this opening request, in English.",
                "wait_for_interruption",
            )
        if self._ordered_phase == "wait_for_interruption" and eligible and heard:
            return self._emit_ordered(
                state,
                "interruption",
                "generate_message",
                "The assistant has now actually started speaking. Interrupt briefly: "
                "you need to change the trip to only TWO days, not three. Politely ask "
                "to revise the plan. Do not assume the assistant made an error or "
                "accepted the correction. Keep this to one or two English sentences.",
                "wait_for_backchannel",
            )
        if self._ordered_phase == "wait_for_backchannel" and eligible and heard:
            if self._ordered_boundary(
                state,
                "Current shared stage: user backchannel. Answer YES only when the "
                "audio heard after the user's correction contains a complete useful "
                "idea that can naturally receive a brief acknowledgement. Do not "
                "claim the correction was accepted unless the audio supports that.",
            ):
                return self._emit_ordered(
                    state,
                    "user_backchannel",
                    "backchannel",
                    "Acknowledge the completed heard idea briefly, without taking the floor.",
                    "wait_for_long_turn",
                )
        if self._ordered_phase == "wait_for_long_turn" and heard:
            pause = state.time_since_last_other_talk >= math.ceil(
                OPEN_MODEL_DEMO_TRIGGER_SECONDS / self.tick_duration_seconds
            )
            boundary = eligible and self._ordered_boundary(
                state,
                "Current shared stage: the user wants to add a longer personal "
                "description. Answer YES at a completed assistant thought where "
                "the user can naturally take the floor. Answer NO mid-phrase.",
            )
            if pause or boundary:
                return self._emit_ordered(
                    state,
                    "model_backchannel_opportunity",
                    "generate_message",
                    "In English, explain your mother's difficulty with stairs, "
                    "your interest in history and local food, and a budget of 600 "
                    "yuan per person per day excluding the hotel. Describe a past "
                    "trip and preferences in one continuous 60-90 word turn. The "
                    "trip is now TWO days. Include natural sentence boundaries "
                    "for listener acknowledgements; ask at most one question at "
                    "the end. React to what was actually heard.",
                    "free_dialogue",
                )
        if self._ordered_phase == "free_dialogue":
            # Remove the one-shot long-turn instruction before natural follow-ups.
            state.system_messages = [
                message
                for message in state.system_messages
                if not (message.content or "").startswith("[ordered-demo-stage] ")
            ]
            action = super()._next_turn_taking_action(state)
            if action.action in {"generate_message", "backchannel", "stop_talking"}:
                self.events.append(
                    {
                        "tick_index": state.tick_count - 1,
                        "phase": "dynamic_followup",
                        "kind": action.action,
                        "intent": action.info,
                        "heard_model_active_seconds": self._audible_model_seconds,
                        "heard_snapshot": self.asr_events[-1]
                        if self.asr_events
                        else None,
                    }
                )
            return action
        return TurnTakingAction(
            action="wait", info=f"ordered stage: {self._ordered_phase}"
        )

    def _next_turn_taking_action(self, state):
        if self.scenario == "ordered_conversation":
            return self._next_ordered_action(state)
        if self.scenario == "long_conversation":
            # A mostly silent native tick must not count as a full second of speech.
            eligible = (
                self._last_model_active
                and self._audible_model_seconds >= OPEN_MODEL_DEMO_TRIGGER_SECONDS
            )
            interruption, backchannel = (
                self.enable_user_initiated_interruption,
                self.use_llm_backchannel,
            )
            try:
                self.enable_user_initiated_interruption = interruption and eligible
                self.use_llm_backchannel = backchannel and eligible
                action = super()._next_turn_taking_action(state)
            finally:
                self.enable_user_initiated_interruption = interruption
                self.use_llm_backchannel = backchannel
            if action.action in {"generate_message", "backchannel", "stop_talking"}:
                self.events.append(
                    {
                        "tick_index": state.tick_count - 1,
                        "kind": action.action,
                        "intent": action.info,
                        "heard_model_active_seconds": self._audible_model_seconds,
                        "heard_snapshot": self.asr_events[-1]
                        if self.asr_events
                        else None,
                    }
                )
                if action.action == "generate_message":
                    self._audible_model_seconds = 0.0
            return action
        if state.output_streaming_queue:
            return TurnTakingAction(
                action="keep_talking", info="finish scenario utterance"
            )
        if not self._opened:
            self._opened = True
            if self.scenario == "model_backchannel":
                self._triggered = True
                self.events.append(
                    {
                        "tick_index": state.tick_count - 1,
                        "kind": "model_backchannel_opportunity",
                        "intent": "continuing user speech",
                    }
                )
            return TurnTakingAction(action="generate_message", info="scenario opening")
        threshold = math.ceil(
            OPEN_MODEL_DEMO_TRIGGER_SECONDS / self.tick_duration_seconds
        )
        if not self._triggered and state.consecutive_other_speaking_ticks >= threshold:
            if self.scenario in {"correction", "user_backchannel"}:
                self._triggered = True
                kind = self.scenario
                self.events.append(
                    {
                        "tick_index": state.tick_count - 1,
                        "kind": kind,
                        "intent": "seek floor with correction"
                        if kind == "correction"
                        else "acknowledge without seeking floor",
                    }
                )
                if kind == "correction":
                    state.system_messages.append(
                        SystemMessage(
                            role="system",
                            content="Now make the substantive correction: you have only ONE day, not three. Say this briefly and ask to revise the plan. Do not add a new topic.",
                        )
                    )
                return TurnTakingAction(
                    action="generate_message"
                    if kind == "correction"
                    else "backchannel",
                    info=f"scenario {kind}",
                )
        if (
            self.scenario == "qa"
            and not self._followed_up
            and state.time_since_last_other_talk >= threshold
            and any(event["transcript"].strip() for event in self.asr_events)
        ):
            self._followed_up = True
            self._triggered = True
            self.events.append(
                {
                    "tick_index": state.tick_count - 1,
                    "kind": "qa",
                    "intent": "respond after heard answer",
                }
            )
            return TurnTakingAction(
                action="generate_message", info="scenario follow-up"
            )
        return TurnTakingAction(action="wait", info="waiting for scenario opportunity")
