"""Reproducible scheduling of scenarios on the real τ voice user simulator."""

import math
from copy import deepcopy

import numpy as np

from tau2.config import (
    OPEN_MODEL_DEMO_ACTIVITY_FRAME_MS,
    OPEN_MODEL_DEMO_ASR_REFRESH_SECONDS,
    OPEN_MODEL_DEMO_ASR_WINDOW_SECONDS,
    OPEN_MODEL_DEMO_TRIGGER_SECONDS,
)
from tau2.data_model.audio import AudioData
from tau2.data_model.message import SystemMessage, TurnTakingAction
from tau2.data_model.voice import TranscriptionConfig
from tau2.user.user_simulator_streaming import VoiceStreamingUserSimulator
from tau2.voice.transcription.transcribe import transcribe_audio
from tau2.voice.utils.audio_activity import active_pcm16_frames
from tau2.voice.utils.audio_preprocessing import convert_to_pcm16, resample_audio
from tau2.voice.utils.incremental_transcript import IncrementalTranscript

SCENARIOS = {
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
                if self.scenario == "long_conversation":
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
                        if self.scenario == "long_conversation":
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
        if self.scenario == "long_conversation" and result[0].contains_speech:
            self._substantive_turns += 1
        return result

    def _listener_context(self, state):
        if self.scenario != "long_conversation":
            return super()._listener_context(state)
        snapshot = next((e for e in reversed(self.asr_events) if e["transcript"]), None)
        return (
            SCENARIOS[self.scenario]
            + "\nReact only to actually heard mistakes or completed semantic units. "
            + "Do not interrupt simply because enough words have been spoken. "
            + f"\nLatest authoritative heard ASR snapshot: {snapshot}"
        )

    def _next_turn_taking_action(self, state):
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
