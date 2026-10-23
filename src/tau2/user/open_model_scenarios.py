"""Reproducible scheduling of scenarios on the real τ voice user simulator."""

import math
from copy import deepcopy

from tau2.config import (
    OPEN_MODEL_DEMO_ASR_REFRESH_SECONDS,
    OPEN_MODEL_DEMO_ASR_WINDOW_SECONDS,
    OPEN_MODEL_DEMO_TRIGGER_SECONDS,
)
from tau2.data_model.audio import AudioData
from tau2.data_model.message import SystemMessage, TurnTakingAction
from tau2.data_model.voice import TranscriptionConfig
from tau2.user.user_simulator_streaming import VoiceStreamingUserSimulator
from tau2.voice.transcription.transcribe import transcribe_audio

SCENARIOS = {
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

    def get_init_state(self, message_history=None):
        """Reset scenario and ASR state for an independent conversation."""
        self.events = []
        self.asr_events = []
        self._heard = bytearray()
        self._heard_seconds = self._refresh_seconds = 0.0
        self._heard_speech = self._triggered = self._opened = False
        self._followed_up = False
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
                        incoming.content = f"[ASR heard {start:.2f}–{self._heard_seconds:.2f}s] {transcript}"
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

    def _next_turn_taking_action(self, state):
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
