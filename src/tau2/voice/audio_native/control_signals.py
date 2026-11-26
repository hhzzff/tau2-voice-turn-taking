"""Live-tested signal contracts; candidates never drive runtime playback.

A contract describes observed wire/native fields and a local playback policy.
It does not attribute a provider/controller decision to the autonomous model.
Exact models, endpoints, checkpoints and trace hashes are in the evidence file.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

EVIDENCE_FILE = "docs/control-signals-live-evidence.json"
TESTED_MODELS = {
    "openai_live": ("gpt-live-1",),
    "openai": ("gpt-realtime-2.1",),
    "gemini": ("gemini-3.1-flash-live-preview",),
    "doubao_s2s": ("1.2.6.0", "1.2.6.1"),
    "minicpm": ("MiniCPM-o-4.5",),
    "salmonn": ("SALMONN-Omni-0519-fc180-aes-er0-chunk30-llama31-checkpoint-44000",),
    "freeze_omni": ("Freeze-Omni",),
    "moshi": ("Moshika-pytorch-bf16",),
}


class ControlSignal(BaseModel):
    """Meaning, playback scope and evidence of a native signal."""

    model_config = ConfigDict(frozen=True)
    transition: str
    playback: Literal["cancel_pending", "drain", "truncate_at_onset", "observe"]
    source: Literal[
        "native_runtime", "provider", "external_policy", "playback_controller"
    ]
    verification: Literal[
        "unverified", "source_checked", "offline_tested", "live_tested"
    ] = "unverified"
    evidence: tuple[str, ...] = ()
    tested_models: tuple[str, ...] = ()
    native_signal: str | None = None
    native_context: str | None = None
    native_value: str | None = None
    observed_values: tuple[str, ...] = ()


def _tested(profile, signal, transition, playback, source="provider", **fields):
    return ControlSignal(
        transition=transition,
        playback=playback,
        source=source,
        verification="live_tested",
        tested_models=TESTED_MODELS[profile],
        evidence=(f"{EVIDENCE_FILE}#{profile}/{signal}",),
        **fields,
    )


CONTROL_SIGNALS: dict[str, dict[str, ControlSignal]] = {
    # Both produce continuous media including quiet frames. There is no discrete
    # interruption signal in the tested contract. Silence must retain its time.
    "openai_live": {},
    "moshi": {},
    "openai": {
        "input_audio_buffer.speech_started": _tested(
            "openai",
            "input_audio_buffer.speech_started",
            "user_speech_started",
            "truncate_at_onset",
        ),
        "response.done": _tested(
            "openai",
            "response.done",
            "generation_status_unknown",
            "observe",
            observed_values=("completed", "cancelled"),
        ),
    },
    "gemini": {
        "interruption": _tested(
            "gemini",
            "interruption",
            "interrupted",
            "cancel_pending",
            native_signal="serverContent.interrupted",
        ),
        "turn.complete": _tested(
            "gemini",
            "turn.complete",
            "turn_closed",
            "drain",
            native_signal="serverContent.turnComplete",
        ),
    },
    "doubao_s2s": {
        "conversation.item.input_audio_transcription.started": _tested(
            "doubao_s2s",
            "conversation.item.input_audio_transcription.started",
            "user_turn_started",
            "cancel_pending",
        ),
        "response.output_audio.done": _tested(
            "doubao_s2s",
            "response.output_audio.done",
            "audio_generation_ended",
            "drain",
        ),
    },
    "minicpm": {
        "is_listen": _tested(
            "minicpm", "is_listen", "listening", "cancel_pending", "native_runtime"
        ),
        "end_of_turn": _tested(
            "minicpm", "end_of_turn", "turn_ended", "drain", "native_runtime"
        ),
        "external_interrupt": _tested(
            "minicpm",
            "external_interrupt",
            "interrupted",
            "cancel_pending",
            "external_policy",
        ),
        "forced_listen": _tested(
            "minicpm", "forced_listen", "listening", "cancel_pending", "external_policy"
        ),
    },
    "salmonn": {
        "listening_after_playback": _tested(
            "salmonn",
            "listening_after_playback",
            "listening",
            "observe",
            "playback_controller",
        ),
        "playback_turn_ended": _tested(
            "salmonn",
            "playback_turn_ended",
            "turn_ended",
            "drain",
            "playback_controller",
            native_signal="text_eot_and_playback_drained",
            native_context="local_playback",
        ),
        "native_assistant_start": _tested(
            "salmonn", "native_assistant_start", "speaking", "observe", "native_runtime"
        ),
    },
    "freeze_omni": {
        # A returned stat requires its phase. sl is also written by the external
        # controller during resets; it must not unconditionally cancel playback.
        "listening.stat=cl": _tested(
            "freeze_omni",
            "listening.stat=cl",
            "listening",
            "observe",
            "native_runtime",
            native_signal="stat",
            native_context="listening",
            native_value="cl",
        ),
        "listening.stat=ss": _tested(
            "freeze_omni",
            "listening.stat=ss",
            "ready_to_reply",
            "observe",
            "native_runtime",
            native_signal="stat",
            native_context="listening",
            native_value="ss",
        ),
        "generation.stat=cs": _tested(
            "freeze_omni",
            "generation.stat=cs",
            "speaking",
            "observe",
            "native_runtime",
            native_signal="stat",
            native_context="generation",
            native_value="cs",
        ),
        "generation.stat=sl": _tested(
            "freeze_omni",
            "generation.stat=sl",
            "generation_ended",
            "drain",
            "native_runtime",
            native_signal="stat",
            native_context="generation",
            native_value="sl",
        ),
    },
    # Existing providers outside the eight-model test scope remain recognized,
    # but their untested candidates do not become published runtime contracts.
    "xai": {},
    "nova": {},
    "qwen": {},
    "livekit": {},
}


def _candidate(transition, playback, source="provider", evidence=()):
    return ControlSignal(
        transition=transition, playback=playback, source=source, evidence=evidence
    )


CONTROL_SIGNAL_CANDIDATES: dict[str, dict[str, ControlSignal]] = {
    "salmonn": {
        "native_listening_transition": _candidate(
            "listening",
            "cancel_pending",
            "native_runtime",
            ("scripts/open_model_workers/salmonn_session.py::NativeSession",),
        ),
    },
    "xai": {
        "input_audio_buffer.speech_started": _candidate(
            "user_speech_started", "cancel_pending"
        ),
        "response.done": _candidate("generation_ended", "drain"),
    },
    "nova": {
        "speechStarted": _candidate("user_speech_started", "cancel_pending"),
        "bargeIn": _candidate("interrupted", "cancel_pending"),
        "completionEnd": _candidate("generation_ended", "drain"),
    },
    "qwen": {
        "input_audio_buffer.speech_started": _candidate(
            "user_speech_started", "cancel_pending"
        ),
        "response.cancelled": _candidate("interrupted", "cancel_pending"),
        "response.done": _candidate("generation_ended", "drain"),
    },
    "livekit": {
        "speech_started": _candidate("user_speech_started", "observe"),
        "interrupted": _candidate("interrupted", "cancel_pending"),
        "tts_completed": _candidate("generation_ended", "drain"),
    },
}

REJECTED_CONTROL_SIGNALS = {
    "salmonn": {
        "native_speech_end": "EOS is a playback-controller input marker, not a required model prediction",
    },
}

EVALUATION_MODELS = {
    "gpt_live": {
        "profile": "openai_live",
        "integration": "partial",
        "mode": "continuous_audio",
        "silence_policy": "preserve_media_time",
        "interruption_contract": "continuous_media",
    },
    "gpt_realtime": {
        "profile": "openai",
        "integration": "implemented",
        "mode": "events",
        "interruption_contract": "event_verified",
    },
    "doubao_s2s": {
        "profile": "doubao_s2s",
        "integration": "reference_only",
        "mode": "events",
        "interruption_contract": "event_verified",
    },
    "gemini_live": {
        "profile": "gemini",
        "integration": "implemented",
        "mode": "events",
        "interruption_contract": "event_verified",
    },
    "minicpm": {
        "profile": "minicpm",
        "integration": "implemented",
        "mode": "state_flags",
        "interruption_contract": "external_control_verified",
    },
    "salmonn": {
        "profile": "salmonn",
        "integration": "implemented",
        "mode": "native_tokens",
        "interruption_contract": "unconfirmed",
    },
    "freeze_omni": {
        "profile": "freeze_omni",
        "integration": "reference_only",
        "mode": "state_machine",
        "interruption_contract": "external_controller_required",
    },
    "moshi": {
        "profile": "moshi",
        "integration": "reference_only",
        "mode": "continuous_audio",
        "silence_policy": "preserve_media_time",
        "interruption_contract": "continuous_media",
    },
}
for target in EVALUATION_MODELS.values():
    target["native_protocol_tested"] = True
    target["tau_adapter_live_validated"] = False
    target["tested_models"] = TESTED_MODELS[target["profile"]]
    target["evidence"] = EVIDENCE_FILE


def control_event(
    profile: str,
    signal: str,
    payload: dict | None = None,
    *,
    model_id: str | None = None,
) -> dict | None:
    """Look up an admitted contract; untested models/candidates stay unknown.

    Without model_id this inspects the contract definition. Runtime callers can
    supply the resolved model ID to avoid transferring evidence across versions.
    """
    contract = CONTROL_SIGNALS[profile].get(signal)
    if contract is None or (
        model_id is not None and model_id not in contract.tested_models
    ):
        return None
    normalized = {
        "type": "playback_control",
        "profile": profile,
        "signal": signal,
        **contract.model_dump(mode="json"),
    }
    normalized["native_signal"] = contract.native_signal or signal
    if model_id is not None:
        normalized["runtime_model"] = model_id
    if profile == "openai" and signal == "response.done":
        payload = payload or {}
        response = payload.get("response") or {}
        status = response.get("status", payload.get("status"))
        identity = response.get("id", payload.get("response_id"))
        normalized["response_status"] = status
        normalized["status_live_tested"] = status in contract.observed_values
        transition, playback = {
            "completed": ("generation_ended", "drain"),
            "cancelled": ("response_cancelled", "cancel_pending"),
            "failed": ("generation_failed", "observe"),
            "incomplete": ("generation_incomplete", "observe"),
        }.get(status, ("generation_status_unknown", "observe"))
        if status == "cancelled":
            normalized.update(playback_scope="response", target_response_id=identity)
            if identity is None:
                playback = "observe"
        if status not in contract.observed_values:
            normalized["verification"] = (
                "source_checked" if status in {"failed", "incomplete"} else "unverified"
            )
        normalized.update(transition=transition, playback=playback)
    return normalized
