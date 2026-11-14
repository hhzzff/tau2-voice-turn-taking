"""Public control contracts shared by open and closed model integrations."""

import pytest

from tau2.voice.audio_native.control_signals import CONTROL_SIGNALS, control_event


@pytest.mark.parametrize(
    "profile,signal,transition,playback,source",
    [
        ("minicpm", "is_listen", "listening", "cancel_pending", "native_runtime"),
        ("minicpm", "end_of_turn", "turn_ended", "drain", "native_runtime"),
        (
            "minicpm",
            "external_interrupt",
            "interrupted",
            "cancel_pending",
            "external_policy",
        ),
        ("salmonn", "native_assistant_start", "speaking", "observe", "native_runtime"),
        (
            "openai",
            "input_audio_buffer.speech_started",
            "user_speech_started",
            "truncate_at_onset",
            "provider",
        ),
        ("gemini", "interruption", "interrupted", "cancel_pending", "provider"),
        ("gemini", "turn.complete", "turn_closed", "drain", "provider"),
    ],
)
def test_control_contract_keeps_signal_action_and_origin_distinct(
    profile, signal, transition, playback, source
):
    event = control_event(profile, signal)
    assert event["signal"] == signal
    assert event["transition"] == transition
    assert event["playback"] == playback
    assert event["source"] == source


@pytest.mark.parametrize("profile", CONTROL_SIGNALS)
def test_unknown_signal_is_not_invented_as_an_interruption(profile):
    assert control_event(profile, "unknown_control_signal") is None


def test_actual_closed_provider_event_normalization_preserves_raw_event():
    from tau2.voice.audio_native.gemini.discrete_time_adapter import (
        DiscreteTimeGeminiAdapter,
    )
    from tau2.voice.audio_native.gemini.events import GeminiInterruptionEvent

    class Provider:
        needs_reconnection = False
        model = "gemini-3.1-flash-live-preview"

        async def connect(self, **kwargs):
            pass

        async def disconnect(self):
            pass

        async def send_audio(self, audio):
            assert audio

        async def receive_events_for_duration(self, duration):
            return [GeminiInterruptionEvent()]

    adapter = DiscreteTimeGeminiAdapter(1000, provider=Provider())
    adapter.connect("test", [], None)
    try:
        result = adapter.run_tick(b"\xff" * 8000)
        assert result.events == [GeminiInterruptionEvent()]
        assert result.serializable_events[0] == control_event(
            "gemini", "interruption", model_id="gemini-3.1-flash-live-preview"
        )
    finally:
        adapter.disconnect()


@pytest.mark.parametrize(
    "status,transition,playback",
    [
        ("completed", "generation_ended", "drain"),
        ("cancelled", "response_cancelled", "cancel_pending"),
        ("failed", "generation_failed", "observe"),
        ("incomplete", "generation_incomplete", "observe"),
        (None, "generation_status_unknown", "observe"),
    ],
)
@pytest.mark.parametrize("envelope", ["wire", "parsed"])
def test_realtime_done_requires_status_before_selecting_playback(
    status, transition, playback, envelope
):
    payload = (
        {"response": {"status": status, "id": "response-to-test"}}
        if envelope == "wire"
        else {"status": status, "response_id": "response-to-test"}
    )
    event = control_event("openai", "response.done", payload)
    assert event["transition"] == transition
    assert event["playback"] == playback
    assert event["response_status"] == status


def test_evaluation_targets_include_all_eight_without_inventing_native_signals():
    from tau2.voice.audio_native.control_signals import EVALUATION_MODELS

    assert set(EVALUATION_MODELS) == {
        "gpt_live",
        "gpt_realtime",
        "doubao_s2s",
        "gemini_live",
        "minicpm",
        "salmonn",
        "freeze_omni",
        "moshi",
    }
    for target in EVALUATION_MODELS.values():
        assert target["profile"] in CONTROL_SIGNALS
        assert target["native_protocol_tested"] is True
        assert target["tau_adapter_live_validated"] is False
    assert CONTROL_SIGNALS["moshi"] == {}
    assert CONTROL_SIGNALS["freeze_omni"]
    assert "stat=sl" not in CONTROL_SIGNALS["freeze_omni"]
    assert "input_audio_buffer.speech_started" not in CONTROL_SIGNALS["openai_live"]


def test_continuous_audio_models_never_treat_quiet_audio_as_a_cancel_event():
    from tau2.voice.audio_native.control_signals import EVALUATION_MODELS

    for model in ("gpt_live", "moshi"):
        target = EVALUATION_MODELS[model]
        assert target["mode"] == "continuous_audio"
        assert target["silence_policy"] == "preserve_media_time"
        assert (
            control_event(target["profile"], "input_audio_buffer.speech_started")
            is None
        )
        assert control_event(target["profile"], "silence") is None


def test_doubao_opening_turn_is_not_reported_as_an_interruption():
    event = control_event(
        "doubao_s2s", "conversation.item.input_audio_transcription.started"
    )
    assert event["transition"] == "user_turn_started"
    assert event["playback"] == "cancel_pending"


def test_delayed_realtime_cancel_identifies_its_response_not_a_new_reply():
    event = control_event(
        "openai", "response.done", {"response": {"id": "old", "status": "cancelled"}}
    )
    assert event["playback_scope"] == "response"
    assert event["target_response_id"] == "old"


def test_unverified_provider_contracts_are_candidates_not_runtime_actions():
    from tau2.voice.audio_native.control_signals import CONTROL_SIGNAL_CANDIDATES

    for profile in ("xai", "nova", "qwen", "livekit"):
        assert CONTROL_SIGNAL_CANDIDATES[profile]
        assert CONTROL_SIGNALS[profile] == {}


def test_freeze_omni_listening_and_generation_states_have_distinct_actions():
    end = control_event("freeze_omni", "generation.stat=sl")
    assert end["native_signal"] == "stat"
    assert end["native_context"] == "generation"
    assert end["native_value"] == "sl"
    assert end["playback"] == "drain"
    assert control_event("freeze_omni", "listening.stat=ss")["playback"] == "observe"
    assert control_event("freeze_omni", "stat=sl") is None


def test_model_version_without_live_evidence_does_not_inherit_verified_contract():
    assert (
        control_event(
            "openai", "input_audio_buffer.speech_started", model_id="gpt-realtime-1.5"
        )
        is None
    )
    assert (
        control_event(
            "openai", "input_audio_buffer.speech_started", model_id="gpt-realtime-2.1"
        )["verification"]
        == "live_tested"
    )


def test_cancel_without_response_identity_cannot_clear_a_replacement_response():
    event = control_event("openai", "response.done", {"status": "cancelled"})
    assert event["transition"] == "response_cancelled"
    assert event["playback"] == "observe"
    assert event["target_response_id"] is None


def test_salmonn_end_is_a_playback_controller_transition_not_a_prediction():
    event = control_event("salmonn", "playback_turn_ended")
    assert event["source"] == "playback_controller"
    assert event["playback"] == "drain"
    assert control_event("salmonn", "native_speech_end") is None
    returned = control_event("salmonn", "listening_after_playback")
    assert returned["source"] == "playback_controller"
    assert returned["verification"] == "live_tested"
    assert control_event("salmonn", "native_listening_transition") is None


def test_every_published_signal_has_an_actual_trace_witness():
    import json
    from pathlib import Path

    ledger = json.loads(
        (
            Path(__file__).parents[3] / "docs/control-signals-live-evidence.json"
        ).read_text()
    )
    for profile, signals in CONTROL_SIGNALS.items():
        for signal, contract in signals.items():
            assert contract.verification == "live_tested"
            assert contract.tested_models
            witnesses = ledger["signals"][profile][signal]["witnesses"]
            assert witnesses
            assert all(len(w["trace_sha256"]) == 64 for w in witnesses)
    for profile in ("openai_live", "moshi"):
        assert ledger["continuous_media"][profile]
