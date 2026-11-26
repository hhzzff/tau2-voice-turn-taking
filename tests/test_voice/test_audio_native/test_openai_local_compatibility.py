"""Local-user additions must preserve existing provider defaults and event data."""

import json

import pytest
from websockets.protocol import State

from tau2.config import DEFAULT_OPENAI_TRANSCRIPTION_MODEL, DEFAULT_OPENAI_VOICE
from tau2.voice.audio_native.openai.provider import (
    OpenAIRealtimeProvider,
    OpenAIVADConfig,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("modality", ["text", "audio", "audio_in_text_out"])
async def test_existing_callers_keep_provider_defaults(modality):
    class Socket:
        state = State.OPEN

        async def send(self, data):
            self.session = json.loads(data)["session"]

        async def recv(self):
            return json.dumps({"type": "session.updated", "session": self.session})

    provider = OpenAIRealtimeProvider(api_key="unit-test", model="unit-test")
    provider.ws = Socket()
    await provider.configure_session("legacy caller", [], OpenAIVADConfig(), modality)
    session = provider.ws.session
    assert "max_output_tokens" not in session
    assert session["output_modalities"] == ["audio" if modality == "audio" else "text"]
    if modality == "text":
        assert "audio" not in session
    else:
        assert session["audio"]["input"]["format"] == {"type": "audio/pcmu"}
        assert session["audio"]["input"]["transcription"] == {
            "model": DEFAULT_OPENAI_TRANSCRIPTION_MODEL,
            "language": "en",
        }
        if modality == "audio":
            assert session["audio"]["output"]["voice"] == DEFAULT_OPENAI_VOICE
        else:
            assert "output" not in session["audio"]


@pytest.mark.asyncio
async def test_raw_receiver_preserves_response_status_and_unknown_fields():
    event = {
        "type": "response.done",
        "response": {
            "id": "response-1",
            "status": "incomplete",
            "status_details": {"reason": "max_output_tokens"},
            "new_server_field": {"nested": [1, 2]},
        },
    }

    class Socket:
        state = State.OPEN

        async def __aiter__(self):
            yield json.dumps(event)

    provider = OpenAIRealtimeProvider(api_key="unit-test", model="unit-test")
    provider.ws = Socket()
    assert [row async for row in provider.receive_raw_events()] == [event]
