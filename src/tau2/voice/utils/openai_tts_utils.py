"""OpenAI text-to-speech utilities."""

import os

from openai import OpenAI

from tau2.data_model.audio import AudioData
from tau2.data_model.voice import OpenAITTSConfig


def tts_openai(text: str, config: OpenAITTSConfig) -> AudioData:
    """Text to speech using the OpenAI audio speech API."""
    api_key = config.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in config or environment")

    client = OpenAI(api_key=api_key)
    kwargs = {
        "model": config.model,
        "voice": config.voice,
        "input": text,
        "response_format": config.response_format,
        "speed": config.speed,
    }
    if config.instructions:
        kwargs["instructions"] = config.instructions

    response = client.audio.speech.create(**kwargs)
    audio_bytes = response.content if hasattr(response, "content") else response.read()
    if not audio_bytes:
        raise ValueError("OpenAI TTS returned empty audio")

    return AudioData(data=bytes(audio_bytes), format=config.output_audio_format)
