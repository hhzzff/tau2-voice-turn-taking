import base64
import json
import os

import requests

from tau2.data_model.audio import AudioData
from tau2.data_model.voice import VolcengineTTSConfig


def tts_volcengine(text: str, config: VolcengineTTSConfig) -> AudioData:
    """Text to speech using Volcengine OpenSpeech TTS."""
    api_key = config.api_key or os.getenv("VOLCENGINE_TTS_API_KEY")
    if not api_key:
        raise ValueError("VOLCENGINE_TTS_API_KEY not found in config or environment")

    additions = {
        "disable_markdown_filter": False,
        "disable_emoji_filter": False,
        "enable_latex_tn": True,
        **config.additions,
    }
    payload = {
        "req_params": {
            "text": text,
            "speaker": config.speaker,
            "additions": json.dumps(additions),
            "audio_params": {
                "format": config.audio_format,
                "sample_rate": config.sample_rate,
                "enable_subtitle": config.enable_subtitle,
            },
        }
    }
    headers = {
        "X-Api-Key": api_key,
        "X-Api-Resource-Id": config.resource_id,
        "Content-Type": "application/json",
        "Connection": "keep-alive",
        "X-Control-Require-Usage-Tokens-Return": "*",
    }

    response = requests.post(
        config.url,
        headers=headers,
        json=payload,
        stream=True,
        timeout=60,
    )
    response.raise_for_status()

    audio_data = bytearray()
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        data = json.loads(line)
        code = data.get("code", 0)
        if code == 0 and data.get("data"):
            audio_data.extend(base64.b64decode(data["data"]))
        elif code == 20000000:
            break
        elif code > 0:
            raise ValueError(f"Volcengine TTS error: {data}")

    if not audio_data:
        raise ValueError("Volcengine TTS returned empty audio")

    return AudioData(data=bytes(audio_data), format=config.output_audio_format)
