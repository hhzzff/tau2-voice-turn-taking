import asyncio
import base64
import io
import json
import os
import wave

import requests
import websockets
from dotenv import load_dotenv

from tau2.config import (
    DEFAULT_OPENAI_OUTPUT_SAMPLE_RATE,
    DEFAULT_OPENAI_REALTIME_BASE_URL,
    DEFAULT_WHISPER_MODEL,
)
from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat
from tau2.data_model.voice import TranscriptionConfig, TranscriptionResult
from tau2.utils.retry import api_retry
from tau2.voice.utils.audio_preprocessing import (
    convert_to_mono,
    convert_to_pcm16,
    resample_audio,
)

load_dotenv()


@api_retry
def transcribe_audio(
    audio_data: AudioData, config: TranscriptionConfig
) -> TranscriptionResult:
    try:
        pcm_audio_data, conversion_error = _convert_audio_to_pcm16_mono_24000(
            audio_data
        )
        if conversion_error:
            return TranscriptionResult(transcript="", error=conversion_error)

        if config.model == "scribe_v1":
            return transcribe_scribe(pcm_audio_data, config)

        if config.model == "doubao-asr":
            return transcribe_doubao(audio_data, config)

        if config.model in ["nova-2", "nova-3"]:
            return transcribe_deepgram(pcm_audio_data, config)

        elif config.model == "whisper-1":
            return transcribe_whisper(pcm_audio_data, config)

        elif config.model in ["gpt-4o-transcribe", "gpt-4o-mini-transcribe"]:
            return asyncio.run(transcribe_gpt4o_realtime(pcm_audio_data, config))

        else:
            return TranscriptionResult(
                transcript="", error=f"Unsupported transcription model: {config.model}"
            )

    except Exception as e:
        return TranscriptionResult(
            transcript="", error=f"Transcription failed: {str(e)}"
        )


def _convert_audio_to_pcm16_mono_24000(audio_data: AudioData) -> AudioData:
    try:
        pcm16_audio = convert_to_pcm16(audio_data)
        mono_audio = convert_to_mono(pcm16_audio)
        resampled_audio = resample_audio(mono_audio, DEFAULT_OPENAI_OUTPUT_SAMPLE_RATE)
        return resampled_audio, None

    except Exception as e:
        return None, f"Audio conversion error: {str(e)}"


def _validate_pcm16_mono_24000(audio_format: AudioFormat) -> None:
    if (
        audio_format.encoding != AudioEncoding.PCM_S16LE
        or audio_format.channels != 1
        or audio_format.sample_rate != DEFAULT_OPENAI_OUTPUT_SAMPLE_RATE
    ):
        raise ValueError(
            f"Audio format must be in mono, PCM_S16LE, 24000 Hz, got {audio_format}"
        )


def transcribe_deepgram(
    audio_data: AudioData, config: TranscriptionConfig
) -> TranscriptionResult:
    """Transcribe audio data using Deepgram API.
    Audio data must be in mono, 16-bit PCM format.
    Args:
        audio_data: AudioData object
        config: TranscriptionConfig object
    Returns:
        TranscriptionResult object
    """
    _validate_pcm16_mono_24000(audio_data.format)

    try:
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            return TranscriptionResult(
                transcript="", error="DEEPGRAM_API_KEY not found in environment"
            )

        url = "https://api.deepgram.com/v1/listen"

        params = {
            "model": config.model,
            "punctuate": str(config.deepgram_punctuate).lower(),
            "smart_format": str(config.deepgram_smart_format).lower(),
            "encoding": "linear16",
            "sample_rate": str(DEFAULT_OPENAI_OUTPUT_SAMPLE_RATE),
            "channels": "1",
        }

        if config.language:
            params["language"] = config.language

        params.update(config.extra_options)

        headers = {"Authorization": f"Token {api_key}", "Content-Type": "audio/raw"}

        response = requests.post(
            url, headers=headers, params=params, data=audio_data.data
        )
        response.raise_for_status()

        result = response.json()

        channels = result.get("results", {}).get("channels", [])
        if not channels:
            return TranscriptionResult(
                transcript="", error="No transcription channels found in response"
            )

        alternatives = channels[0].get("alternatives", [])
        if not alternatives:
            return TranscriptionResult(
                transcript="", error="No transcription alternatives found in response"
            )

        transcript = alternatives[0].get("transcript", "")
        confidence = alternatives[0].get("confidence", None)

        return TranscriptionResult(transcript=transcript, confidence=confidence)

    except Exception as e:
        return TranscriptionResult(
            transcript="", error=f"Deepgram transcription failed: {str(e)}"
        )


def transcribe_whisper(
    audio_data: AudioData, config: TranscriptionConfig
) -> TranscriptionResult:
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return TranscriptionResult(
                transcript="", error="OPENAI_API_KEY not found in environment"
            )

        _validate_pcm16_mono_24000(audio_data.format)

        wav_buffer = io.BytesIO()

        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(DEFAULT_OPENAI_OUTPUT_SAMPLE_RATE)
            wav_file.writeframes(audio_data.data)

        wav_buffer.seek(0)
        wav_data = wav_buffer.read()

        url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {api_key}"}
        files = {"file": ("audio.wav", wav_data, "audio/wav")}
        data = {"model": DEFAULT_WHISPER_MODEL}

        if config.language:
            lang_code = (
                config.language.split("-")[0].lower()
                if "-" in config.language
                else config.language.lower()
            )
            data["language"] = lang_code

        data.update(config.extra_options)

        response = requests.post(url, headers=headers, files=files, data=data)

        if response.status_code != 200:
            return TranscriptionResult(
                transcript="",
                error=f"OpenAI API error {response.status_code}: {response.text}",
            )

        result = response.json()
        return TranscriptionResult(transcript=result.get("text", ""))

    except Exception as e:
        return TranscriptionResult(
            transcript="", error=f"OpenAI transcription failed: {str(e)}"
        )


async def transcribe_gpt4o_realtime(
    audio_data: AudioData, config: TranscriptionConfig
) -> TranscriptionResult:
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return TranscriptionResult(
                transcript="", error="OPENAI_API_KEY not found in environment"
            )

        _validate_pcm16_mono_24000(audio_data.format)

        audio_base64 = base64.b64encode(audio_data.data).decode("utf-8")

        url = f"{DEFAULT_OPENAI_REALTIME_BASE_URL}?model=gpt-realtime"

        headers = {"Authorization": f"Bearer {api_key}", "OpenAI-Beta": "realtime=v1"}

        transcript = ""
        error_msg = None

        async with websockets.connect(url, additional_headers=headers) as ws:
            session_config = {
                "type": "session.update",
                "session": {
                    "modalities": ["text"],
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": {"model": config.model},
                    "turn_detection": {
                        "type": "server_vad",
                        "silence_duration_ms": config.openai_silence_duration_ms,
                    },
                },
            }

            await ws.send(json.dumps(session_config))

            await ws.send(
                json.dumps({"type": "input_audio_buffer.append", "audio": audio_base64})
            )
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

            while True:
                try:
                    event = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))

                    if (
                        event["type"]
                        == "conversation.item.input_audio_transcription.completed"
                    ):
                        transcript = event.get("transcript", "")
                        break
                    elif event["type"] == "error":
                        error_msg = f"OpenAI API error: {event}"
                        break
                except asyncio.TimeoutError:
                    break

            if error_msg:
                return TranscriptionResult(transcript="", error=error_msg)

            return TranscriptionResult(transcript=transcript)

    except Exception as e:
        return TranscriptionResult(
            transcript="", error=f"OpenAI Realtime transcription failed: {str(e)}"
        )


def transcribe_doubao(
    audio_data: AudioData, config: TranscriptionConfig
) -> TranscriptionResult:
    """Recognize delivered audio with Doubao's documented flash file API.

    Official protocol: docs.volcengine.com/docs/DoubaoVoice/recording-file-recognition-lite-http
    Status 20000000 is recognition success; 20000003 is a normal no-speech result.
    See docs.volcengine.com/docs/DoubaoVoice/error-code-query for status meanings.
    """
    from uuid import uuid4

    from tau2.config import DOUBAO_ASR_RESOURCE_ID, DOUBAO_ASR_URL

    key = os.environ.get("VOLCENGINE_ASR_API_KEY")
    if not key:
        return TranscriptionResult(
            transcript="", error="VOLCENGINE_ASR_API_KEY is required"
        )
    pcm = resample_audio(convert_to_mono(convert_to_pcm16(audio_data)), 16000)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm.data)
    response = requests.post(
        DOUBAO_ASR_URL,
        headers={
            "X-Api-Key": key,
            "X-Api-Resource-Id": DOUBAO_ASR_RESOURCE_ID,
            "X-Api-Request-Id": str(uuid4()),
            "X-Api-Sequence": "-1",
        },
        json={
            "user": {"uid": "tau-voice-demo"},
            "audio": {"data": base64.b64encode(buffer.getvalue()).decode()},
            "request": {"model_name": "bigmodel"},
        },
        timeout=60,
    )
    response.raise_for_status()
    status = response.headers.get("X-Api-Status-Code")
    if status == "20000003":
        return TranscriptionResult(transcript="", provider_status=status)
    if status != "20000000":
        return TranscriptionResult(transcript="", error=f"Doubao ASR status {status}")
    result = response.json().get("result", {})
    if not isinstance(result.get("text"), str):
        return TranscriptionResult(
            transcript="", error="Doubao ASR response has no text"
        )
    return TranscriptionResult(transcript=result["text"], provider_status=status)


def transcribe_scribe(
    audio_data: AudioData, config: TranscriptionConfig
) -> TranscriptionResult:
    """Recognize delivered WAV audio with the documented ModelHub Scribe route."""
    from tau2.utils.byted_modelhub import elevenlabs_request

    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(audio_data.format.sample_rate)
        wav.writeframes(audio_data.data)
    data = {"model_id": "scribe_v1"}
    if config.language:
        data["language_code"] = config.language.split("-")[0].lower()
    try:
        response = elevenlabs_request(
            "/speech-to-text",
            files={"file": ("delivered.wav", output.getvalue(), "audio/wav")},
            data=data,
        ).json()
        text = response.get("text")
        if not isinstance(text, str):
            raise ValueError("Missing Scribe transcript")
        return TranscriptionResult(transcript=text, provider_status="200")
    except Exception as exc:
        return TranscriptionResult(
            transcript="", error=f"Scribe transcription failed: {type(exc).__name__}"
        )
