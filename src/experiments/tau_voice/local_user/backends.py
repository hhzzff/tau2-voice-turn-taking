"""Streaming client for the existing Qwen3-TTS service."""

import httpx

from .tts_quality import normalize_date_whitespace


class ProviderError(RuntimeError):
    """Expose provider/status only, never credentials or raw response bodies."""

    def __init__(self, provider, status):
        super().__init__(f"{provider}: {status}")


def checked(response: httpx.Response, provider: str):
    """Raise a bounded transport error without logging response bodies."""
    if response.is_error:
        raise ProviderError(provider, response.status_code)


class QwenTTS:
    """Emit 20 ms PCM frames as the service produces them."""

    def __init__(self, cfg, client, rec=None):
        self.cfg, self.client, self.rec = cfg, client, rec

    async def stream(self, text):
        """Yield PCM16 blocks from the pinned voice service; reject rate mismatch and empty output."""

        def emit(kind, **values):
            if self.rec:
                self.rec.emit(kind, **values)

        normalized = (
            normalize_date_whitespace(text)
            if self.cfg.tts.normalize_date_whitespace
            else text
        )
        if normalized != text:
            emit("tts_text_normalized", original_text=text, tts_text=normalized)
        body = {
            "text": normalized,
            "speaker": self.cfg.tts.speaker,
            "language": "Chinese",
            "chunk_size": self.cfg.tts.stream_chunk_tokens,
            "seed": self.cfg.tts.seed,
            "reference_sha256": self.cfg.tts.reference_sha256,
        }
        emit(
            "tts_http_request",
            characters=len(normalized),
            sample_rate=self.cfg.audio.sample_rate,
        )
        total = 0
        async with self.client.stream("POST", self.cfg.tts.url, json=body) as response:
            emit("tts_http_headers", status=response.status_code)
            checked(response, "qwen3-tts")
            if response.headers.get("x-sample-rate") != str(self.cfg.audio.sample_rate):
                raise ProviderError("qwen3-tts", "unexpected_sample_rate")
            async for pcm in response.aiter_bytes(chunk_size=640):
                if len(pcm) % 2:
                    raise ProviderError("qwen3-tts", "unaligned_pcm")
                total += len(pcm)
                yield pcm
        if not total:
            raise ProviderError("qwen3-tts", "empty_audio")
        emit("tts_http_complete", bytes=total)
