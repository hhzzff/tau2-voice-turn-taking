"""X-ASR native online transducer, with decoding off the acquisition event loop."""

from __future__ import annotations

import asyncio
import re

import numpy as np

from .recording import Transcript, now


class XASR:
    def __init__(self, cfg, rec):
        self.cfg, self.rec = cfg, rec
        self.recognizer = self.stream = None
        self.q = asyncio.Queue()
        self.revision = 0
        self.segment = 0
        self.last_text = ""

    async def reset(self, *, preserve_events=False):
        """Reuse resident weights after a completed segment; never reload the model."""
        self.stream = await asyncio.to_thread(self.recognizer.create_stream)
        self.segment += 1
        self.revision = 0
        self.last_text = ""
        if not preserve_events:
            self.q = asyncio.Queue()
        self.rec.emit("asr_segment_start", segment=self.segment)

    async def start(self):
        def load():
            import sherpa_onnx

            root = self.cfg.asr.model_path
            files = {
                key: root / name
                for key, name in {
                    "encoder": "encoder-160ms.onnx",
                    "decoder": "decoder-160ms.onnx",
                    "joiner": "joiner-160ms.onnx",
                    "tokens": "tokens.txt",
                }.items()
            }
            for path in files.values():
                if not path.is_file():
                    raise FileNotFoundError(path)
            self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                **{key: str(value) for key, value in files.items()},
                num_threads=self.cfg.asr.num_threads,
                sample_rate=self.cfg.audio.sample_rate,
                feature_dim=80,
                decoding_method="greedy_search",
                model_type="zipformer2",
                provider=self.cfg.asr.execution_provider,
                enable_endpoint_detection=False,
            )
            self.stream = self.recognizer.create_stream()

        await asyncio.to_thread(load)
        self.rec.emit(
            "asr_connected",
            provider="x-asr",
            model=self.cfg.asr.model,
            chunk_ms=160,
            execution_provider=self.cfg.asr.execution_provider,
            genuine_streaming=True,
            word_timestamps_available=False,
        )

    def _decode(self, pcm=None, final=False):
        begin = now()
        if pcm is not None:
            samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
            self.stream.accept_waveform(self.cfg.audio.sample_rate, samples)
        if final:
            self.stream.input_finished()
        steps = 0
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
            steps += 1
        text = self.recognizer.get_result(self.stream)
        text = re.sub(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", "", text)
        text = re.sub(r"\s+([，。！？,.!?])", r"\1", text).strip()
        return text, steps, begin, now()

    def _publish(self, result, final=False):
        text, steps, begin, end = result
        if steps:
            self.rec.emit(
                "asr_compute",
                at=end,
                duration_ms=(end - begin) / 1e6,
                decode_steps=steps,
            )
        if text and (text != self.last_text or final):
            self.last_text = text
            self.revision += 1
            self.q.put_nowait(
                Transcript(
                    self.rec.run_id,
                    f"xasr:{self.segment}",
                    self.revision,
                    text,
                    final,
                    None,
                    end,
                )
            )

    async def feed(self, frame):
        self._publish(await asyncio.to_thread(self._decode, frame.pcm))

    async def finish(self, *, padding_ms=0, end_events=True):
        self.rec.emit("asr_flush_submit", decoder_padding_ms=padding_ms)
        if padding_ms:
            # Algorithmic right-context padding, not additional captured audio or clock time.
            padding = bytes(self.cfg.audio.sample_rate * padding_ms // 1000 * 2)
            self._publish(await asyncio.to_thread(self._decode, padding))
        self._publish(await asyncio.to_thread(self._decode, None, True), final=True)
        if end_events:
            self.q.put_nowait(None)

    async def events(self):
        while (event := await self.q.get()) is not None:
            yield event

    async def close(self):
        self.stream = self.recognizer = None
