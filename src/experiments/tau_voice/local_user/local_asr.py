"""X-ASR native online transducer, with decoding off the acquisition event loop."""

from __future__ import annotations

import asyncio
import re

import numpy as np

from .recording import Transcript, now


class XASR:
    """Own one online recognizer and publish revisions of actually played PCM."""

    def __init__(self, cfg, rec):
        self.cfg, self.rec = cfg, rec
        self.recognizer = self.stream = None
        self.q = asyncio.Queue()
        self.revision = 0
        self.segment = 0
        self.last_text = ""
        self._workers = set()

    async def _run_worker(self, operation, *args):
        # Cancelling to_thread's awaiter does not stop native decoding. Retain
        # the worker until close() can join it before releasing native handles.
        worker = asyncio.create_task(asyncio.to_thread(operation, *args))
        self._workers.add(worker)
        try:
            result = await asyncio.shield(worker)
        except asyncio.CancelledError:
            raise  # close() must still retrieve the worker result or exception.
        except BaseException:
            self._workers.discard(worker)
            raise
        else:
            self._workers.discard(worker)
            return result

    async def reset(self, *, preserve_events=False):
        """Reuse resident weights after a completed segment; never reload the model."""
        self.stream = await self._run_worker(self.recognizer.create_stream)
        self.segment += 1
        self.revision = 0
        self.last_text = ""
        if not preserve_events:
            self.q = asyncio.Queue()
        self.rec.emit("asr_segment_start", segment=self.segment)

    async def start(self):
        """Load resident model weights off-loop and create the first online stream."""

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

        await self._run_worker(load)
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
                    session_id=self.rec.run_id,
                    event_id=f"xasr:{self.segment}",
                    revision=self.revision,
                    text=text,
                    is_final=final,
                    audio_end_ms=None,
                    received_mono_ns=end,
                )
            )

    async def feed(self, frame):
        """Decode played PCM off-loop and publish only completed observations."""
        self._publish(await self._run_worker(self._decode, frame.pcm))

    async def finish(self, *, padding_ms=0, end_events=True):
        """Flush decoder right context without treating padding as captured audio."""
        self.rec.emit("asr_flush_submit", decoder_padding_ms=padding_ms)
        if padding_ms:
            # Algorithmic right-context padding, not additional captured audio or clock time.
            padding = bytes(self.cfg.audio.sample_rate * padding_ms // 1000 * 2)
            self._publish(await self._run_worker(self._decode, padding))
        self._publish(await self._run_worker(self._decode, None, True), final=True)
        if end_events:
            self.q.put_nowait(None)

    async def events(self):
        """Yield local ASR revisions until the producer sends its terminal event."""
        while (event := await self.q.get()) is not None:
            yield event

    async def close(self):
        """Join in-flight native work before releasing the recognizer and stream."""
        try:
            results = await asyncio.gather(*self._workers, return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    raise result
        finally:
            self._workers.clear()
            self.stream = self.recognizer = None
