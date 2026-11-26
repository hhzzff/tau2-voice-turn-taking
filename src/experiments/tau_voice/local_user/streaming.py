"""Bounded text -> synthesis -> PCM pipeline shared by simulator and recordings."""

from __future__ import annotations

import asyncio
from contextlib import aclosing

from .audio import first_effective
from .backends import ProviderError
from .tts_quality import PauseMonitor


def update_spoken_text(turn):
    """Complete transmitted clauses are confirmed; open/partial clauses are uncertain."""
    sent = turn.get("sent_source_samples", 0)
    segments = turn.get("speech_segments", [])
    completed = [
        s
        for s in segments
        if s.get("output_end_sample") is not None and s["output_end_sample"] <= sent
    ]
    turn["spoken_text"] = "".join(s["text"] for s in completed)
    pending = [
        s
        for s in segments
        if s["output_start_sample"] < sent
        and (s.get("output_end_sample") is None or sent < s["output_end_sample"])
    ]
    turn["unconfirmed_spoken_tail"] = pending[0]["text"] if pending else ""


class LeadingSilence:
    """Remove only the initial quiet prefix, retaining 10 ms before detected onset.

    No bytes after onset are discarded. Callers log raw TTS timing before feeding
    this gate, so onset trimming cannot masquerade as faster model generation.
    """

    def __init__(self, threshold, sample_rate=16000):
        self.threshold = threshold
        self.keep_bytes = sample_rate // 100 * 2
        self.pending = b""
        self.started = False
        self.removed_samples = 0

    def feed(self, pcm):
        if self.started:
            return pcm
        data = self.pending + pcm
        onset = first_effective(data, self.threshold)
        cut = (
            max(0, onset * 2 - self.keep_bytes)
            if onset is not None
            else max(0, len(data) - self.keep_bytes)
        )
        self.removed_samples += cut // 2
        if onset is None:
            self.pending = data[cut:]
            return b""
        self.started = True
        self.pending = b""
        return data[cut:]


class SpeechStream:
    def __init__(
        self,
        cfg,
        llm,
        tts,
        emit,
        *,
        snapshot=None,
        action=None,
        history=None,
        text=None,
    ):
        self.cfg, self.llm, self.tts, self.emit = cfg, llm, tts, emit
        self.snapshot, self.action, self.history = snapshot, action, history
        self.prepared_text, self.text = text, text or ""

    async def audio(self):
        text_queue, pcm_queue = asyncio.Queue(maxsize=2), asyncio.Queue(maxsize=32)
        step = self.cfg.audio.sample_rate * self.cfg.audio.frame_ms // 1000 * 2

        async def read_text():
            async def source():
                if self.prepared_text is not None:
                    yield self.prepared_text
                elif self.cfg.llm.synthesis_unit == "clause":
                    async with aclosing(
                        self.llm.segments(
                            self.snapshot,
                            self.action,
                            history=self.history,
                            split_clauses=True,
                        )
                    ) as stream:
                        async for part in stream:
                            yield part
                else:
                    yield await self.llm.generate(
                        self.snapshot, self.action, history=self.history
                    )

            async with aclosing(source()) as stream:
                async for index, part in enumerate_async(stream):
                    if self.prepared_text is None:
                        self.text += part
                        if index == 0:
                            self.emit(
                                "llm_usable",
                                text=part,
                                synthesis_unit=self.cfg.llm.synthesis_unit,
                            )
                        self.emit("llm_segment_ready", segment=index, text=part)
                    await text_queue.put((index, part))
            if self.prepared_text is None:
                self.emit("llm_complete", text=self.text)
            await text_queue.put(None)

        async def synthesize():
            first_pcm = first_voice = True
            pending_pcm = b""
            output_samples = 0
            while (item := await text_queue.get()) is not None:
                index, text = item
                segment_start = output_samples
                if index == 0:
                    self.emit("tts_submit")
                self.emit(
                    "tts_segment_submit",
                    segment=index,
                    text=text,
                    output_start_sample=segment_start,
                )
                gate = LeadingSilence(
                    self.cfg.audio.effective_rms, self.cfg.audio.sample_rate
                )
                pauses = PauseMonitor(
                    self.cfg.audio.sample_rate,
                    self.cfg.audio.effective_rms,
                    self.cfg.tts.pause_warning_ms,
                )
                async with aclosing(self.tts.stream(text)) as stream:
                    async for pcm in stream:
                        if first_pcm:
                            self.emit("tts_first_pcm")
                            first_pcm = False
                        if (
                            first_voice
                            and first_effective(pcm, self.cfg.audio.effective_rms)
                            is not None
                        ):
                            self.emit("tts_first_voice")
                            first_voice = False
                        for warning in pauses.feed(pcm):
                            self.emit("tts_pause_warning", segment=index, **warning)
                        if self.cfg.tts.trim_leading_silence:
                            pcm = gate.feed(pcm)
                        output_samples += len(pcm) // 2
                        # Reblock across TTS chunks/clauses: only the final frame may be short.
                        # Otherwise the paced sender's final-frame padding would create inner gaps.
                        pending_pcm += pcm
                        while len(pending_pcm) >= step:
                            await pcm_queue.put(pending_pcm[:step])
                            pending_pcm = pending_pcm[step:]
                self.emit("tts_audio_quality", segment=index, **pauses.finish())
                self.emit(
                    "tts_segment_complete",
                    segment=index,
                    removed_leading_samples=gate.removed_samples,
                    output_start_sample=segment_start,
                    output_end_sample=output_samples,
                    text=text,
                )
            if first_voice:
                raise ProviderError("tts", "silent_output")
            if pending_pcm:
                await pcm_queue.put(pending_pcm)
            self.emit("tts_complete")

        async def produce():
            try:
                async with asyncio.TaskGroup() as group:
                    group.create_task(read_text())
                    group.create_task(synthesize())
            except Exception as exc:
                await pcm_queue.put(exc)
            else:
                await pcm_queue.put(None)

        producer = asyncio.create_task(produce())
        try:
            while (item := await pcm_queue.get()) is not None:
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)


async def enumerate_async(source):
    index = 0
    async for item in source:
        yield index, item
        index += 1
