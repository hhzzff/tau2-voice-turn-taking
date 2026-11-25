"""Real-time audio clocks independent of user inference and network reception."""

import asyncio

from tau2.user.local.recording import now


class ContinuousVoiceDriver:
    """No discrete-time freeze: each task observes the same monotonic clock.

    Playback has its own task, so even transport backpressure cannot suspend
    listening. Missed deadlines are logged and never replayed in a fast burst.
    """

    def __init__(self, user, target, recorder):
        self.user, self.target, self.rec = user, target, recorder
        self.state = user.get_init_state()
        self.tasks = []
        self.tick_counts = {"playback": 0, "send": 0}

    async def _clock(self, name, action):
        deadline = now()
        while True:
            await asyncio.sleep(max(0, (deadline - now()) / 1e9))
            at = now()
            late = (at - deadline) / 1e6
            if late > 5:
                self.rec.emit("audio_clock_late", clock=name, lateness_ms=late)
            await action(at)
            self.tick_counts[name] += 1
            if self.tick_counts[name] % 50 == 0:
                self.rec.emit(
                    "audio_clock_progress", clock=name, ticks=self.tick_counts[name]
                )
            deadline = max(deadline + 20_000_000, at + 19_000_000)
            if now() > deadline:
                deadline = now() + 20_000_000

    async def _play(self, at):
        self.target.play_frame(at)

    async def _send(self, at):
        chunk, self.state = self.user.get_next_chunk(self.state)
        sent = await self.target.send_packet(self.user.packet)
        self.user.acknowledge_delivery(chunk.chunk_id, sent)

    async def run(self, timeout):
        """Run until scenario end, failure, or a real wall-clock deadline."""
        runtime = self.user.runtime
        self.tasks = [
            asyncio.create_task(self.target.receive(), name="target-receive"),
            asyncio.create_task(
                self._clock("playback", self._play), name="audio-playback"
            ),
            asyncio.create_task(self._clock("send", self._send), name="audio-send"),
        ]
        try:
            done, _ = await asyncio.wait(
                [*self.tasks, *runtime.tasks],
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                return "timeout"
            for task in done:
                task.result()
            if runtime.control_task not in done:
                raise RuntimeError("Background audio task ended unexpectedly")
            return "ok"
        finally:
            for task in self.tasks:
                task.cancel()
            await asyncio.gather(*self.tasks, return_exceptions=True)
