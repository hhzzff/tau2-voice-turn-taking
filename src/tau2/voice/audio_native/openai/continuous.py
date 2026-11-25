"""OpenAI event/playback adapter for a continuously running duplex experiment."""

import base64
import hashlib
from dataclasses import dataclass, field

from tau2.config import DEFAULT_LOCAL_USER_TARGET_OUTPUT_TOKENS
from tau2.data_model.audio import AudioEncoding, AudioFormat, audio_bytes_to_string
from tau2.data_model.message import AssistantMessage
from tau2.user.local.audio import first_effective
from tau2.user.local.observation import HeardItem
from tau2.user.local.recording import now
from tau2.user.local_streaming_user import LOCAL_PCM
from tau2.voice.audio_native.audio_converter import StreamingPCMConverter
from tau2.voice.audio_native.openai.provider import OpenAIVADConfig, OpenAIVADMode


@dataclass
class OutputBuffer:
    """Unplayed samples are private to the transport, never user context."""

    heard: HeardItem
    content_index: int = 0
    pcm: bytearray = field(default_factory=bytearray)
    converter: StreamingPCMConverter = field(default_factory=StreamingPCMConverter)

    def add(self, pcm=b"", final=False):
        if self.heard.cancelled or self.heard.complete:
            return
        self.pcm.extend(self.converter.convert_output(pcm, final=final))
        self.heard.buffered_samples = len(self.pcm) // 2
        self.heard.complete = final


class ContinuousOpenAITarget:
    """Reuse the tau provider; cancellation follows server VAD, never B intent.

    Remote transcripts remain audit-only. The user's input is exactly the PCM
    released by play_frame at the local listening clock, including underruns.
    """

    def __init__(self, provider, user, rec):
        self.provider, self.user, self.rec = provider, user, rec
        self.runtime = user.runtime
        self.playback = self.runtime.playback
        self.playback.model = provider.model
        self.playback.causal_replies = True
        self.buffers = {}
        self.transcripts = {}
        self.first_voice = set()
        self.last_input = {"for_turn": 0, "speech_kind": "main"}
        self.input_converter = StreamingPCMConverter()
        self.cancelled_responses = set()

    async def start(
        self, instructions, max_output_tokens=DEFAULT_LOCAL_USER_TARGET_OUTPUT_TOKENS
    ):
        """Await an acknowledged native-PCM configuration before starting clocks."""
        await self.provider.connect()
        await self.provider.configure_session(
            instructions,
            [],
            OpenAIVADConfig(mode=OpenAIVADMode.SEMANTIC_VAD, eagerness="medium"),
            modality="audio",
            audio_format=AudioFormat(
                encoding=AudioEncoding.PCM_S16LE, sample_rate=24000
            ),
            voice="marin",
            transcription_language=None,
            max_output_tokens=max_output_tokens,
        )
        self.rec.emit(
            "target_session",
            session={
                k: self.provider.session_config.get(k)
                for k in ("id", "model", "audio", "max_output_tokens", "instructions")
            },
        )

    async def receive(self):
        """Receive continuously, independent of ASR, generation and playback."""
        async for event in self.provider.receive_raw_events():
            await self._process_event(event)
        raise RuntimeError("Target WebSocket ended before the trial")

    async def _process_event(self, event):
        kind, at = event["type"], now()
        if kind == "error":
            error = {
                k: event.get("error", {}).get(k) for k in ("type", "code", "param")
            }
            self.rec.emit("target_error", error=error)
            raise RuntimeError(f"Target error: {error}")
        if kind == "response.created":
            response = event["response"]
            self.playback.responses[response["id"]] = {
                "id": response["id"],
                "created_ns": at,
                **self.last_input,
            }
        elif kind in {"response.output_audio.delta", "response.audio.delta"}:
            pcm = base64.b64decode(event["delta"], validate=True)
            item_id, rid = event["item_id"], event["response_id"]
            if item_id not in self.buffers:
                meta = self.playback.responses.get(rid, self.last_input)
                heard = HeardItem(item_id, rid, meta["for_turn"])
                heard.cancelled = rid in self.cancelled_responses
                self.playback.items[item_id] = heard
                self.buffers[item_id] = OutputBuffer(
                    heard, event.get("content_index", 0)
                )
                self.rec.emit(
                    "target_first_pcm_rx",
                    item_id=item_id,
                    response_id=rid,
                    turn=heard.for_turn,
                )
            self.buffers[item_id].add(pcm)
            self.rec.emit(
                "target_audio_chunk",
                at=at,
                item_id=item_id,
                bytes=len(pcm),
                sha256=hashlib.sha256(pcm).hexdigest(),
            )
            return
        elif kind in {"response.output_audio.done", "response.audio.done"}:
            if output := self.buffers.get(event.get("item_id")):
                output.add(final=True)
        elif kind == "response.done":
            response = event["response"]
            entry = self.playback.responses.setdefault(
                response["id"], dict(self.last_input)
            )
            entry.update(
                done_ns=at,
                status=response["status"],
                status_details=response.get("status_details"),
                usage=response.get("usage"),
            )
            for output in self.buffers.values():
                if output.heard.response_id == response["id"]:
                    output.add(final=True)
        elif kind in {
            "response.output_audio_transcript.done",
            "response.audio_transcript.done",
        }:
            self.transcripts[event["item_id"]] = event.get("transcript", "")
        elif kind == "input_audio_buffer.speech_started":
            self.rec.emit(
                "target_vad_speech_started",
                at=at,
                turn=self.runtime.sent_turn,
                audio_start_ms=event.get("audio_start_ms"),
            )
            for rid, response in self.playback.responses.items():
                if not response.get("done_ns"):
                    self.cancelled_responses.add(rid)
            for output in self.buffers.values():
                item = output.heard
                if item.ended or item.cancelled:
                    continue
                item.cancelled = True
                discarded = len(output.pcm)
                output.pcm.clear()
                item.buffered_samples = 0
                self.end_item(item, "server_vad")
                self.rec.emit(
                    "target_playback_stop",
                    item_id=item.item_id,
                    turn=self.runtime.sent_turn,
                    played_ms=item.played / 16,
                    discarded_bytes=discarded,
                    source="server_vad",
                )
                await self.provider.truncate_item(
                    item.item_id, output.content_index, item.played // 16
                )
        self.rec.emit("target_event", at=at, event=event)

    async def send_packet(self, packet):
        """Submit audio and audit the native bytes; host then issues a receipt."""
        raw = self.input_converter.convert_input(packet["pcm"])
        turn = packet["turn"]
        # The API may receive/respond before send() returns: attribution is an
        # explicit offered-input tag, separate from the successful receipt.
        if turn and packet["voice"]:
            self.last_input = {
                "for_turn": turn["index"]
                if not turn.get("non_main_input")
                else self.runtime.sent_turn,
                "speech_kind": turn.get("kind", "main"),
            }
            if self.last_input["speech_kind"] == "held_speech":
                self.last_input["speech_kind"] = "main"
        await self.provider.send_audio(raw)
        stamp = now()
        self.rec.emit(
            "transport_audio_submit",
            sequence=packet["sequence"],
            sent_ns=stamp,
            native_samples=len(raw) // 2,
            input_samples=len(packet["pcm"]) // 2,
            native_sha256=hashlib.sha256(raw).hexdigest(),
        )
        return stamp

    def finish_playback(self):
        """Flush only already-played audio when a trial ends mid-utterance."""
        for output in self.buffers.values():
            item = output.heard
            if item.started_ns is not None and not item.ended:
                self.end_item(item, "trial_end")

    def end_item(self, item, reason):
        """End only what was actually audible; unplayed transcripts stay private."""
        if item.ended:
            return
        item.ended = True
        self.playback.last_end_ns = now()
        if item.started_ns is not None:
            self.runtime.end_agent(item, reason)
        self.rec.emit(
            "target_playback_end",
            item_id=item.item_id,
            response_id=item.response_id,
            turn=item.for_turn,
            reason=reason,
            played_samples=item.played,
        )

    def play_frame(self, at):
        """Release at most 20 ms, or an audible underrun, per real clock tick."""
        for output in self.buffers.values():
            item = output.heard
            if item.ended or item.cancelled:
                continue
            if len(output.pcm) < 640 and not item.complete:
                if item.started_ns is None:
                    return
                pcm = bytes(640)
                self.rec.emit("target_playback_underrun", item_id=item.item_id)
            elif not output.pcm:
                self.end_item(item, "complete")
                continue
            else:
                pcm = bytes(output.pcm[:640])
                del output.pcm[:640]
                item.buffered_samples = len(output.pcm) // 2
                item.played += len(pcm) // 2
            if item.started_ns is None:
                item.started_ns = at
                self.runtime.begin_agent(item)
                self.rec.emit(
                    "target_first_play",
                    item_id=item.item_id,
                    response_id=item.response_id,
                    turn=item.for_turn,
                    at=at,
                )
            if first_effective(pcm, 0.005) is not None:
                self.playback.last_voice_ns = at
                if item.response_id not in self.first_voice:
                    self.first_voice.add(item.response_id)
                    self.rec.emit(
                        "target_first_voice_play",
                        item_id=item.item_id,
                        response_id=item.response_id,
                        turn=item.for_turn,
                        at=at,
                    )
            self.user.observe_played(
                AssistantMessage(
                    role="assistant",
                    content=None,
                    audio_format=LOCAL_PCM,
                    audio_content=audio_bytes_to_string(pcm),
                    utterance_ids=[item.item_id],
                    raw_data={"played_ns": at},
                    is_final_chunk=False,
                )
            )
            if item.complete and not output.pcm:
                self.end_item(item, "complete")
            return
