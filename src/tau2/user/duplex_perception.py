"""Observe only assistant audio already delivered to the user simulator."""

import json
import time
import wave
from pathlib import Path

from tau2.config import DEFAULT_WHISPER_MODEL
from tau2.data_model.audio import PCM_SAMPLE_RATE, AudioData, AudioEncoding, AudioFormat
from tau2.data_model.voice import TranscriptionConfig
from tau2.voice.transcription.transcribe import transcribe_audio
from tau2.voice.utils.audio_activity import is_audible_chunk
from tau2.voice.utils.audio_preprocessing import convert_to_pcm16, resample_audio


class PlayedAudioPerception:
    """Transcribe growing played audio prefixes at user decision checkpoints.

    Recognition may revise the last segment, so each snapshot replaces its text
    rather than appending ASR hypotheses. No generated/unplayed audio or model
    transcript is supplied to the recognizer. This synchronous baseline adds
    transcription latency; it is not a low-latency streaming ASR implementation.
    """

    def __init__(self, language: str):
        self.config = TranscriptionConfig(
            model=DEFAULT_WHISPER_MODEL,
            language="zh" if language == "Chinese" else "en",
        )
        self.segments = []
        self.snapshots = []
        self.current = None

    def start_user_turn(self):
        """Keep speech heard after a user utterance in a new history entry."""
        self.current = None

    def observe(self, chunk, start_seconds: float):
        """Append this delivered audio chunk, including quiet speech frames."""
        if chunk is None or not chunk.audio_content or chunk.audio_format is None:
            return
        ids = tuple(chunk.utterance_ids or [])
        # An idle padded tick has no item, text or energy. Do not repeatedly
        # transcribe growing trailing silence after an utterance has finished.
        if not ids and not chunk.content and not is_audible_chunk(chunk):
            return
        audio = resample_audio(
            convert_to_pcm16(
                AudioData(data=chunk.get_audio_bytes(), format=chunk.audio_format)
            ),
            PCM_SAMPLE_RATE,
        )
        if not audio.data:
            return
        if self.current is None or self.current["utterance_ids"] != ids:
            self.current = {
                "segment_id": len(self.segments),
                "utterance_ids": ids,
                "seconds": start_seconds,
                "audio": bytearray(),
                "recognized_bytes": 0,
                "text": "",
            }
            self.segments.append(self.current)
        self.current["audio"].extend(audio.data)
        self.current["audio_end_seconds"] = start_seconds + audio.duration

    def history(self, user_history: list, decision_seconds: float) -> list:
        """Return user turns and ASR text from the delivered assistant prefixes.

        Recognition failure blocks this decision. Falling back to proportional
        text would silently reintroduce the same perception error.
        """
        heard = []
        for segment in self.segments:
            data = bytes(segment["audio"])
            if len(data) != segment["recognized_bytes"]:
                started = time.perf_counter()
                result = transcribe_audio(
                    AudioData(data=data, format=self.audio_format), self.config
                )
                row = {
                    "segment_id": segment["segment_id"],
                    "decision_seconds": decision_seconds,
                    "audio_start_seconds": segment["seconds"],
                    "audio_end_seconds": segment["audio_end_seconds"],
                    "audio_bytes": len(data),
                    "audio_file": f"heard_audio/assistant_{segment['segment_id']:03d}.wav",
                    "model": self.config.model,
                    "transcript": result.transcript,
                    "error": result.error,
                    "generation_time_seconds": time.perf_counter() - started,
                }
                self.snapshots.append(row)
                if result.error:
                    raise RuntimeError("Played assistant audio transcription failed")
                segment["recognized_bytes"] = len(data)
                segment["text"] = result.transcript
            heard.append(
                {
                    "role": "assistant",
                    "seconds": segment["seconds"],
                    "audio_end_seconds": segment["audio_end_seconds"],
                    "text": segment["text"],
                    "text_source": "asr_of_played_audio",
                    "segment_id": segment["segment_id"],
                }
            )
        users = [dict(entry) for entry in user_history if entry["role"] == "user"]
        return sorted(users + heard, key=lambda entry: entry["seconds"])[-60:]

    @property
    def audio_format(self) -> AudioFormat:
        """The stored mono PCM format used by every perception snapshot."""
        return AudioFormat(
            encoding=AudioEncoding.PCM_S16LE, sample_rate=PCM_SAMPLE_RATE
        )

    def save(self, output_dir: Path):
        """Save recognition results and audio for reproducing each prefix."""
        audio_dir = output_dir / "heard_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        for segment in self.segments:
            # Both simulator.stop() and the runner's finally block save reports.
            # These files belong to this perception instance and may grow between
            # checkpoints; replace them atomically instead of failing on reuse.
            target = audio_dir / f"assistant_{segment['segment_id']:03d}.wav"
            pending = target.with_suffix(".pending.wav")
            with wave.open(str(pending), "wb") as audio:
                audio.setnchannels(self.audio_format.channels)
                audio.setsampwidth(self.audio_format.sample_width)
                audio.setframerate(self.audio_format.sample_rate)
                audio.writeframes(bytes(segment["audio"]))
            pending.replace(target)
        (output_dir / "perception.json").write_text(
            json.dumps(
                {
                    "source": "asr_of_played_audio",
                    "note": "Each snapshot uses only the first audio_bytes of its audio_file. Recognition text can be revised. This is checkpoint ASR, not word alignment.",
                    "snapshots": self.snapshots,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
