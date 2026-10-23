"""Native playback adapted from SALMONN FDB; see SALMONN_PROVENANCE.json."""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
TTS_SENTENCE_SEPARATORS = (
    "。",
    "！",
    "？",
    ".",
    "!",
    "?",
    "\n",
    "*\n",
    "**\n",
    "\n\n",
    ".\n",
)


class CosyPromptRuntime(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_speech_feat: Any
    prompt_speech_token: Any
    spk_embedding: Any
    resample_24k_to_16k: Any


class _Sentence(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    embeddings: list[torch.Tensor] = Field(default_factory=list)
    sealed: bool = False


class FDBSpeechPlayback:
    """Queue only synthesized audio and return exactly one 16 kHz clock block.

    Both the saved FDB output and the model's self-audio must use ``step()``'s
    returned block. Audio that has not been played can therefore be cancelled
    without revising past waveform samples. Inference runtime does not advance
    this simulated clock.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device,
        data_args,
        *,
        runtime: CosyPromptRuntime | None = None,
        max_drain_iterations: int = 600,
        clause_v2: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.clause_v2 = bool(clause_v2)
        self.block_samples = int(round(INPUT_SAMPLE_RATE * float(data_args.delta)))
        if self.block_samples <= 0:
            raise ValueError("delta must produce a positive playback block")
        if int(data_args.num_text_tokens) <= 0:
            raise ValueError("num_text_tokens must be positive")
        self.lead_seconds = 0.2 if int(data_args.num_text_tokens) == 2 else 0.4
        self.max_drain_iterations = int(max_drain_iterations)
        if self.max_drain_iterations <= 0:
            raise ValueError("max_drain_iterations must be positive")
        if runtime is None:
            raise ValueError("Cosy prompt runtime must be explicit")
        self.runtime = runtime
        self.think_id = self._single_token_id(tokenizer.think_token)
        self.eot_id = self._single_token_id("<|eot_id|>")
        self.use_agent_id = tokenizer.get_vocab().get("<|use_agent|>")
        self.separator_ids = set()
        for separator in TTS_SENTENCE_SEPARATORS:
            token_ids = self._tokenize(separator)
            if token_ids:
                self.separator_ids.add(int(token_ids[-1]))
        # CosyVoice2's nonfinal flow output withholds this many codec tokens.
        self.lookahead_tokens = int(self.model.cosy.flow.pre_lookahead_len)
        self.codec_sample_rate = float(self.model.cosy.flow.input_frame_rate)
        if self.lookahead_tokens < 0 or self.codec_sample_rate <= 0:
            raise ValueError("invalid CosyVoice flow lookahead/frame rate")
        # HiFT withholds an overlap tail; avoid feeding a nonfinal chunk too
        # short to produce any audio beyond that tail.
        self.minimum_codec_chunk = max(
            1,
            math.floor(
                int(self.model.cosy.source_cache_len)
                * self.codec_sample_rate
                / OUTPUT_SAMPLE_RATE
            )
            + 1,
        )
        self.reset()

    def _tokenize(self, text):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _single_token_id(self, text):
        token_ids = self._tokenize(text)
        if len(token_ids) != 1:
            raise ValueError(f"{text!r} must encode to one token")
        return int(token_ids[0])

    def reset(self):
        self.active = False
        self.text_closed = False
        self._sentences: deque[_Sentence] = deque()
        self._audio: deque[torch.Tensor] = deque()
        self._audio_offset = 0
        self._lead_remaining = 0
        self._ticks = 0
        self._drain_iterations = 0
        self._reset_sentence_decoder()
        self._metadata = {
            "lead_seconds": self.lead_seconds,
            "flush_events": [],
            "tts_errors": [],
            "rendered_samples_16k": 0,
            "played_samples_16k": 0,
            "cancelled_samples_16k": 0,
            "underflow_samples_16k": 0,
            "interrupted": False,
        }

    def _reset_sentence_decoder(self):
        self._embedded_y = None
        self._generated = None
        self._text_index = 0
        self._last_sentence_length = 0
        self._codec_offset = 0
        self._hift_cache = None

    def begin_turn(self):
        if self.active and not self.drained:
            raise RuntimeError(
                "interrupt or drain the current speech before beginning another turn"
            )
        self.reset()
        self.active = True
        self._lead_remaining = int(round(self.lead_seconds * INPUT_SAMPLE_RATE))
        self._sentences.append(_Sentence())

    @property
    def metadata(self):
        return {
            **self._metadata,
            "flush_events": [dict(event) for event in self._metadata["flush_events"]],
            "tts_errors": [dict(error) for error in self._metadata["tts_errors"]],
            "playback_ticks": self._ticks,
            "tts_drain_iterations": self._drain_iterations,
            "text_closed": self.text_closed,
        }

    @property
    def _tts_pending(self):
        return any(sentence.embeddings for sentence in self._sentences)

    @property
    def drained(self):
        return not self.active or (
            self.text_closed and not self._tts_pending and not self._audio
        )

    def append_token(self, token_id: int, hidden_state: torch.Tensor | None):
        if not self.active or self.text_closed:
            raise RuntimeError("cannot append speech tokens outside an open turn")
        token_id = int(token_id)
        if token_id == self.use_agent_id:
            raise RuntimeError("no-agent FDB inference emitted a use-agent token")
        sentence = self._sentences[-1]
        if token_id != self.think_id or not self.clause_v2:
            if hidden_state is None:
                raise ValueError(
                    "spoken/EOT tokens require their native LM hidden state"
                )
            sentence.embeddings.append(hidden_state.detach())
        pending = len(sentence.embeddings)
        reason = None
        if self.clause_v2 and token_id == self.think_id and pending:
            reason = "think"
        elif token_id == self.eot_id:
            self.text_closed = True
            if pending:
                reason = "eot"
        elif token_id in self.separator_ids and pending > 15:
            reason = "punctuation"
        if reason:
            sentence.sealed = True
            self._metadata["flush_events"].append(
                {
                    "reason": reason,
                    "token_id": token_id,
                    "pending_tts_tokens": pending,
                    "tick": self._ticks,
                }
            )
            self._sentences.append(_Sentence())

    def interrupt(self):
        """Discard all unheard waveform/codecs and retain diagnostics."""
        self._metadata["cancelled_samples_16k"] += (
            sum(chunk.shape[-1] for chunk in self._audio) - self._audio_offset
        )
        self._metadata["interrupted"] = self._metadata["interrupted"] or self.active
        self.active = False
        self.text_closed = True
        self._sentences.clear()
        self._audio.clear()
        self._audio_offset = 0
        self._lead_remaining = 0
        self._reset_sentence_decoder()

    def _enqueue_waveform(self, waveform):
        waveform = self.runtime.resample_24k_to_16k(waveform.detach().float().cpu())
        waveform = waveform.detach().float().cpu().reshape(1, -1).contiguous()
        if waveform.shape[-1]:
            self._audio.append(waveform)
            self._metadata["rendered_samples_16k"] += waveform.shape[-1]

    def _vocode_available(self, complete: bool):
        codecs = self._generated or []
        if not codecs:
            if complete:
                raise RuntimeError(
                    "CosyVoice completed a nonempty sentence with no codecs"
                )
            return
        available_end = len(codecs) if complete else len(codecs) - self.lookahead_tokens
        if (
            not complete
            and available_end - self._codec_offset < self.minimum_codec_chunk
        ):
            return
        if available_end <= self._codec_offset and self._hift_cache is None:
            return
        # CosyVoice2 recomputes flow features on the available codec prefix,
        # then token_offset removes previously consumed features. HiFT's
        # overlap cache is handed back once; emitted audio is never rerendered.
        waveform, self._hift_cache, _ = self.model.cosy.tts_salmonn(
            codecs,
            self.runtime.spk_embedding,
            token_offset=self._codec_offset,
            flow_prompt_speech_token=self.runtime.prompt_speech_token,
            prompt_speech_feat=self.runtime.prompt_speech_feat,
            finalize=complete,
            hift_cache=self._hift_cache,
        )
        self._codec_offset = available_end
        self._enqueue_waveform(waveform)

    def _synthesize_step(self):
        if not self._tts_pending:
            return
        if self.text_closed:
            self._drain_iterations += 1
            if self._drain_iterations > self.max_drain_iterations:
                raise RuntimeError(
                    f"CosyVoice exceeded {self.max_drain_iterations} post-EOT drain iterations"
                )
        sentence = self._sentences[0]
        current_length = len(sentence.embeddings)
        text_chunk = int(self.model.cosy.llm.text_chunk)
        # Match evaluate_hf_new's live-prefix scheduling: a text boundary is
        # not required before Cosy can finish the available prefix. If it does,
        # completion below starts a fresh sentence for later LM tokens. In
        # clause-v2 mode THINK seals text without adding a hidden embedding, so
        # repeated THINKs keep draining that sentence without creating new ones.
        if (
            current_length % text_chunk != 0
            and current_length != self._last_sentence_length
        ):
            self._last_sentence_length = current_length
            return
        current = torch.cat(sentence.embeddings, dim=1)
        lengths = torch.tensor([current.shape[1]], device=current.device)
        projected = self.model.llm_syn_proj(current)
        kwargs = {"text_idx_num": self._text_index, "token_input": self._embedded_y}
        if self._embedded_y is not None:
            kwargs["generated"] = self._generated
        result = self.model.cosy.llm.inference_salmonn_stream(
            projected, lengths, **kwargs
        )
        self._embedded_y, self._generated, _, complete = result
        self._text_index += text_chunk
        self._last_sentence_length = current_length
        if complete:
            # Release GPU embeddings before the final vocoder allocation.
            sentence.embeddings.clear()
            del current, projected
        self._vocode_available(bool(complete))
        if complete:
            self._sentences.popleft()
            if not self._sentences:
                self._sentences.append(_Sentence())
            self._reset_sentence_decoder()

    @torch.inference_mode()
    def step(self) -> torch.Tensor:
        """Run one synthesis iteration and play exactly one audio block."""
        output = torch.zeros((1, self.block_samples), dtype=torch.float32)
        if not self.active:
            return output
        try:
            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                self._synthesize_step()
        except Exception as exc:
            self._metadata["tts_errors"].append(
                {"tick": self._ticks, "type": type(exc).__name__, "error": str(exc)}
            )
            raise
        self._ticks += 1
        position = min(self._lead_remaining, self.block_samples)
        self._lead_remaining -= position
        while position < self.block_samples and self._audio:
            chunk = self._audio[0]
            count = min(
                self.block_samples - position, chunk.shape[-1] - self._audio_offset
            )
            output[:, position : position + count] = chunk[
                :, self._audio_offset : self._audio_offset + count
            ]
            position += count
            self._audio_offset += count
            self._metadata["played_samples_16k"] += count
            if self._audio_offset == chunk.shape[-1]:
                self._audio.popleft()
                self._audio_offset = 0
        if position < self.block_samples and not self.drained:
            self._metadata["underflow_samples_16k"] += self.block_samples - position
        return output
