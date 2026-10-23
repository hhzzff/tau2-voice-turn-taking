"""Causal frontend adapted from pi-bench; see SALMONN_PROVENANCE.json.

Only stable features of audio already received are exposed to the decoder.
No continuation controller or complete-recording inference is included.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from types import MethodType
from typing import Any

BRIDGE_SCHEMA_VERSION = "salmonn-causal-speech-bridge-v2"
SAMPLE_RATE = 16000
FBANK_FRAME_LENGTH_SAMPLES = 400
FBANK_FRAME_SHIFT_SAMPLES = 160
FBANK_LEFT_REFLECTION_SAMPLES = 120
SUBSAMPLING_FACTOR = 2
SPEECH_PROJECTION_GROUP = 4


def conv2_subsampled_frame_count(total_fbank_frames: int) -> int:
    """Return the external factor-two Conv2dSubsampling output length."""
    if total_fbank_frames < 0:
        raise ValueError("total_fbank_frames must be non-negative")
    return max(0, (total_fbank_frames - 1) // SUBSAMPLING_FACTOR)


def stable_subsampled_frame_count(total_fbank_frames: int) -> int:
    """Return outputs independent of the Conv2d right-boundary zero pad.

    Full output ``j`` depends on raw time indices ``[2*j-1, 2*j+3]``.
    It is stable exactly when ``2*j+3 <= total_fbank_frames-1``.  The input to
    this function must itself be a stable Fbank prefix, not the reflected last
    frame returned by an offline ``snip_edges=False`` extraction.
    """
    if total_fbank_frames < 0:
        raise ValueError("total_fbank_frames must be non-negative")
    return max(0, (total_fbank_frames - 2) // SUBSAMPLING_FACTOR)


def canonical_stable_fbank_frame_count(total_waveform_samples: int) -> int:
    """Return canonical Lhotse frames complete without right-end reflection.

    The audited Fbank uses a 400-sample window, a 160-sample shift and
    ``snip_edges=False``.  Its streaming implementation applies the same
    120-sample left reflection as offline extraction, but retains the final
    incomplete window instead of reflecting the current right boundary.
    """
    if total_waveform_samples < 0:
        raise ValueError("total_waveform_samples must be non-negative")
    return max(
        0,
        (total_waveform_samples - FBANK_LEFT_REFLECTION_SAMPLES)
        // FBANK_FRAME_SHIFT_SAMPLES,
    )


def raw_receptive_interval(subsampled_index: int) -> tuple[int, int]:
    """Inclusive raw-fbank receptive interval for a global output index."""
    if subsampled_index < 0:
        raise ValueError("subsampled_index must be non-negative")
    return 2 * subsampled_index - 1, 2 * subsampled_index + 3


@dataclass(frozen=True)
class TailRecomputePlan:
    previous_stable_frames: int
    total_fbank_frames: int
    stable_total_frames: int
    tail_start_fbank_frame: int
    local_output_start: int
    new_stable_frames: int


def tail_recompute_plan(
    previous_stable_frames: int,
    total_fbank_frames: int,
) -> TailRecomputePlan:
    """Plan a small exact subsampler tail aligned to the global stride phase."""
    if previous_stable_frames < 0:
        raise ValueError("previous_stable_frames must be non-negative")
    stable_total = stable_subsampled_frame_count(total_fbank_frames)
    if previous_stable_frames > stable_total:
        raise ValueError("previous stable frame count exceeds current stable prefix")

    if previous_stable_frames == 0:
        tail_start = 0
        local_start = 0
    else:
        # Include one prior local output.  Dropping that output removes the
        # artificial left boundary pad while retaining its real raw context.
        tail_start = max(0, 2 * previous_stable_frames - 2)
        local_start = previous_stable_frames - tail_start // 2

    return TailRecomputePlan(
        previous_stable_frames=previous_stable_frames,
        total_fbank_frames=total_fbank_frames,
        stable_total_frames=stable_total,
        tail_start_fbank_frame=tail_start,
        local_output_start=local_start,
        new_stable_frames=stable_total - previous_stable_frames,
    )


def _tail_subsample(
    encoder_embed: Any,
    all_fbank: Any,
    plan: TailRecomputePlan,
    torch_module: Any,
) -> Any | None:
    if plan.new_stable_frames == 0:
        return None
    tail = all_fbank[:, plan.tail_start_fbank_frame :]
    tail_lens = torch_module.tensor(
        [int(tail.shape[1])],
        dtype=torch_module.long,
        device=tail.device,
    )
    local_outputs, _ = encoder_embed(tail, tail_lens)
    stop = plan.local_output_start + plan.new_stable_frames
    if int(local_outputs.shape[1]) < stop:
        raise RuntimeError(
            "subsampler tail returned too few frames: "
            f"available={local_outputs.shape[1]} required={stop}"
        )
    return local_outputs[:, plan.local_output_start : stop]


class CausalTransformerCacheBridge:
    """Incrementally call the external causal transformer with DynamicCache."""

    def __init__(
        self, causal_audio_encoder: Any, torch_module: Any | None = None
    ) -> None:
        if torch_module is None:
            import torch as torch_module  # type: ignore[no-redef]
        from transformers.cache_utils import DynamicCache

        self.torch = torch_module
        self.encoder = causal_audio_encoder.encoder
        if not bool(getattr(self.encoder, "is_causal", False)):
            raise ValueError("SALMONN transformer_v2 encoder is not causal")
        self.encoder.eval()
        self.cache = DynamicCache()
        self.position = 0

    def _attention_mask(self, hidden_states: Any, positions: Any) -> Any | None:
        implementation = str(
            getattr(
                self.encoder.layers[0].self_attn.config, "_attn_implementation", "eager"
            )
        )
        if implementation == "flash_attention_2":
            # All rows are valid and FlashAttention reads module.is_causal.
            return None

        total_keys = int(positions[-1].item()) + 1
        key_positions = self.torch.arange(
            total_keys,
            dtype=positions.dtype,
            device=positions.device,
        )
        blocked = key_positions.unsqueeze(0) > positions.unsqueeze(1)
        mask = self.torch.zeros(
            (int(positions.shape[0]), total_keys),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        mask.masked_fill_(blocked, self.torch.finfo(hidden_states.dtype).min)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward_chunk(self, embedded_frames: Any) -> Any:
        if self.torch.is_grad_enabled():
            raise RuntimeError(
                "causal transformer append requires inference_mode/no_grad"
            )
        if embedded_frames.ndim != 3 or int(embedded_frames.shape[0]) != 1:
            raise ValueError("causal speech bridge supports batch size one only")
        query_frames = int(embedded_frames.shape[1])
        if query_frames <= 0:
            return embedded_frames

        positions = self.torch.arange(
            self.position,
            self.position + query_frames,
            dtype=self.torch.long,
            device=embedded_frames.device,
        )
        position_ids = positions.unsqueeze(0)
        hidden_states = embedded_frames
        position_embeddings = self.encoder.rotary_emb(hidden_states, position_ids)
        attention_mask = self._attention_mask(hidden_states, positions)

        for layer in self.encoder.layers[: self.encoder.config.num_hidden_layers]:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=self.cache,
                use_cache=True,
                cache_position=positions,
                position_embeddings=position_embeddings,
            )[0]

        hidden_states = self.encoder.norm(hidden_states)
        self.position += query_frames
        cache_length = int(self.cache.get_seq_length())
        if cache_length != self.position:
            raise RuntimeError(
                f"speech encoder cache length {cache_length} != position {self.position}"
            )
        return hidden_states


class SpeechProjectionAccumulator:
    """Apply the checkpoint's 4-frame speech-to-LLM projection without padding."""

    def __init__(self, model: Any, torch_module: Any) -> None:
        setting = tuple(int(item) for item in model.speech_enc_llm_proj_setting)
        if not setting or setting[0] != SPEECH_PROJECTION_GROUP:
            raise ValueError(
                "audited checkpoint requires speech_enc_llm_proj_setting[0] == 4"
            )
        self.model = model
        self.torch = torch_module
        self.residual = None
        self.total_projected_tokens = 0

    def append(self, encoder_frames: Any) -> Any | None:
        normalized = self.model.ln_speech(encoder_frames)
        if self.residual is not None:
            normalized = self.torch.cat((self.residual, normalized), dim=1)
        group_count = int(normalized.shape[1]) // SPEECH_PROJECTION_GROUP
        used = group_count * SPEECH_PROJECTION_GROUP
        self.residual = normalized[:, used:].clone()
        if group_count == 0:
            return None

        batch, _, dimension = normalized.shape
        grouped = normalized[:, :used].reshape(
            batch,
            group_count,
            dimension * SPEECH_PROJECTION_GROUP,
        )
        projected = self.model.speech_enc_llm_proj(grouped)
        self.total_projected_tokens += group_count
        return projected


class AppendableCausalSpeechBridge:
    """End-to-end append-only *stable* Fbank to speech-embedding bridge."""

    def __init__(self, model: Any, torch_module: Any | None = None) -> None:
        if torch_module is None:
            import torch as torch_module  # type: ignore[no-redef]
        if str(getattr(model, "speech_encoder_type", "")) != "transformer_v2":
            raise ValueError("audited bridge supports transformer_v2 only")
        if not bool(getattr(model.speech_encoder.encoder, "is_causal", False)):
            raise ValueError("checkpoint speech encoder must be causal")

        self.model = model
        self.torch = torch_module
        self.all_fbank = None
        self.stable_subsampled_frames = 0
        self.transformer = CausalTransformerCacheBridge(
            model.speech_encoder,
            torch_module=torch_module,
        )
        self.projection = SpeechProjectionAccumulator(model, torch_module)
        self.total_appends = 0
        self.total_speech_tokens = 0

    def append_stable_fbank(
        self,
        fbank_chunk: Any,
    ) -> tuple[Any | None, dict[str, Any]]:
        if self.torch.is_grad_enabled():
            raise RuntimeError("causal speech append requires inference_mode/no_grad")
        if fbank_chunk.ndim != 3 or int(fbank_chunk.shape[0]) != 1:
            raise ValueError("fbank chunk must have shape [1, time, feature]")
        if int(fbank_chunk.shape[1]) <= 0:
            raise ValueError("fbank chunk must be non-empty")
        if self.all_fbank is None:
            self.all_fbank = fbank_chunk.clone()
        else:
            if int(fbank_chunk.shape[2]) != int(self.all_fbank.shape[2]):
                raise ValueError("fbank feature dimension changed")
            self.all_fbank = self.torch.cat((self.all_fbank, fbank_chunk), dim=1)

        total_fbank_frames = int(self.all_fbank.shape[1])
        plan = tail_recompute_plan(
            self.stable_subsampled_frames,
            total_fbank_frames,
        )
        projected = None
        autocast = getattr(self.model, "maybe_autocast", None)
        context = (
            autocast(getattr(self.model, "autocast_dtype", self.torch.bfloat16))
            if callable(autocast)
            else nullcontext()
        )
        with context:
            subsampled = _tail_subsample(
                self.model.speech_encoder.encoder_embed,
                self.all_fbank,
                plan,
                self.torch,
            )
            if subsampled is not None:
                encoded = self.transformer.forward_chunk(subsampled)
                projected = self.projection.append(encoded)

        self.stable_subsampled_frames = plan.stable_total_frames
        new_tokens = 0 if projected is None else int(projected.shape[1])
        self.total_speech_tokens += new_tokens
        self.total_appends += 1
        ledger = {
            "schema_version": BRIDGE_SCHEMA_VERSION,
            "append_index": self.total_appends - 1,
            "new_stable_fbank_frames": int(fbank_chunk.shape[1]),
            "total_stable_fbank_frames": total_fbank_frames,
            "new_stable_subsampled_frames": plan.new_stable_frames,
            "total_stable_subsampled_frames": plan.stable_total_frames,
            "tail_start_fbank_frame": plan.tail_start_fbank_frame,
            "tail_local_output_start": plan.local_output_start,
            "speech_encoder_cache_frames": self.transformer.position,
            "new_speech_tokens": new_tokens,
            "total_speech_tokens": self.total_speech_tokens,
            "projection_residual_encoder_frames": (
                0
                if self.projection.residual is None
                else int(self.projection.residual.shape[1])
            ),
            "reflected_right_boundary_fbank_frames_supplied": 0,
            "future_fbank_frames_supplied": 0,
        }
        return projected, ledger


class CanonicalStreamingFbank:
    """Expose only Fbank frames complete in the waveform revealed so far."""

    def __init__(self, fbank: Any, torch_module: Any | None = None) -> None:
        if torch_module is None:
            import torch as torch_module  # type: ignore[no-redef]
        config = fbank.config
        if int(config.sampling_rate) != SAMPLE_RATE:
            raise ValueError("audited Fbank requires 16 kHz")
        if abs(float(config.frame_length) - 0.025) > 1e-12:
            raise ValueError("audited Fbank requires a 25 ms frame")
        if abs(float(config.frame_shift) - 0.01) > 1e-12:
            raise ValueError("audited Fbank requires a 10 ms shift")
        if bool(config.snip_edges):
            raise ValueError("audited Fbank requires snip_edges=False")
        if float(config.dither) != 0.0:
            raise ValueError("streaming parity requires dither=0")
        self.fbank = fbank
        self.torch = torch_module
        self.context = None
        self.total_waveform_samples = 0
        self.total_stable_fbank_frames = 0
        self.append_count = 0

    def append_waveform(self, waveform_chunk: Any) -> tuple[Any, dict[str, Any]]:
        if waveform_chunk.ndim != 2 or int(waveform_chunk.shape[0]) != 1:
            raise ValueError("waveform chunk must have shape [1, samples]")
        sample_count = int(waveform_chunk.shape[1])
        if sample_count <= 0:
            raise ValueError("waveform chunk must be non-empty")
        device_chunk = waveform_chunk.to(self.fbank.device)
        features, self.context = self.fbank.extractor.online_inference(
            device_chunk,
            context=self.context,
        )
        self.total_waveform_samples += sample_count
        self.total_stable_fbank_frames += int(features.shape[1])
        expected = canonical_stable_fbank_frame_count(self.total_waveform_samples)
        if self.total_stable_fbank_frames != expected:
            raise RuntimeError(
                "canonical streaming Fbank count mismatch: "
                f"observed={self.total_stable_fbank_frames} expected={expected}"
            )
        ledger = {
            "schema_version": BRIDGE_SCHEMA_VERSION,
            "append_index": self.append_count,
            "new_waveform_samples": sample_count,
            "total_waveform_samples": self.total_waveform_samples,
            "new_stable_fbank_frames": int(features.shape[1]),
            "total_stable_fbank_frames": self.total_stable_fbank_frames,
            "withheld_right_boundary_frames": 1,
            "right_boundary_reflection_used": False,
        }
        self.append_count += 1
        return features, ledger


class SpeechEmbeddingQueue:
    """FIFO used by the unmodified model's ``prepare_inputs_embeds`` path."""

    def __init__(self, torch_module: Any) -> None:
        self.torch = torch_module
        self.pending = None
        self.archive = None
        self.total_appended = 0
        self.total_consumed = 0

    @property
    def available(self) -> int:
        return 0 if self.pending is None else int(self.pending.shape[0])

    def append(self, projected: Any | None) -> None:
        if projected is None:
            return
        if projected.ndim != 3 or int(projected.shape[0]) != 1:
            raise ValueError(
                "projected speech embeddings must have shape [1, time, dim]"
            )
        value = projected[0]
        self.archive = (
            value.clone()
            if self.archive is None
            else self.torch.cat((self.archive, value), dim=0)
        )
        self.pending = (
            value.clone()
            if self.pending is None
            else self.torch.cat((self.pending, value), dim=0)
        )
        self.total_appended += int(value.shape[0])

    def take(self, n_speech_tokens: Any) -> Any:
        counts = [int(item) for item in n_speech_tokens]
        if len(counts) != 1:
            raise ValueError("queued speech bridge supports batch size one only")
        count = counts[0]
        if count <= 0:
            raise ValueError("queued encode_speech requires a positive token count")
        if self.available < count:
            raise RuntimeError(
                f"speech embedding underflow: available={self.available} requested={count}"
            )
        result = self.pending[:count]
        remainder = self.pending[count:]
        self.pending = None if int(remainder.shape[0]) == 0 else remainder
        self.total_consumed += count
        return result


class QueuedEncodeSpeechOverride:
    """Temporarily route audio-pad replacement through a workspace FIFO."""

    def __init__(self, model: Any, queue: SpeechEmbeddingQueue) -> None:
        self.model = model
        self.queue = queue
        self.original = None

    def __enter__(self) -> "QueuedEncodeSpeechOverride":
        if self.original is not None:
            raise RuntimeError("queued encode_speech override is already installed")
        self.original = self.model.encode_speech
        queue = self.queue

        def queued_encode_speech(
            unused_model: Any,
            unused_env_fbank_feature: Any,
            unused_env_fbank_feature_len: Any,
            unused_assist_fbank_feature: Any,
            unused_assist_fbank_feature_len: Any,
            n_speech_tokens: Any,
        ) -> Any:
            return queue.take(n_speech_tokens)

        self.model.encode_speech = MethodType(queued_encode_speech, self.model)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.original is not None:
            self.model.encode_speech = self.original
            self.original = None
