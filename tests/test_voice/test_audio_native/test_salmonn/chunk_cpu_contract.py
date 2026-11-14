"""Compare streaming chunk boundaries with the checkpoint's full encoder oracle."""

import os
import sys

os.environ.setdefault("USE_TF", "0")
from pathlib import Path

import torch

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/open_model_workers"))
sys.path.insert(0, "/mnt/bn/audio-visual-llm-data6/yuwenyi/playground/salmonn-omni-dev")
from models.streaming_transformer_encoder_v2.audio_encoder import (  # noqa: E402
    AttributeDict,
    CausalAudioEncoder,
)

if "--unbounded" in sys.argv:
    from salmonn_causal import AppendableCausalSpeechBridge as ChunkedCausalSpeechBridge
else:
    from salmonn_causal import ChunkedCausalSpeechBridge


class Model(torch.nn.Module):
    def __init__(self, seconds):
        super().__init__()
        self.encode_chunk_size = seconds
        self.speech_encoder_type = "transformer_v2"
        self.speech_enc_llm_proj_setting = ("4", "32")
        self.speech_encoder = CausalAudioEncoder(
            AttributeDict(
                causal=True,
                encoder_dim=32,
                num_layers=2,
                use_flash_attention=0,
                attention_dropout=0.0,
                num_heads=4,
                subsampling_factor=2,
                feature_dim=128,
            ),
            num_mels=128,
        )
        self.ln_speech = torch.nn.LayerNorm(32)
        self.speech_enc_llm_proj = torch.nn.Linear(128, 24)


with torch.inference_mode():
    for frames_per_chunk in [24, 3000]:
        torch.manual_seed(42)
        model = Model(frames_per_chunk / 100).eval()
        x = torch.randn(1, frames_per_chunk * 2 + 3, 128)

        # The installed eager attention requires a 4D additive mask. Native
        # forward supplies a 2D valid-frame mask intended for FlashAttention.
        # Adapt that external API boundary while running the full native encoder.
        def eager_mask(module, args, kwargs):
            hidden = args[0]
            n = hidden.shape[1]
            mask = hidden.new_zeros((1, 1, n, n))
            mask.masked_fill_(
                torch.ones(n, n, dtype=torch.bool).triu(1),
                torch.finfo(hidden.dtype).min,
            )
            kwargs["attention_mask"] = mask
            return args, kwargs

        handles = [
            layer.register_forward_pre_hook(eager_mask, with_kwargs=True)
            for layer in model.speech_encoder.encoder.layers
        ]
        encoded_chunks = []
        # Native code concatenates independently encoded chunks BEFORE projection.
        for start in [0, frames_per_chunk]:
            encoded, _ = model.speech_encoder.forward_encoder(
                x[:, start : start + frames_per_chunk], torch.tensor([frames_per_chunk])
            )
            encoded_chunks.append(encoded)
        normalized = model.ln_speech(torch.cat(encoded_chunks, dim=1))
        complete = (normalized.shape[1] // 4) * 4
        expected = model.speech_enc_llm_proj(
            normalized[:, :complete].reshape(1, -1, 128)
        )
        # Incremental production bridge supplies its own correctly sized mask.
        for handle in handles:
            handle.remove()
        bridge = ChunkedCausalSpeechBridge(model, torch)
        actual = []
        for start in range(0, x.shape[1], 8):
            projected, ledger = bridge.append_stable_fbank(x[:, start : start + 8])
            if projected is not None:
                actual.append(projected)
        actual = torch.cat(actual, dim=1)
        assert actual.shape == expected.shape, (actual.shape, expected.shape)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        assert ledger["completed_encoder_chunks"] == 2
        print(
            "PASS native encoder chunk",
            frames_per_chunk,
            "frames; tokens",
            actual.shape[1],
            flush=True,
        )
