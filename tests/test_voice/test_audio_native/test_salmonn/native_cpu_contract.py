"""Run the worker contract with a tiny external model and real causal frontend.

Executed in SALMONN's Python environment. Only external model-loading, tokenizer,
CosyVoice and device APIs are substituted; the worker, acoustic stream, projection,
KV stepping, native control cadence and playback queue are production code.
"""

import base64
import json
import os
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

torch.set_num_threads(1)
os.environ.setdefault("USE_TF", "0")

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts" / "open_model_workers"))
RUNTIME = json.loads(Path(sys.argv[1]).read_text())
PACKAGE = Path(RUNTIME["assets"]["package_root"])
TRAINING = json.loads((Path(RUNTIME["checkpoint"]).parent / "config.json").read_text())
HAS_SPEECH_END = bool(TRAINING["model_args"].get("add_end_of_speech", False))
sys.path.insert(0, str(PACKAGE))
from models.streaming_transformer_encoder_v2.audio_encoder import (  # noqa: E402
    AttributeDict,
    CausalAudioEncoder,
)
from salmonn_worker import SalmonnWorker  # noqa: E402


class Tokenizer:
    audio_pad_token = "<audio>"
    audio_pad_token_id = 1
    think_token = "<think>"
    end_of_speech_token = "<|end_of_speech|>"
    eos_token_id = 4
    vocabulary = {
        "<audio>": 1,
        "<think>": 2,
        "<|start_header_id|>": 3,
        "<|eot_id|>": 4,
        "<|end_of_speech|>": 6,
    }

    def __call__(self, value, **kwargs):
        words = re.findall(r"<audio>|<think>|<\|[^>]+\|>|[^<]+", value)
        ids = [self.vocabulary.get(word, 5) for word in words]
        return {
            "input_ids": torch.tensor([ids]) if kwargs.get("return_tensors") else ids
        }

    def convert_tokens_to_ids(self, value):
        return self.vocabulary[value]

    def get_vocab(self):
        return self.vocabulary

    def decode(self, ids, **kwargs):
        return "".join("hi " for value in ids if value == 5)


class Cosy:
    def __init__(self):
        self.flow = SimpleNamespace(pre_lookahead_len=3, input_frame_rate=25)
        self.source_cache_len = 0
        self.llm = SimpleNamespace(text_chunk=1, inference_salmonn_stream=self.generate)

    def generate(self, projected, lengths, **kwargs):
        return None, list(range(10)), list(range(10)), True

    def tts_salmonn(self, codecs, speaker, **kwargs):
        return torch.full((1, 2400), 0.25), None, None


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = AttributeDict(
            causal=True,
            encoder_dim=32,
            num_layers=2,
            use_flash_attention=0,
            attention_dropout=0.0,
            num_heads=4,
            subsampling_factor=2,
            feature_dim=128,
        )
        self.speech_encoder = CausalAudioEncoder(config, num_mels=128)
        self.speech_encoder_type = "transformer_v2"
        self.encode_chunk_size = TRAINING["model_args"].get("encode_chunk_size", -1)
        self.speech_enc_llm_proj_setting = ("4", "32")
        self.ln_speech = torch.nn.LayerNorm(32)
        self.speech_enc_llm_proj = torch.nn.Linear(128, 24)
        self.config = SimpleNamespace(max_position_embeddings=8192)
        self.embed_layer_idx = 0
        self.cosy = Cosy()
        self.llm_syn_proj = torch.nn.Identity()
        self.calls = 0
        self.protocol_state = "initial"
        self.completed_turns = 0
        self.answer_remaining = 0

    def to(self, *args, **kwargs):
        return self

    def reset_speech_cache(self):
        pass

    def encode_speech(self, *args):
        raise AssertionError("worker must provide causal embeddings")

    def forward(self, input_ids, n_speech_tokens, **kwargs):
        if self.calls == 0:
            # Upstream evaluate_hf.py default/llama3 prefix: bare context,
            # then audio, then the user header and eot. No system wrapper.
            assert input_ids[0].tolist() == [5, 5, 1, 3, 5, 5, 5, 4], (
                "SALMONN initial decoder prefix must match the trained upstream template",
                input_ids[0].tolist(),
            )
        if n_speech_tokens[0]:
            self.encode_speech(None, None, None, None, n_speech_tokens)
        if HAS_SPEECH_END:
            # Real AES checkpoints do not need to predict end-of-speech. The
            # reference playback controller appends it after EOT + drained TTS.
            incoming = input_ids[0].tolist()
            if 6 in incoming:
                assert incoming == [1, 6, 2], (
                    "one current audio block must precede controller EOS + think",
                    incoming,
                )
                self.completed_turns += 1
                self.protocol_state = "await_user_header"
                token = 5  # must be ignored by the controller handshake
            elif self.protocol_state == "await_user_header":
                assert incoming == [1, 3, 5, 5, 5, 4], incoming
                token = 3 if self.completed_turns < 2 else 2
                self.protocol_state = "first_answer" if token == 3 else "done"
            elif self.protocol_state == "initial":
                token = 2 if self.calls == 0 else 3
                if token == 3:
                    self.protocol_state = "first_answer"
            elif self.protocol_state == "first_answer":
                token = 5
                self.answer_remaining = 8 if self.completed_turns == 0 else 4
                self.protocol_state = "answering"
            elif self.protocol_state == "answering":
                token = 5 if self.answer_remaining else 4
                if self.answer_remaining:
                    self.answer_remaining -= 1
                else:
                    self.protocol_state = "waiting_for_playback"
            else:
                token = 2
        else:
            tokens = [2, 3, *([5] * 9), 4, 3]
            token = tokens[self.calls] if self.calls < len(tokens) else 2
        self.calls += 1
        logits = torch.zeros((1, input_ids.shape[1], 8))
        logits[..., token] = 1
        return SimpleNamespace(
            logits=logits,
            past_key_values=(self.calls,),
            hidden_states=[torch.ones((1, input_ids.shape[1], 24))],
        )


class Frontend:
    def _extract_speech_feat(self, audio):
        return torch.zeros((1, 5, 80)), torch.tensor([5])

    def _extract_spk_embedding(self, audio):
        return torch.zeros((1, 192))

    def _extract_speech_token(self, audio):
        return torch.ones((1, 5), dtype=torch.long), torch.tensor([5])


utils = ModuleType("utils")
utils.__file__ = str(PACKAGE / "utils.py")
utils.load_model = lambda *args: (Tokenizer(), TinyModel())
cosy = ModuleType("cosyvoice.cli.cosyvoice")
cosy.CosyVoice2 = lambda *args, **kwargs: SimpleNamespace(frontend=Frontend())
real_device = torch.device
with (
    patch.dict(sys.modules, {"utils": utils, "cosyvoice.cli.cosyvoice": cosy}),
    patch("torch.cuda.is_available", return_value=True),
    patch("torch.device", side_effect=lambda *args, **kwargs: real_device("cpu")),
    patch("torchaudio.load", return_value=(torch.zeros((1, 2400)), 24000)),
):
    worker = SalmonnWorker()
    for episode in range(2):
        info = worker.connect(RUNTIME, "Keep talking naturally.")
        assert (
            info["checkpoint_variant"]
            == Path(RUNTIME["checkpoint"]).resolve().parent.name
        )
        try:
            worker.step(bytes(2558))
        except ValueError:
            pass
        else:
            raise AssertionError("accepted a short tick")
        received = []
        append = worker.session.fbank.append_waveform

        def capture(waveform):
            received.append(waveform.clone())
            return append(waveform)

        worker.session.fbank.append_waveform = capture
        episode_ticks = 1510 if "--long" in sys.argv else (96 if HAS_SPEECH_END else 16)
        outputs = [worker.step(bytes(2560)) for _ in range(episode_ticks)]
        for index in range(1, len(outputs)):
            previous = base64.b64decode(outputs[index - 1]["audio_pcm16_b64"])
            expected = (
                (
                    torch.frombuffer(bytearray(previous), dtype=torch.int16).float()
                    / 32768.0
                )
                if previous
                else torch.zeros(1280)
            )
            if HAS_SPEECH_END:
                expected = torch.zeros(1280)  # new checkpoint was trained with er0
            assert torch.allclose(received[index].squeeze(), expected), (
                "SALMONN must hear the previous actually played block, not silence "
                "or queued future output"
            )
        assert outputs[0]["audio_pcm16_b64"] == ""
        assert outputs[0]["events"][-1]["event"] == "frontend_startup"
        if "--long" in sys.argv:
            chunk_frames = round(TRAINING["model_args"]["encode_chunk_size"] * 100)
            expected_tokens = []
            for n in range(1, episode_ticks + 1):
                full, remainder = divmod(8 * n - 1, chunk_frames)
                encoded_frames = full * ((chunk_frames - 1) // 2)
                encoded_frames += max(0, (remainder - 2) // 2)
                expected_tokens.append(encoded_frames // 4)
            assert [
                item["events"][0]["speech_tokens_total"] for item in outputs
            ] == expected_tokens
            waits = [
                e
                for item in outputs
                for e in item["events"]
                if e["event"] == "frontend_wait"
            ]
            assert len(waits) == 1, waits
            assert worker.session.queue.total_consumed == expected_tokens[-1]
            print(
                "long contract: one causal frontend wait, all input packets preserved",
                flush=True,
            )
        else:
            assert [
                item["events"][0]["speech_tokens_total"] for item in outputs
            ] == list(range(episode_ticks))
        assert [item["events"][0]["input_samples_total"] for item in outputs] == [
            1280 * n for n in range(1, episode_ticks + 1)
        ]
        audible = [
            base64.b64decode(item["audio_pcm16_b64"])
            for item in outputs
            if item["audio_pcm16_b64"]
        ]
        assert audible and all(len(chunk) == 2560 for chunk in audible)
        transitions = [
            e
            for item in outputs
            for e in item["events"]
            if e["event"] == "native_listening_transition"
        ]
        if HAS_SPEECH_END:
            speech_ends = [
                e
                for item in outputs
                for e in item["events"]
                if e["event"] == "playback_turn_ended"
            ]
            starts = [
                e
                for item in outputs
                for e in item["events"]
                if e["event"] == "native_assistant_start"
            ]
            assert len(speech_ends) == 2
            assert all(e["source"] == "playback_controller" for e in speech_ends)
            assert all(e["playback"]["text_closed"] for e in speech_ends)
            assert len(starts) == 2
            assert worker.session.model.completed_turns == 2
            assert not any(
                e["event"] == "native_speech_end"
                for item in outputs
                for e in item["events"]
            )
        else:
            assert len(transitions) == 1
            assert transitions[0]["playback"]["interrupted"]
            assert transitions[0]["playback"]["cancelled_samples_16k"] > 0
            assert all(not item["audio_pcm16_b64"] for item in outputs[-3:])
        worker.disconnect()
        worker.disconnect()
print(
    f"SALMONN CPU worker contract passed: 2 fresh episodes, {2 * episode_ticks} causal ticks"
)
