"""Native persistent SALMONN execution; imported only by its model interpreter.

See SALMONN_PROVENANCE.json for source adaptation and exact runtime files.
"""

from __future__ import annotations

import base64
import gc
import hashlib
import importlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace


class Args(SimpleNamespace):
    """Attribute and mapping-style arguments expected by native model loading."""

    def get(self, key, default=None):
        return getattr(self, key, default)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class NativeSession:
    """One model session with append-only acoustic and language-model caches."""

    def __init__(self, config, system_prompt):
        import numpy as np
        import torch
        import torchaudio
        from salmonn_causal import (
            AppendableCausalSpeechBridge,
            CanonicalStreamingFbank,
            ChunkedCausalSpeechBridge,
            QueuedEncodeSpeechOverride,
            SpeechEmbeddingQueue,
        )
        from salmonn_playback import CosyPromptRuntime, FDBSpeechPlayback

        self.torch, self.np = torch, np
        self.override = None
        self.model = None
        self.playback = None
        assets = config["assets"]
        settings = config.get("settings", {})
        package = Path(assets["package_root"]).resolve()
        provenance = json.loads(
            Path(__file__).with_name("SALMONN_PROVENANCE.json").read_text()
        )
        expected = provenance["external_source_sha256"]
        observed = {name: _hash(package / name) for name in expected}
        if observed != expected:
            changed = [name for name in expected if expected[name] != observed[name]]
            raise RuntimeError(f"SALMONN runtime source changed: {changed}")
        for root in reversed(
            [package, assets["audio_visual_models_root"], assets["matcha_tts_root"]]
        ):
            if not Path(root).is_dir():
                raise FileNotFoundError(root)
            sys.path.insert(0, str(root))
        utils = importlib.import_module("utils")
        if Path(utils.__file__).resolve().parent != package:
            raise RuntimeError("SALMONN utils import came from another package")
        from cosyvoice.cli.cosyvoice import CosyVoice2
        from lhotse import Fbank, FbankConfig

        training_path = Path(config["checkpoint"]).resolve().parent / "config.json"
        training = json.loads(training_path.read_text())
        model_values, data_values = training["model_args"], training["data_args"]
        expected_config = {
            "backbone": "llama3",
            "speech_encoder_type": "transformer_v2",
            "speech_synthesizer_type": "cosy",
            "simple_mode": True,
            "add_assist_stream": True,
        }
        for key, value in expected_config.items():
            if model_values.get(key) != value:
                raise ValueError(f"Unsupported SALMONN training setting {key}")
        for key, value in {
            "template": "default",
            "delta": 0.08,
            "num_audio_embs": 1,
            "num_text_tokens": 1,
        }.items():
            if data_values.get(key) != value:
                raise ValueError(f"Unsupported SALMONN training setting {key}")
        if settings.get("device", "cuda") != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("SALMONN native session requires a CUDA worker")
        if settings.get("dtype", "bfloat16") != "bfloat16":
            raise ValueError("SALMONN requires bfloat16 model weights")
        if settings.get("greedy", True) is not True:
            raise ValueError("SALMONN MVP uses greedy native decoding")
        seed = int(settings.get("seed", 42))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = bool(
            settings.get("matmul_allow_tf32", False)
        )
        torch.backends.cudnn.allow_tf32 = bool(settings.get("cudnn_allow_tf32", True))
        torch.set_float32_matmul_precision(
            settings.get("float32_matmul_precision", "highest")
        )
        model_values.update(
            ckpt_path=str(Path(config["checkpoint"]).resolve()),
            lazy_encode=True,
            speech_synthesizer_type="cosy",
            autocast_dtype=settings.get("speech_encoder_autocast_dtype", "bfloat16"),
        )
        self.device = torch.device("cuda")
        self.tokenizer, model = utils.load_model(
            Args(**model_values), Args(**data_values)
        )
        self.model = model.to(self.device).to(torch.bfloat16).eval()
        self.model.lazy_encode = True
        self.model.incremental_audio_cache = True
        self.model.reset_speech_cache()
        fbank = Fbank(FbankConfig(num_mel_bins=128))
        self.fbank = CanonicalStreamingFbank(fbank, torch)
        self.speech = (
            ChunkedCausalSpeechBridge(model, torch)
            if float(getattr(model, "encode_chunk_size", -1)) > 0
            else AppendableCausalSpeechBridge(model, torch)
        )
        self.queue = SpeechEmbeddingQueue(torch)
        self.override = QueuedEncodeSpeechOverride(model, self.queue)
        self.override.__enter__()
        with torch.inference_mode():
            frontend = CosyVoice2(
                assets["cosyvoice_checkpoint"], load_jit=False, load_trt=False
            ).frontend
            prompt, rate = torchaudio.load(assets["speaker_prompt_audio"])
            prompt = prompt.float().mean(dim=0, keepdim=True)
            if rate != 24000:
                prompt = torchaudio.functional.resample(prompt, rate, 24000)
            feature, _ = frontend._extract_speech_feat(prompt.to(self.device))
            prompt16 = torchaudio.functional.resample(prompt, 24000, 16000).to(
                self.device
            )
            speaker = frontend._extract_spk_embedding(prompt16)
            codec, _ = frontend._extract_speech_token(prompt16)
            runtime = CosyPromptRuntime(
                prompt_speech_feat=feature,
                prompt_speech_token=codec,
                spk_embedding=speaker,
                resample_24k_to_16k=torchaudio.transforms.Resample(24000, 16000),
            )
        self.playback = FDBSpeechPlayback(
            model,
            self.tokenizer,
            self.device,
            Args(**data_values),
            runtime=runtime,
            clause_v2=False,
        )
        self.input_ids = self._tokens(
            "<|begin_of_text|>"
            + system_prompt
            + self.tokenizer.audio_pad_token
            + "<|start_header_id|>user<|end_header_id|>\n\n<|eot_id|>"
        )
        self.think = self._tokens(self.tokenizer.think_token)
        self.start_id = int(self.tokenizer.convert_tokens_to_ids("<|start_header_id|>"))
        self.eot_id = int(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        self.speech_end_id = (
            int(self.tokenizer.convert_tokens_to_ids("<|end_of_speech|>"))
            if model_values.get("add_end_of_speech", False)
            else None
        )
        self.cached_length = 0
        self.past = None
        self.tick = 0
        self.echo_ratio = float(
            settings.get("echo_ratio", data_values.get("simple_mode_echo_ratio", 1.0))
        )
        if not 0 <= self.echo_ratio <= 1:
            raise ValueError("SALMONN echo_ratio must be between zero and one")
        self.played_audio = np.zeros(1280, dtype=np.float32)
        self.speaking = False
        self.end_text = False
        self._text_eot_tick = None
        self._playback_end_pending = False
        self._decoded_this_tick = False
        self.output_ids = []
        self.previous_text = ""
        self.dummy_feature = torch.zeros((1, 1, 128), device=self.device)
        self.dummy_len = torch.ones((1,), device=self.device, dtype=torch.long)
        self.provenance = {
            "checkpoint": str(Path(config["checkpoint"]).resolve()),
            "checkpoint_variant": Path(config["checkpoint"]).resolve().parent.name,
            "training_config_sha256": _hash(training_path),
            "runtime_source_sha256": observed,
            "input_sample_rate": 16000,
            "output_sample_rate": 16000,
            "tick_ms": 80,
            "frontend_startup_ticks": 1,
            "playback_lead_seconds": self.playback.lead_seconds,
            "echo_ratio": self.echo_ratio,
            "self_audio_source": "previous actually played 80ms block",
            "greedy": True,
            "seed": seed,
            "speech_end_token_id": self.speech_end_id,
            "speech_end_control": "text_eot_then_playback_drain_then_input_marker",
            "encode_chunk_seconds": model_values.get("encode_chunk_size", -1),
        }

    def _tokens(self, text):
        return self.tokenizer(
            text,
            padding=False,
            return_token_type_ids=False,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0].to(self.device)

    def _append(self, *values):
        self.input_ids = self.torch.cat((self.input_ids, *values))

    def _decode(self):
        torch = self.torch
        total = int(self.input_ids.shape[0])
        limit = int(getattr(self.model.config, "max_position_embeddings", 8192))
        if total >= limit:
            raise RuntimeError("SALMONN context window exhausted; start a new session")
        suffix = self.input_ids[self.cached_length :]
        if not suffix.numel():
            raise RuntimeError("SALMONN decoder has no unconsumed prefix")
        count = int((suffix == self.tokenizer.audio_pad_token_id).sum().item())
        if count > self.queue.available:
            raise RuntimeError("SALMONN decoder requested audio not yet received")
        result = self.model(
            input_ids=suffix.unsqueeze(0),
            attention_mask=torch.ones((1, total), dtype=torch.long, device=self.device),
            past_key_values=self.past,
            use_cache=True,
            cache_position=torch.arange(self.cached_length, total, device=self.device),
            env_fbank_feature=self.dummy_feature,
            env_fbank_feature_len=self.dummy_len,
            assist_fbank_feature=None,
            assist_fbank_feature_len=None,
            n_speech_tokens=[count],
            output_hidden_states=True,
            return_dict=True,
        )
        self.past = result.past_key_values
        if self.past is None:
            raise RuntimeError("SALMONN runtime returned no decoder KV cache")
        self.cached_length = total
        token = result.logits[:, -1, :].argmax(dim=-1)
        hidden = result.hidden_states[self.model.embed_layer_idx][..., -1:, :]
        return int(token.item()), token, hidden

    def _next_audio(self, tail):
        self._append(self._tokens(self.tokenizer.audio_pad_token), tail)

    def _accept_text(self, token_id, hidden):
        self.output_ids.append(token_id)
        self.playback.append_token(token_id, hidden)
        if token_id == self.eot_id:
            self.end_text = True
            self._text_eot_tick = self.tick

    def step(self, pcm16):
        """Append one received frame and advance native generation once."""
        if len(pcm16) != 2560:
            raise ValueError("SALMONN step requires 1280 PCM16 samples (80 ms)")
        torch, np = self.torch, self.np
        self._decoded_this_tick = False
        audio = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        # The incoming user block and previous playback share the same clock.
        # Never feed synthesized-but-unplayed future audio back to the model.
        audio = audio + self.played_audio * self.echo_ratio
        with torch.inference_mode():
            features, fbank_event = self.fbank.append_waveform(
                torch.from_numpy(audio).unsqueeze(0)
            )
            projected, speech_event = self.speech.append_stable_fbank(
                features.to(self.device)
            )
            self.queue.append(projected)
            events = [
                {
                    "event": "audio_ingest",
                    "tick": self.tick,
                    "input_samples_total": self.fbank.total_waveform_samples,
                    "self_audio_nonzero_samples": int(
                        np.count_nonzero(self.played_audio)
                    ),
                    "echo_ratio": self.echo_ratio,
                    "speech_tokens_total": self.queue.total_appended,
                    "fbank": fbank_event,
                    "speech": speech_event,
                }
            ]
            if self.tick == 0:
                if self.queue.available != 0:
                    raise RuntimeError(
                        "SALMONN causal frontend startup contract changed"
                    )
                self.tick += 1
                events.append({"event": "frontend_startup", "pending_audio_tokens": 1})
                return {
                    "audio_pcm16_b64": "",
                    "sample_rate": 16000,
                    "text": "",
                    "events": events,
                }
            suffix = self.input_ids[self.cached_length :]
            required = int((suffix == self.tokenizer.audio_pad_token_id).sum().item())
            if required > self.queue.available:
                events.append(
                    {
                        "event": "frontend_wait",
                        "pending_audio_tokens": required,
                        "available_audio_tokens": self.queue.available,
                    }
                )
                return self._finish_tick("", events)
            token_id, token, hidden = self._decode()
            self._decoded_this_tick = True
            events.append(
                {
                    "event": "native_token",
                    "token_id": token_id,
                    "speech_tokens_consumed": self.queue.total_consumed,
                    "speech_tokens_pending": self.queue.available,
                    "decoder_cached_tokens": self.cached_length,
                }
            )
            text = ""
            if self._playback_end_pending:
                # Match upstream's end_of_speech state: this prediction only
                # consumes the controller marker. It is not a new response.
                self._playback_end_pending = False
                self._text_eot_tick = None
                self._next_audio(
                    self._tokens(
                        "<|start_header_id|>user<|end_header_id|>\n\n<|eot_id|>"
                    )
                )
                events.append(
                    {
                        "event": "listening_after_playback",
                        "source": "playback_controller",
                        "ignored_native_prediction": token_id,
                    }
                )
            elif token_id == self.start_id:
                if not self.speaking:
                    self.speaking = True
                    self.end_text = False
                    self._text_eot_tick = None
                    self.output_ids = []
                    self.previous_text = ""
                    if self.playback.active and not self.playback.drained:
                        self.playback.interrupt()
                        events.append(
                            {
                                "event": "native_new_turn_replaced_playback",
                                "playback": self.playback.metadata,
                            }
                        )
                    self.playback.begin_turn()
                    self._append(
                        self._tokens(
                            "<|start_header_id|>assistant<|end_header_id|>\n\n"
                        )
                    )
                    token_id, token, hidden = self._decode()
                    events.append(
                        {"event": "native_assistant_start", "token_id": token_id}
                    )
                    if token_id != self.start_id:
                        self._accept_text(token_id, hidden)
                if token_id == self.start_id:
                    # This transition is predicted by the model, never triggered
                    # by a scripted user onset. Unheard output is cancelled.
                    self.playback.interrupt()
                    events.append(
                        {
                            "event": "native_listening_transition",
                            "playback": self.playback.metadata,
                        }
                    )
                    self.speaking = False
                    self.end_text = False
                    self._text_eot_tick = None
                    self._next_audio(
                        self._tokens(
                            "<|start_header_id|>user<|end_header_id|>\n\n<|eot_id|>"
                        )
                    )
                else:
                    self._next_audio(token)
            elif self.speaking:
                if not self.end_text:
                    self._accept_text(token_id, hidden)
                else:
                    token = self.think
                self._next_audio(token)
            else:
                self._next_audio(self.think)
            if self.speaking:
                current = self.tokenizer.decode(
                    self.output_ids, skip_special_tokens=True
                )
                text = (
                    current[len(self.previous_text) :]
                    if current.startswith(self.previous_text)
                    else current
                )
                self.previous_text = current
            return self._finish_tick(text, events)

    def _finish_tick(self, text, events):
        """Advance playback even when the causal frontend needs more audio."""
        torch, np = self.torch, self.np
        output = self.playback.step()
        encoded = (
            output.squeeze(0)
            .clamp(-1, 1)
            .mul(32767)
            .round()
            .to(torch.int16)
            .numpy()
            .astype("<i2")
            .tobytes()
        )
        self.played_audio = (
            np.frombuffer(encoded, dtype="<i2").astype(np.float32) / 32768.0
        )
        if (
            self.speech_end_id is not None
            and self._decoded_this_tick
            and self.speaking
            and self.end_text
            and self._text_eot_tick is not None
            and self.tick > self._text_eot_tick
            and self.playback.drained
        ):
            # Upstream fdb_streaming.py inserts EOS only after text EOT has been
            # consumed and the last real playback samples have been delivered.
            # Keep the single pending audio pad; replace its THINK tail with
            # EOS + THINK. Waiting for the model to predict EOS can loop forever.
            if not torch.equal(self.input_ids[-self.think.numel() :], self.think):
                raise RuntimeError("SALMONN playback end has no pending THINK tail")
            self.input_ids = torch.cat(
                (
                    self.input_ids[: -self.think.numel()],
                    self._tokens("<|end_of_speech|>"),
                    self.think,
                )
            )
            self._playback_end_pending = True
            self.speaking = False
            self.end_text = False
            events.append(
                {
                    "event": "playback_turn_ended",
                    "source": "playback_controller",
                    "reason": "text_eot_and_playback_drained",
                    "text_eot_tick": self._text_eot_tick,
                    "appended_token_id": self.speech_end_id,
                    "playback": self.playback.metadata,
                }
            )
        events.append({"event": "playback", **self.playback.metadata})
        self.tick += 1
        return {
            "audio_pcm16_b64": base64.b64encode(encoded).decode("ascii")
            if any(encoded)
            else "",
            "sample_rate": 16000,
            "text": text,
            "events": events,
        }

    def close(self):
        """Restore instance overrides and release all session caches."""
        if self.override is not None:
            self.override.__exit__(None, None, None)
            self.override = None
        if self.playback is not None:
            self.playback.interrupt()
        if self.model is not None:
            self.model.reset_speech_cache()
        self.__dict__.clear()
        gc.collect()
        import torch

        torch.cuda.empty_cache()
