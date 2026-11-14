"""Exercise cumulative minimum-length handling through public native playback."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[4] / "scripts/open_model_workers")
)
from salmonn_playback import CosyPromptRuntime, FDBSpeechPlayback


class Tokenizer:
    think_token = "think"

    def __call__(self, text, **kwargs):
        return {"input_ids": [{"think": 1, "<|eot_id|>": 2}.get(text, 3)]}

    def get_vocab(self):
        return {}


def episode(initial_tokens):
    calls = []

    class Cosy:
        source_cache_len = 0
        flow = SimpleNamespace(pre_lookahead_len=0, input_frame_rate=25)

        def __init__(self):
            self.llm = SimpleNamespace(
                text_chunk=4, inference_salmonn_stream=self.generate
            )

        def generate(
            self,
            text,
            length,
            *,
            text_idx_num,
            token_input,
            generated=None,
            min_token_text_ratio=2,
        ):
            calls.append((text_idx_num, min_token_text_ratio))
            if token_input is None:
                assert min_token_text_ratio == 2
                return torch.zeros(1, 17, 8), list(range(initial_tokens)), [], False
            if initial_tokens >= 8:
                # Real capture: four text tokens, 12 codecs already generated;
                # a fresh local minimum of eight rejects valid EOS forever.
                if int(length.item() * min_token_text_ratio) > 0:
                    raise RuntimeError(
                        "sampling reaches max_trials 50 and still get eos when ignore_eos is True"
                    )
                return token_input, generated, [], True
            assert min_token_text_ratio > 0, (
                "Minimum disabled before enough speech existed"
            )
            return token_input, list(range(8)), [], True

        def tts_salmonn(self, codecs, speaker, **kwargs):
            return torch.ones(1, 1280), None, None

    model = SimpleNamespace(cosy=Cosy(), llm_syn_proj=lambda x: x)
    runtime = CosyPromptRuntime(
        prompt_speech_feat=None,
        prompt_speech_token=None,
        spk_embedding=None,
        resample_24k_to_16k=lambda x: x,
    )
    playback = FDBSpeechPlayback(
        model,
        Tokenizer(),
        "cpu",
        SimpleNamespace(delta=0.08, num_text_tokens=1),
        runtime=runtime,
        clause_v2=False,
    )
    playback.begin_turn()
    for token in [5, 5, 5, 2]:
        playback.append_token(token, torch.zeros(1, 1, 8))
    for _ in range(20):
        playback.step()
        if playback.drained:
            break
    assert playback.drained
    assert (
        playback.metadata["rendered_samples_16k"]
        == playback.metadata["played_samples_16k"]
    )
    assert playback.metadata["cancelled_samples_16k"] == 0
    return calls


if __name__ == "__main__":
    episode(12)
    episode(3)
    print("cumulative EOS minimum and normal waveform drain passed")
