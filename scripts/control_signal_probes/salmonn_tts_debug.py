"""Capture and replay the exact failing CosyVoice call without changing weights."""

import argparse
import json
from pathlib import Path

from open_equivalence_worker import Runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--replay-only", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = args.root
    source = root.parent / "20261113-open-equivalence-50"
    config = json.loads((source / "configs/salmonn.json").read_text())
    prompt = (source / "salmonn/0-direct/prompt.txt").read_text()
    runtime = Runtime("salmonn", "direct", config, prompt)
    if args.verify:
        import base64
        import wave

        data = (root / "input.pcm16").read_bytes()
        with (
            (root / "verified-events.jsonl").open("w") as events,
            wave.open(str(root / "verified.wav"), "wb") as audio,
        ):
            audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            for offset in range(0, len(data), 2560):
                event = runtime.step(data[offset : offset + 2560])
                audio.writeframes(base64.b64decode(event.pop("audio_pcm16_b64")))
                events.write(json.dumps(event) + "\n")
        result = dict(
            ticks=len(data) // 2560,
            provenance=runtime.provenance,
            playback=runtime.native.playback.metadata,
        )
        (root / "verified.json").write_text(json.dumps(result, indent=2))
        print("VERIFIED", json.dumps(result), flush=True)
        runtime.close()
        return
    import torch

    llm = runtime.native.model.cosy.llm
    original = llm.inference_salmonn_stream
    tick = 0

    def traced(*a, **kw):
        initial_kw = dict(kw)
        if "generated" in initial_kw:
            initial_kw["generated"] = list(initial_kw["generated"])
        rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state()
        try:
            return original(*a, **kw)
        except RuntimeError as exc:
            if "max_trials" not in str(exc):
                raise
            torch.save(
                dict(args=a, kwargs=initial_kw, rng=rng, cuda_rng=cuda_rng, tick=tick),
                root / "failure.pt",
            )
            print(
                "CAPTURED",
                json.dumps(
                    dict(
                        tick=tick,
                        text_len=int(a[1].item()),
                        text_index=kw["text_idx_num"],
                        generated_before=len(initial_kw.get("generated", [])),
                        generated_after=len(kw.get("generated", [])),
                        token_input_shape=list(kw["token_input"].shape),
                    )
                ),
                flush=True,
            )
            raise

    llm.inference_salmonn_stream = traced
    if not args.replay_only:
        data = (root / "input.pcm16").read_bytes()
        try:
            for offset in range(0, len(data), 2560):
                tick = offset // 2560 + 1
                runtime.step(data[offset : offset + 2560])
        except RuntimeError:
            if not (root / "failure.pt").exists():
                raise
        else:
            raise RuntimeError("Expected failure did not reproduce")
    saved = torch.load(root / "failure.pt", weights_only=False)
    for name in ["original", "total_minimum_met", "different_rng"]:
        torch.set_rng_state(saved["rng"].cpu())
        torch.cuda.set_rng_state(saved["cuda_rng"].cpu())
        a = saved["args"]
        kw = saved["kwargs"].copy()
        kw["generated"] = list(kw.get("generated", []))
        if name == "total_minimum_met":
            assert len(kw["generated"]) >= int(a[1].item()) * 2
            kw["min_token_text_ratio"] = 0.0
        if name == "different_rng":
            torch.manual_seed(43)
            torch.cuda.manual_seed_all(43)
        import time

        start = time.monotonic()
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                answer = original(*a, **kw)
        except RuntimeError as exc:
            print(
                "REPLAY",
                name,
                round(time.monotonic() - start, 3),
                repr(exc),
                flush=True,
            )
        else:
            print(
                "REPLAY",
                name,
                round(time.monotonic() - start, 3),
                "success",
                json.dumps(
                    dict(
                        generated=len(answer[1]),
                        new_tokens=len(answer[2]),
                        final=bool(answer[3]),
                        finite=bool(torch.isfinite(a[0]).all().item()),
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
