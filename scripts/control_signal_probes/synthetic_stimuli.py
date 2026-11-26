"""Generate only fixed, non-personal synthetic English probe utterances."""

import argparse
import base64
import hashlib
import json
import os
import sys
import wave
from pathlib import Path

import requests

sys.path.insert(0, "/mnt/bn/audio-visual-llm-data6/terumi/pi-bench")
from inference.repo_credentials import load_repo_credentials  # noqa: E402

TEXTS = {
    "greeting": "Hello. Please explain a relaxed three day travel plan for Beijing, including the sights, food, and how to get around each day. Please give me plenty of detail.",
    "interruption": "Please stop speaking. I have an important correction. The trip is only two days, not three. Please stop and listen to me.",
    "backchannel": "Yes, that makes sense. Please go on.",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--language", choices=["en", "zh"], default="en")
    parser.add_argument(
        "--scenario", choices=["controls", "equivalence"], default="controls"
    )
    args = parser.parse_args()
    root = args.output
    texts = (
        TEXTS
        if args.language == "en"
        else {
            "greeting": "你好，请详细介绍一下北京三日游的行程，每天怎么玩、吃什么，以及交通怎么安排。",
            "interruption": "请停一下，我有一个重要的修改。我的行程只有两天，不是三天。请先停下来听我说。",
            "backchannel": "对，说得有道理，请继续。",
        }
    )
    if args.scenario == "equivalence":
        if args.language != "en":
            parser.error("The fixed equivalence stimulus is English")
        texts = {
            "greeting": "Hello. Please count from one to one hundred, slowly.",
            "interruption": "Stop counting. What is seven plus five? Answer with just the number.",
        }
    root.mkdir(parents=True, exist_ok=False)
    load_repo_credentials()
    headers = {
        "X-Api-Key": os.environ["DOUBAO_S2S_API"],
        "X-Api-Resource-Id": "seed-tts-2.0",
        "Content-Type": "application/json",
    }
    metadata = {}
    for name, text in texts.items():
        response = requests.post(
            "https://openspeech.bytedance.com/api/v3/tts/unidirectional",
            headers=headers,
            json={
                "req_params": {
                    "text": text,
                    "speaker": "zh_female_xiaohe_uranus_bigtts",
                    "audio_params": {"format": "pcm", "sample_rate": 16000},
                }
            },
            stream=True,
            timeout=60,
        )
        response.raise_for_status()
        pcm = bytearray()
        for line in response.iter_lines():
            if not line:
                continue
            event = json.loads(line)
            if event.get("code") == 0 and event.get("data"):
                pcm.extend(base64.b64decode(event["data"]))
            elif event.get("code") == 20000000:
                break
            elif event.get("code", 0) > 0:
                raise RuntimeError(
                    "Synthetic TTS returned error code " + str(event["code"])
                )
        if not pcm:
            raise RuntimeError("No synthetic TTS output")
        path = root / f"{name}.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(pcm)
        metadata[name] = {
            "text": text,
            "seconds": len(pcm) / 32000,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        print(name, metadata[name]["seconds"], flush=True)
    (root / "provenance.json").write_text(
        json.dumps(
            {
                "source": "new fixed synthetic text, no user conversation data",
                "utterances": metadata,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
