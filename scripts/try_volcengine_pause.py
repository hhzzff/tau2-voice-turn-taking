"""Compare natural and SSML pauses using Volcengine's HTTP streaming TTS."""

import argparse
import ast
import base64
import html
import json
import os
import time
import wave
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

URL = "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
TEXT = (
    "For tomorrow's dinner, I'd like something with "
    "lentils and rice, preferably all in one pan."
)
PREFIX = "For tomorrow's dinner, I'd like something with"
CONTINUATION = "lentils and rice, preferably all in one pan."
SAMPLE_RATE = 24000


def resolve_key(example: Path | None) -> str:
    """Read credentials locally without executing example code or logging them."""
    load_dotenv()
    key = (
        os.getenv("VOLCENGINE_TTS_API_KEY")
        or os.getenv("DOUBAO_TTS_API_KEY")
        or os.getenv("DOUBAO_SEED_API_KEY")
    )
    if key:
        return key
    if example:
        tree = ast.parse(example.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for name, value in zip(node.keys, node.values):
                    if isinstance(name, ast.Constant) and name.value == "X-Api-Key":
                        try:
                            key = ast.literal_eval(value)
                        except (ValueError, TypeError):
                            continue
                        if isinstance(key, str) and key:
                            return key
    raise ValueError("Configure VOLCENGINE_TTS_API_KEY in the environment or .env")


def silence_intervals(raw: bytes) -> list[dict]:
    """Measure low-energy intervals in 10 ms frames, excluding edge silence."""
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    width = SAMPLE_RATE // 100
    frames = samples[: len(samples) // width * width].reshape(-1, width)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    # This is an energy measurement, not word alignment or a semantic VAD.
    threshold = max(20.0, float(np.max(rms)) * 0.01)
    quiet = rms < threshold
    spans = []
    start = None
    for i, silent in enumerate([*quiet, False]):
        if silent and start is None:
            start = i
        elif not silent and start is not None:
            if start > 0 and i < len(quiet) and i - start >= 20:
                spans.append(
                    {
                        "start_seconds": start / 100,
                        "end_seconds": i / 100,
                        "duration_seconds": (i - start) / 100,
                    }
                )
            start = None
    return spans


def main() -> None:
    """Generate comparable samples and an offline playback report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--credential-example", type=Path)
    parser.add_argument("--speaker", default="zh_female_vv_uranus_bigtts")
    parser.add_argument(
        "--pause-ms", type=int, choices=range(1, 10001), metavar="1..10000"
    )
    args = parser.parse_args()
    key = resolve_key(args.credential_example)
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [
        ("plain", "普通完整句", {}),
        (
            "ssml_1500ms",
            "SSML：with 后停顿 1500 ms",
            {"ssml": f'<speak>{PREFIX}<break time="1500ms"/> {CONTINUATION}</speak>'},
        ),
        (
            "ssml_3000ms",
            "SSML：with 后停顿 3000 ms",
            {"ssml": f'<speak>{PREFIX}<break time="3000ms"/> {CONTINUATION}</speak>'},
        ),
        (
            "instruction_1500ms",
            "语气指令：自然思考并停顿约 1.5 秒",
            {
                "context_texts": [
                    "Speak English naturally, as someone thinking aloud. After the "
                    "word 'with', pause silently for about one and a half seconds "
                    "with an unfinished intonation, then smoothly finish the same "
                    "sentence. Do not add filler words."
                ]
            },
        ),
    ]
    if args.pause_ms is not None:
        milliseconds = args.pause_ms
        cases = [
            (
                f"ssml_{milliseconds}ms",
                f"SSML: {milliseconds} ms pause after with",
                {
                    "ssml": f'<speak>{PREFIX}<break time="{milliseconds}ms"/> {CONTINUATION}</speak>'
                },
            )
        ]
    results = []
    for name, label, extra in cases:
        extra = dict(extra)
        additions = {"disable_markdown_filter": False}
        if "context_texts" in extra:
            additions["context_texts"] = extra.pop("context_texts")
        params = {
            "text": TEXT,
            "speaker": args.speaker,
            "audio_params": {
                "format": "pcm",
                "sample_rate": SAMPLE_RATE,
                "enable_subtitle": True,
            },
            "additions": json.dumps(additions),
            **extra,
        }
        (args.output / f"{name}.request.json").write_text(
            json.dumps({"req_params": params}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        started = time.perf_counter()
        row = {"name": name, "label": label, "request": params}
        try:
            audio = bytearray()
            metadata = []
            completed = False
            with requests.post(
                URL,
                headers={
                    "X-Api-Key": key,
                    "X-Api-Resource-Id": "seed-tts-2.0",
                    "Content-Type": "application/json",
                },
                json={"req_params": params},
                stream=True,
                timeout=(15, 90),
            ) as response:
                row["http_status"] = response.status_code
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    message = json.loads(line)
                    code = message.get("code", 0)
                    metadata.append({k: v for k, v in message.items() if k != "data"})
                    if code == 0 and message.get("data"):
                        audio.extend(base64.b64decode(message["data"]))
                    elif code == 20000000:
                        completed = True
                        break
                    elif code:
                        row["provider_code"] = code
                        raise RuntimeError("Provider returned a non-success status")
            if not audio or not completed:
                raise RuntimeError("Empty or incomplete TTS stream")
            filename = f"{name}.wav"
            with wave.open(str(args.output / filename), "wb") as recording:
                recording.setnchannels(1)
                recording.setsampwidth(2)
                recording.setframerate(SAMPLE_RATE)
                recording.writeframes(audio)
            row.update(
                status="success",
                audio_file=filename,
                duration_seconds=len(audio) / 2 / SAMPLE_RATE,
                internal_silences=silence_intervals(audio),
            )
            (args.output / f"{name}.response.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            row["status"] = "error"
            row["error_type"] = type(exc).__name__
        row["wall_seconds"] = round(time.perf_counter() - started, 2)
        results.append(row)
        (args.output / "results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {k: v for k, v in row.items() if k != "request"}, ensure_ascii=True
            )
        )
        # Stop on authentication failure rather than repeating rejected requests.
        if row.get("http_status") in (401, 403):
            break
    page = [
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>豆包 TTS 句中停顿试听</title>",
        "<style>body{max-width:900px;margin:40px auto;padding:0 20px;"
        "font-family:Segoe UI,Microsoft YaHei,sans-serif;background:#faf8f4;color:#302e2a}"
        "article{padding:20px;margin:20px 0;background:white;border-radius:12px}"
        "audio{width:100%}p{line-height:1.8}code{overflow-wrap:anywhere}</style>",
        "<h1>豆包 TTS：句中停顿对比</h1>",
        f"<p>{html.escape(TEXT)}</p>",
        f"<p>音色：{html.escape(args.speaker)}；24 kHz 单声道 PCM。每项仅一次合成，客户端未拼接静音。</p>",
        "<p>静音检测采用 10 ms 能量帧，列出内部不少于 0.2 秒的低能量区间。"
        "它不代表词级对齐，也不能单独证明停顿发生在指定词后；请结合试听与字幕判断。</p>",
    ]
    for row in results:
        page.append(f"<article><h2>{html.escape(row['label'])}</h2>")
        if row["status"] == "success":
            page.append(
                f'<audio controls preload="metadata" src="{row["audio_file"]}"></audio>'
            )
            page.append(f"<p>总时长：{row['duration_seconds']:.2f} 秒。</p>")
            for span in row["internal_silences"]:
                page.append(
                    f"<p>内部低能量区间：{span['start_seconds']:.2f}–"
                    f"{span['end_seconds']:.2f} 秒，持续 {span['duration_seconds']:.2f} 秒。</p>"
                )
        else:
            page.append(f"<p>合成未成功：{html.escape(row['error_type'])}</p>")
        page.append("</article>")
    page.append("</html>")
    (args.output / "index.html").write_bytes(
        "\n".join(page).encode("ascii", errors="xmlcharrefreplace")
    )


if __name__ == "__main__":
    main()
