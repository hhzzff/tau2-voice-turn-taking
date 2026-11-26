# 通过字节 ModelHub 调用 ElevenLabs：TTS、ASR、Forced Alignment

本文使用仓库 `.env` 中的 `TERUMI_BT_AK`，通过字节 ModelHub 网关调用
ElevenLabs。完整示例见 [byted_elevenlabs_smoke.py](byted_elevenlabs_smoke.py)。

## 1. 接入约定

网关基础地址：

```text
https://aidp.bytedance.net/api/modelhub/online/multimodal/crawl/11labs/v1
```

| 功能 | 方法和相对路径 | 输入 | 输出 |
| --- | --- | --- | --- |
| 查询音色 | `GET /voices` | 查询参数 `ak` | `voices` 列表 |
| TTS | `POST /text-to-speech/{voice_id}` | JSON，见下文 | 音频字节 |
| ASR / STT | `POST /speech-to-text` | multipart：`file`、`model_id`、`language_code` | 识别文本和时间戳 |
| Forced Alignment | `POST /forced-alignment` | multipart：`file`、`text` | `words`、`characters` 及其时间戳 |

鉴权使用查询参数 `ak=TERUMI_BT_AK`，无需 `xi-api-key` 请求头。
这里的 AK 属于字节网关，不是官网的 `ELEVENLABS_API_KEY`。
不同 AK 可能对应不同 ElevenLabs 账号和音色库，应通过网关查询可用音色。

TTS JSON 中的 `model: "elevenlabs"` 是网关路由参数；`model_id` 是实际的
ElevenLabs 模型。这两个字段作用不同。ASR 和对齐按内部教程使用 multipart，
不自行添加 TTS 的 JSON 路由字段。

内部文档的 Forced Alignment 示例使用 `http://`。本示例使用同一主机和路径的
`https://`，实际验证结果见末节。

## 2. 一次运行三个接口

在仓库根目录执行。需要 Python 3.12，以及核心依赖 `requests`、`python-dotenv`；
不需要 ElevenLabs SDK、PyAudio 或本地声卡。

`.env` 中应包含以下变量（将占位内容换成自己的 AK）：

```dotenv
TERUMI_BT_AK=your_gateway_ak
```

已经安装项目依赖时直接运行：

```bash
.venv/bin/python tutorials/byted_elevenlabs_smoke.py \
  --output-dir /tmp/byted-elevenlabs-smoke-01
```

若使用外部环境，可先在该环境安装这两个依赖：

```bash
python -m pip install requests python-dotenv
python tutorials/byted_elevenlabs_smoke.py \
  --env-file .env --output-dir /tmp/byted-elevenlabs-smoke-01
```

脚本依次执行：

1. 查询当前 AK 的音色库，优先使用 Rachel，否则使用返回的第一个音色。
2. 合成一句英文，使用 `eleven_multilingual_v2`，请求 `pcm_16000`。
3. 将原始 PCM 包装为单声道、16 kHz、16-bit little-endian WAV。
4. 将该 WAV 上传一次给 `scribe_v1` ASR。
5. 将同一个 WAV 和原始文本上传一次给 Forced Alignment。

默认测试文本为：

```text
Good morning everyone, thank you for joining the meeting today.
```

这会产生一次短 TTS 合成和两次音频上传，可能产生服务费用。脚本没有自动重试；
输出目录必须不存在，以保留每次运行的证据。可用 `--voice-id` 指定已知音色并跳过
音色查询，或用 `--text` 更换英文测试句。

输出文件：

```text
input.txt       原始合成/对齐文本
tts.wav         实际返回的语音，添加 WAV 容器后保存
asr.json        ASR 的完整 JSON 响应
alignment.json 对齐的完整 JSON 响应
report.json    请求状态、耗时、log ID、模型、音频哈希与验收结果
```

只有三项验收都通过，脚本才以退出码 0 结束：音频非空且有信号；ASR 文本与输入在
忽略大小写、标点后相同；对齐覆盖输入词，时间戳有序且落在音频时长范围内。
原始响应可用于检查具体词和字符，`report.json` 会保留失败时的已完成步骤。

## 3. 分别调用三个接口

以下片段可在仓库根目录的 Python 进程中顺序执行，使用示例脚本中的客户端。
它为每个请求添加 `X-TT-LOGID`，只打印不含 AK 的相对路径、状态和请求耗时。

```python
import os
from pathlib import Path
from dotenv import dotenv_values
from tutorials.byted_elevenlabs_smoke import Gateway

ak = os.getenv("TERUMI_BT_AK") or dotenv_values(".env").get("TERUMI_BT_AK")
if not ak:
    raise RuntimeError("Missing TERUMI_BT_AK")
api = Gateway(ak)
```

### TTS：JSON 输入，二进制输出

```python
voices = api.request("GET", "/voices").json()["voices"]
voice_id = voices[0]["voice_id"]
text = "Good morning everyone, thank you for joining the meeting today."

response = api.request(
    "POST",
    f"/text-to-speech/{voice_id}",
    params={"output_format": "pcm_16000"},
    json={
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "model": "elevenlabs",
        "stream": False,
    },
)
pcm = response.content
```

`pcm_16000` 返回原始 PCM，没有 WAV 文件头。不要直接将它重命名为 `.wav`。
下面的包装不会改变采样数据：

```python
import wave

audio_path = Path("/tmp/byted-elevenlabs-example.wav")
with wave.open(str(audio_path), "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(16000)
    wav.writeframes(pcm)

duration_seconds = len(pcm) / (16000 * 1 * 2)
```

如改为 `output_format="mp3_44100_128"`，应把响应保存为 `.mp3`，并按实际格式
上传；本次验收使用 PCM。`stream: false` 是非流式请求，本教程没有验证实时 TTS。

### ASR：未知文本的识别

```python
with audio_path.open("rb") as file:
    asr = api.request(
        "POST",
        "/speech-to-text",
        files={"file": (audio_path.name, file, "audio/wav")},
        data={"model_id": "scribe_v1", "language_code": "en"},
    ).json()

print(asr["text"])
```

不要手动设置 `Content-Type: multipart/form-data`：`requests` 的 `files=` 会自动生成
带正确 boundary 的请求头。替换 `audio_path` 即可处理自己的 WAV；其他语言应按
官方 API 说明设置 `language_code`，本次仅验证英文短句。

### Forced Alignment：已知文本的时间定位

```python
with audio_path.open("rb") as file:
    alignment = api.request(
        "POST",
        "/forced-alignment",
        files={"file": (audio_path.name, file, "audio/wav")},
        data={"text": text},
    ).json()

for word in alignment["words"]:
    print(word["text"], word["start"], word["end"])

api.session.close()
```

`text` 应是音频实际说出的内容。对齐依赖提供的转写文本，不能替代 ASR。
返回的 `start`、`end` 是相对上传音频起点的秒数；`characters` 提供字符级结果。
`loss` 是服务返回的对齐损失字段，本教程不为它设定未经验证的质量阈值。
本次响应的 `words` 数组包含空格条目。统计实际词数时，应过滤
`word["text"].strip() == ""` 的条目，不能直接将数组长度当作词数。

## 4. 排查失败

- 缺少 AK：检查 `.env` 路径及变量名；`--env-file` 可指定其他文件。进程中的
  `TERUMI_BT_AK` 优先于 `.env`。
- 401 / 403：确认使用的是获准接入该服务的网关 AK；不要改用官网鉴权头。
- 400 / 422：检查字段名、音色、模型、真实音频格式，以及 multipart boundary。
- 超时或网络错误：先确认运行环境能访问该内部网关。脚本使用 10 秒连接超时、
  120 秒读取超时；读取超时不等于请求未在服务端执行，不宜盲目重复付费请求。
- HTTP 200 后验收失败：检查 `asr.json`、`alignment.json` 和实际音频。
  HTTP 成功只证明请求完成，不能单独证明识别或对齐正确。

AK 位于 URL 查询参数中，不要打印完整请求 URL、开启 HTTP 调试日志或上传 `.env`。
示例会隐藏请求异常中的 URL，并为服务错误正文遮盖当前 AK。
请求的 `wall_seconds` 是客户端收到完整响应的耗时，包含网络和服务等待，
不是首包延迟、词时间戳误差或纯模型推理时间。

## 5. 实际验证记录

2026-11-02 14:40:55–14:42:36（Asia/Shanghai），使用本仓库 `.env` 中的
`TERUMI_BT_AK` 完成一次完整在线验证，四个请求均 HTTP 200，无自动重试。
可追溯摘要见 [byted-elevenlabs.verification.json](byted-elevenlabs.verification.json)。

| 请求 | 模型 / 配置 | 客户端完整请求耗时 | 验收结果 |
| --- | --- | ---: | --- |
| 查询音色 | 当前 AK 的音色库 | 99.192 秒 | 返回可用音色列表 |
| TTS | `eleven_multilingual_v2`，`pcm_16000` | 0.778 秒 | 返回 96,596 字节 PCM，3.018625 秒非静音音频 |
| ASR | `scribe_v1`，`language_code=en` | 0.810 秒 | 10 个非空格词，归一化后与输入文本完全一致 |
| Forced Alignment | 同一 WAV + 原始文本，HTTPS | 0.420 秒 | 19 个 words 条目，其中 10 个非空格词；63 个字符条目 |

使用的音色为 `Sarah - Mature, Reassuring, Confident`，
ID 为 `EXAVITQu4vr4xnSDxMaL`，通过当前 AK 的列表查询得到。
ASR 返回：

```text
Good morning, everyone. Thank you for joining the meeting today.
```

Forced Alignment 首词 `Good` 为 0.16–0.28 秒，末词 `today.` 为 2.28–2.66 秒。
所有 words 条目的时间戳有序、未超出音频范围，文本覆盖检查通过。
独立用 `ffprobe` 确认 WAV 可解码，编码为 `pcm_s16le`、16 kHz、单声道，
时长 3.018625 秒。

原始本地产物保留于
[`artifacts/byted-elevenlabs/20261102-smoke/`](../artifacts/byted-elevenlabs/20261102-smoke/)，
包括 `tts.wav`、ASR 和对齐 JSON、请求报告；`artifacts/` 已被仓库忽略。
音频 SHA-256：

```text
7103f2b73ca708fc83d8d92721196d3e9cb3b91f4e878fdffebc1cfcf9f0b80e
```

这是一个英文短句、一个音色、一次运行的接入验证，未验证中文、长音频、并发、
流式 TTS 或对齐精度。表中耗时是该次观察值，不代表延迟分布。
音色查询本次较慢，已知可用音色时可传 `--voice-id` 跳过查询。

## 参考资料

- [ModelHub Elevenlabs 模型使用说明（内部）](https://bytedance.larkoffice.com/wiki/Fj43w6fEaiVusRkGsSecwtDynTf)：网关地址、AK、TTS 路由字段及上传示例；本次读取 revision 1485。
- [ElevenLabs TTS API](https://elevenlabs.io/docs/api-reference/text-to-speech/convert)：音色路径、模型和输出格式。
- [ElevenLabs Speech to Text API](https://elevenlabs.io/docs/api-reference/speech-to-text/convert)：ASR 参数与响应结构。
- [ElevenLabs Forced Alignment API](https://elevenlabs.io/docs/api-reference/forced-alignment/create)：上传音频与文本、词和字符结果。
