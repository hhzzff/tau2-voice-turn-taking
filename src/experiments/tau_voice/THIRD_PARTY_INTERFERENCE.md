# 双音色第三者背景干扰

准备配置（不调用 API）：

```powershell
.venv\Scripts\python.exe -m experiments.tau_voice.third_party_interference --output data/simulations/third_party_prepared_20261028
```

正式运行一条：

```powershell
.venv\Scripts\python.exe -m experiments.tau_voice.third_party_interference --scenario data/simulations/third_party_prepared_20261028/scenario.json --output data/simulations/third_party_prepared_20261028 --run
```

在仓库根目录 `.env` 设置 `OPENAI_API_KEY`。本场景的用户生成和语义判断默认用 `gpt-6.1-terra`（`reasoning_effort=none`，不传 temperature），TTS 用 `gpt-4o-mini-tts`，被测模型用 `gpt-realtime`。无需豆包、ElevenLabs 或 Deepgram。正式运行前可编辑 `scenario.json` 中的 topic、音色、两句背景话语、纠正话语和各时间门槛。主用户与第三者音色必须不同；自定义话题时也应同步修改纠正话语。

默认场景：

`language` 默认是 `Chinese`；英文测试设为 `English`，并将 `topic`、`third_party_texts` 和 `correction` 同步改为英文。每次运行使用新的输出目录，保留此前记录。

1. 主用户以 `nova` 音色与助手讨论京都三日旅行，围绕交通、预算、少走路继续追问。
2. 至少生成一轮主用户发言，且助手连续说话 2 秒后，以 `onyx` 音色播放无关的鸡蛋、牛奶话题。第三者原始 TTS 音量衰减 8 dB，经过混音和现有电话音频链路输入助手。
3. 观察后端 `was_truncated` 标记或至少 0.6 秒的停顿，结合 GPT 对后续实际播放转写的判断，识别助手是否回应第三者。仅当判断为 yes、第三者发言已结束且助手仍在说话时，主用户插入预合成纠正语音，要求不要回应旁人，然后回到旅行话题。没有转而回应时不纠正。
4. 第一阶段观察结束后，至少再生成一轮正常主用户发言，且间隔至少 15 秒，再在助手连续说话 2 秒后播放篮球话题。观察第二次干扰，不追加第二次纠正。
5. 每次背景语音结束后最多观察 18 秒，整条对话最多 240 秒；未触发两次事件、判断不确定、纠正错过助手发言窗口等情况都会记录为 incomplete。

助手仅接收普通助手策略与混合音频，不接收测试脚本、背景话语文本或说话人标签。主用户不随机插话或生成附和语，避免引入额外干扰。背景话语不写入主用户的文字历史。

产物位于输出目录：

- `scenario.json`：实际场景配置。
- `simulation.json`：完整逐 tick 轨迹。
- `audio/events.json`：两次事件时间、截断标记时间、停顿时间、后续转写、语义判断及证据、纠正起止时间、覆盖状态。
- `audio/primary_user.wav`、`audio/third_party.wav`：16 kHz PCM 独立同步音轨，后者包含衰减。
- `audio/both.wav`：用户输入混合音频在左、助手在右；`both_mono.wav`：方便试听的单声道混音。
- `audio/user_labels.txt`、`audio/assistant_labels.txt`、`audio/third_party_labels.txt`：完整转写标签。

事件时间采用用户输出音轨时基；助手观测来自前一个 tick（默认 0.2 秒）。停顿可能是自然结束，不能单独证明第三者造成打断；应结合后端截断标记、语义证据和音频复核。语义判断使用文本和时间，不评价音色自然度。背景音量相对于其自身 TTS 输出衰减，没有模拟房间空间声学。`complete` 表示场景覆盖完成，不表示被测模型通过。失败运行保留 `events.json` 与说话人轨道；重新运行应使用新输出目录。

离线验证：

```powershell
.venv\Scripts\python.exe -m pytest tests/test_streaming/test_third_party_simulator.py -q
```
