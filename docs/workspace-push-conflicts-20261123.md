# 本地语音实验代码推送与冲突清单

检查日期：2026-11-23。目标仓库：`hhzzff/tau2-voice-turn-taking`。
发布分支：`feat/duplex-voice-workspace-20261123`。

## 分支范围

本次包含工作区新增和修改的双工用户模拟器、事件评审、豆包 Seeduplex 适配、Gemini 转写、SSML 停顿测量、背景说话人实验、展示生成脚本、场景配置、相关测试和说明文档。

本地与远端共同基点为 `1a3fec510b27e9d6c2c5652f8938a56483a477f2`。检查时远端 `main` 为 `0f634c66f9a90f573b6ca380229895fa992c391c`，比共同基点多 43 个提交。本分支从共同基点保存本地代码，供单独审阅；尚未整合远端后续的实时调度和声学澄清功能。

未纳入本次分支的文件和产物：

- `runs/`、`data/simulations/`、日志、缓存、实际运行录音和评审输出。
- `docs/docs.7z`、生成的 `docs/background-speaker-audio.html`。
- 本地交接记录 `HANDOFF.md` 的修改和 `provider_suite_results.txt` 的修改。
- 本地未发布的 `37133e5` 演示文档提交，避免连带上传其中约 6 MB 的完整会话录音。

原工作区的分支和未提交文件保留。本次没有覆盖、变基或强制推送远端 `main`。

## 已确认的文本冲突

使用 `git merge-tree --write-tree --messages` 将本分支快照与上述远端 `main` 模拟合并，确认以下 6 个文件需要手工处理。没有在工作目录执行实际合并。

| 文件 | 冲突块数 | 需要协调的内容 |
|---|---:|---|
| `src/experiments/tau_voice/DUPLEX_BENCH.md` | 1 | 远端新增统一报告入口；本地新增多模型比较、统一 Judge、豆包及停顿校准说明。应保留适用于最终实现的两部分说明。 |
| `src/experiments/tau_voice/duplex_bench.py` | 6 | 远端的 OpenAI/ElevenLabs 网关合成配置与本地 Volcengine 合成配置、退出清理和中断证据保存、声学澄清覆盖统计与本地补充事件统计、CLI 参数。 |
| `src/tau2/data_model/voice.py` | 1 | 远端新增 `doubao-asr`、`scribe_v1`，本地新增 `gemini-3.8-flash`；转写模型枚举应结合实现合并。 |
| `src/tau2/user/duplex_events.py` | 5 | 远端的声学澄清场景及关键词字段，与本地事件顺序、补充主动事件、事件指导和预期补全字段。空事件集合的合法条件也不同。 |
| `src/tau2/user/duplex_perception.py` | 1 | 远端通过完整 `TranscriptionConfig` 注入感知配置，本地通过模型名及默认转写常量配置；构造接口及调用点需要统一。 |
| `src/tau2/user/duplex_scenario_user.py` | 6 | TTS 提供商选择和 SSML 停顿、声学退化及证据记录、用户发声状态、ASR 构造调用、助手讲话时的等待条件。 |

## 双方修改但自动合并的文件

以下 7 个文件双方均有改动，Git 本次未报告文本冲突。这只表示可以拼接修改，不代表整合后的功能已经验证：

- `src/tau2/agent/discrete_time_audio_native_agent.py`
- `src/tau2/config.py`
- `src/tau2/data_model/simulation.py`
- `src/tau2/user/duplex_evaluation.py`
- `src/tau2/voice/audio_native/adapter.py`
- `src/tau2/voice/audio_native/openai/live_discrete_time_adapter.py`
- `src/tau2/voice/transcription/transcribe.py`

后续合并时还需要检查远端 100 ms 实时调度与本地同步决策、连续音频适配、观察窗口之间的兼容性，并同时运行双方回归测试。

## 提交前验证

- 已执行 `make check-all` 对应的 Ruff lint 与格式命令（当前 Windows 没有 `make`），检查通过。
- 双工场景、已播放音频感知、背景说话人实验、连续音频适配、Gemini 转写和 Volcengine TTS 的离线测试：101 项通过。
- 音频活动标签测试另有 5 项通过，合计 106 项离线测试通过。
- 已按音频提供商目录要求运行 `run_provider_suite.py`。首次汇总输出遇到 Windows GBK 编码错误；设置 UTF-8 后重跑，结果为 11 passed、18 failed、44 skipped、4 errors。失败项包含 Gemini/xAI 单 tick 超出 300 ms 阈值和 OpenAI Realtime 会话未返回预期的 `session.created`。该在线套件未通过；本次未放宽时延断言，也未将生成的测试结果文件提交。
- 额外运行原有 `test_discrete_time_audio_native_agent.py` 时，因其引用不存在的 `tau2.voice.audio_native.openai.tick_runner` 而收集失败；已核实共同基点中也有同一过期导入。本次没有修改该旧测试。
- 已检查提交清单、Python/JSON 语法、常见凭证格式和 Git 空白错误；未纳入运行产物。

这些验证针对发布分支本身，不代表它已经与最新远端 `main` 完成整合验证。
