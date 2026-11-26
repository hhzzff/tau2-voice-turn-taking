# 双工语音分支合并验证（2026-11-26）

本地分支：`merge/duplex-voice-main-20261126`。

合并输入：发布分支 `713a47a`（包含本地功能提交 `f699dde`）与远端 `main` 的 `0f634c6`。原开发目录保留；合并、修改和测试在 `C:/code/tau2-bench-merge-20261126` 完成。

## 冲突处理与保留功能

| 部分 | 合并方式及验证范围 |
|---|---|
| 场景与决策结构 | 同时保留 `event_order`、补充主动事件、剧情指导、Gemini 转写模型、CTC 预期补全、声学任务配置及最多四个关键词标注。声学单独运行不被自动增加普通事件；普通事件仍默认增加附和和补全。 |
| TTS | 普通语义事件默认 Volcengine Seed TTS 2.0，声学任务默认 OpenAI，ModelHub 网关默认 ElevenLabs。增加 `--tts-provider` 显式选择；实际提供方、模型和音色写入证据。 |
| 停顿 | Volcengine 保留单次 SSML 750ms 停顿；OpenAI/ElevenLabs 可使用显式 PCM 静音路径，播放前都核验停顿。两种路径在证据与文档中区分，不作为完全同条件样本混比。 |
| ASR | `PlayedAudioPerception` 兼容完整 `TranscriptionConfig` 注入和 `model=` 参数。普通运行沿用场景模型，网关保留 Scribe。 |
| 观察及让话 | 声学任务可复用共享让话策略并在非记忆事件观察期回答追问；记忆回答仍保护至完整播放。助手简短附和不会截断用户长陈述。 |
| 评审 | 保留本地的时序与语义集成 Judge、记忆逐事实评审，以及远端的声学有效性、澄清、修复和逐信息值评审。 |
| 开源模型 | 保留 100ms 异步调度、80ms/1000ms 原生输入分块、真实播放记录、控制信号、会话隔离及历史对照入口。默认通用 tick 继续使用远端的 100ms。 |
| 豆包原生语音 | 保留本地 Seeduplex 适配器，并接入远端的控制信号证据记录和公共播放取消方法。 |
| 报告与异常保存 | 同时保留离线报告命令、实际输入与声学证据、部分轨迹和录音保存、Judge 前的规则结果落盘。报告说明同步为集成 Judge 和 100ms 调度。 |

六个文本冲突文件均已处理：`DUPLEX_BENCH.md`、`duplex_bench.py`、`data_model/voice.py`、`duplex_events.py`、`duplex_perception.py`、`duplex_scenario_user.py`。未使用整文件单边覆盖。

## 离线验证

完整语音及流式回归：**599 passed、5 skipped、83 deselected**。其中 83 项按 `not full_duplex_integration` 排除，5 项跳过包含未配置独立 SALMONN 解释器的原生契约测试及可选环境用例。

覆盖本地事件目录、全部事件执行、事件顺序、补充事件、记忆保护、停顿及 CTC、背景说话人实验、连续音频、Gemini 转写、Volcengine TTS，也覆盖远端两种声学扰动、关键词对齐与多关键词、实际交付、追问和修复评审、ModelHub、开源 adapter/worker、100ms 窗口和会话隔离、控制信号及报告。

另新增 11 项组合用例，已经包含在上述 599 项中：

- 单独声学任务与带事件顺序的组合配置互不改变覆盖范围。
- 三种 TTS 经过实际用户输出链路后，停顿位置及合成证据正确；Volcengine 仍只发一次 SSML 请求。
- 启用声学任务后仍保护完整记忆回答。
- 启用声学扰动后，用户长陈述在助手简短附和期间不会被取消。
- 默认及显式 TTS、网关 Scribe 和场景 Gemini ASR 选择正确；运行中断仍保存轨迹、规则结果与声学证据。
- 同一份报告同时保存普通事件的集成语义评审和声学任务评审，互不覆盖。

现有声学让话测试增加 OpenAI/Volcengine 两条路径。第三方插话测试改为按秒驱动，并同时覆盖 100ms、200ms；不放宽生产时序条件。修复 Windows 源码哈希键使用反斜杠的问题；SALMONN 默认配置测试改为替换外部清单与训练配置边界，不再依赖默认离线环境中不存在的集群文件。

Windows 测试使用 `PYTHONUTF8=1`、`PYTHONIOENCODING=utf-8`；隔离目录复用原目录的 Python 3.12 虚拟环境，并通过 `PYTHONPATH` 指向合并后的源码。未改变项目依赖或模型权重。

```powershell
$env:PYTHONPATH='C:\code\tau2-bench-merge-20261126\src'
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
& 'C:\code\tau2-bench\.venv\Scripts\python.exe' -m pytest tests/test_streaming tests/test_voice -q -m 'not full_duplex_integration'
```

Ruff 全仓 lint、格式检查与 Git 空白检查通过。当前 Windows 没有 `make`，执行了 `make check-all` 对应的两条 Ruff 命令。

## 在线实测与限制

| 检查 | 结果 | 可以支持的结论 |
|---|---|---|
| Volcengine 短句单次 SSML 合成 | 成功，16kHz 音频长约 3.44 秒；750ms 请求的停顿实测约 740ms | 本地 TTS 与停顿波形链路可用。 |
| Gemini 转写上述实际合成音频 | 成功，得到非空文字 | Gemini 已播放音频转写调用可用；不代表长对话转写完全准确。 |
| 官方在线 Provider 套件 | 4 passed、21 failed、44 skipped、8 errors | 未通过。包含 OpenAI 会话建立错误、xAI HTTP 403、部分 tick 超过 300ms 阈值；未修改或放宽这些断言。 |
| 合并前分支代表性在线对照 | 同一环境复测 OpenAI 建连、xAI 建连、Gemini 短句时延，3 项均失败 | 这三类代表性问题在未合并代码上也存在；不证明所有在线失败的根因都已定位。 |
| 豆包 Seeduplex 100ms 连接烟测 | 合并前后均失败；基线返回 `45000292`、`tokens_lifetime` 配额耗尽 | 当前凭证配额不能完成真实原生语音验证。不能将其报告为在线通过。 |
| ModelHub、SALMONN 实际环境 | 本机没有 ModelHub 凭证、独立原生解释器及集群模型资源 | 网关和开源模型通过受控回归验证，尚未完成合并后真实服务或权重验收。 |

本次验证证明双方功能在合并后的离线路径中保留；不能宣称所有真实模型端到端测试通过。短语音烟测脚本首次误将工厂返回元组当作 adapter，修正后重测；该次脚本错误与后续实际服务配额失败分开留档。

## 本地证据

所有测试 XML、日志及短录音位于被忽略的 `data/simulations/merge_validation/`。最终完整离线结果为 `voice-all.xml`、`voice-all.log`；在线结果见 `provider-suite.log`、`baseline-online.xml`、`live-smoke.json`、`doubao-live.json`、`baseline-doubao.json`。这些产物和生成的 `provider_suite_results.txt` 不纳入提交。
