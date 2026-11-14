# 四个开源模型的实际编排路径对照

本轮仅测试 SALMONN、Moshi、MiniCPM、Freeze-Omni，不再用闭源跨请求延迟
作为编排器公平性的证据。结果文件保存于
`data/simulations/control-signals/20261112-open-equivalence/`。

## 比较边界

A：完整固定音轨按原生输入块的结束时间实时发送给模型运行时；收到可用
输出后直接进入 FIFO 播放，保留推理计算时间。

B：相同音轨经 `ReplayUser` → `FullDuplexOrchestrator.step()` →
`DiscreteTimeAudioNativeAgent` → `OpenModelAdapter` → 模型运行时。
播放音轨直接拼接编排器保存的各 tick 音频，**没有事后平移、量化或回填**。
当前生产策略是 frozen：推理等待不计入有效对话时间。本轮测量这个现状，
并未把它替换成上一轮 100ms 播放原型。

| 模型 | 输入处理粒度 | 运行时边界 |
|---|---:|---|
| SALMONN checkpoint-44000 | 80ms | A 直接 NativeSession，B 经生产 SalmonnWorker |
| Moshika-pytorch-bf16 | 80ms | 已有 pi-bench 原生帧控制器，B 加实验 transport 绑定 |
| MiniCPM-o-4.5 | 1000ms | A 直接 UnifiedProcessor SDK，B 经生产 MiniCPMWorker |
| Freeze-Omni | 160ms | 已有原生监听/生成参考控制器，B 加实验 transport 绑定 |

Moshi、Freeze 的绑定不是已经注册的正式生产 provider；实际通用 adapter、
agent 和 orchestrator 都有运行，但不能因此宣称这两个 provider 已完整集成。
SALMONN、Moshi、Freeze 两条路径共享已有原生会话/控制器。因此本轮能定位
接入和编排差异，**不能独立证明共享运行时与模型原仓库服务器完全等价**；
尤其 Freeze 参考控制器每个输入步最多执行 16 个生成 token，其并发策略仍需
与原服务器另外核验。MiniCPM 两边均不施加 set_break/force_listen，测试的是
原生 SDK 在相同控制条件下的行为，不宣称参考应用的外部打断策略已通过。

## 控制条件

- 复用同一段 40 秒合成英文音轨：0 秒请求慢慢数到 100，10 秒要求停止并
  回答 7+5；剩余时间为静音。无动态 user LLM、TTS 或闭源模型调用。
- τ-voice 采用 8kHz μ-law；A 同样先经过该格式再送模型，逐块校验最终
  模型输入 hash。这控制了编解码变量，不证明相对原始高采样率输入无损。
- 每模型一个 A100 任务，四次新进程 A/B/B/A。同一任务 GPU UUID 保存在
  `jobs/<model>-gpu.csv`。保留配置中的原生参数，初始化后统一重置生成 seed=42。
- A、B 的系统提示完全一致，保存为每次运行的 `prompt.txt`。Moshi 无文本
  system prompt 通道，不伪造提示生效。
- 记录输入块 hash、输出 PCM hash、文本、控制状态、生成就绪时刻、返回
  时刻、实际 tick 音频和有效时间。A 的 MiniCPM 在 generate 完成后记录可用
  时刻，B 保留生产 worker 在 finalize 后才返回的现状。
- 按逐项绝对误差 ≤100ms 检查；直接路径的重复波动另列，不用均值抵消超限。
- 首次开口由 10ms RMS 窗口检测。第二句话结束后的新有声段不自动等于
  新回答；需结合文本/音频判定。片段缺失、未说话、未恢复都不算通过。

## 本地验证

`tests/test_streaming/test_open_equivalence.py` 四项通过：通过真实 `step()`
送入不同音频区间，每块恰好一次，原生样本数正确，保留初始静音 tick。
沙箱内 asyncio 线程收尾会挂起；同样的纯本地测试在正常执行环境通过
（4 passed，40.61s，包括依赖加载）。这不是 GPU 模型测试的替代。

## 作业与证据

| 模型 | Merlin SID |
|---|---|
| SALMONN | `67c40a23179e6897` |
| Moshi | `fade36c5106806a7` |
| MiniCPM | `20a7f0be401a8049` |
| Freeze-Omni | `b6269a9e1bbf94bc` |

四个提交预检均为 success；将 precheck SID 保留式合并到 request_context 后
在 30 分钟内提交。冻结代码在 `snapshot/`，配置在 `configs/`，hash 清单在
`SHA256SUMS`。每次运行的 `native.jsonl`、`ticks.jsonl`、`played.wav`、
`native-output.pcm16`、`result.json` 保留全部证据；异常写入 `error.json`。

汇总命令（只读已有运行证据，不调用模型）：

```bash
.venv/bin/python scripts/control_signal_probes/summarize_open_equivalence.py \
  data/simulations/control-signals/20261112-open-equivalence
```

## 结果

### MiniCPM 和 Freeze-Omni

| 模型 | A0 / B1 首次开口 ms | A3 / B2 首次开口 ms | B−A | 原始模型输出 |
|---|---:|---:|---:|---|
| MiniCPM | 5790 / 4520 | 5460 / 4520 | -1270 / -940ms | 四次各 40 块逐块完全相同 |
| Freeze-Omni | 5640 / 3730 | 5410 / 3730 | -1910 / -1680ms | 四次各 250 块逐块完全相同 |

两者均超出 100ms 门槛。输入、输出 PCM hash 和可观察控制状态完全一致，
所以这些样本中的差异出现在输出可用时间/编排播放时间上，而不是模型产生
了不同的原始音频。冻结有效时间没有计入真实推理等待；正常尾音和句间空隙
也会受队列影响。不能只看模型输出 hash 就宣布听到的对话等价。

MiniCPM 四次都继续计数，没有按第二句话切换问题；这里两边都没有施加外部
强制监听，不代表模型在另一种原生控制策略下也无法被打断。Freeze 的
参考控制器与服务器并发策略、外部 stop-TTS 信号接入仍有上述独立限制。

### SALMONN

初次直接路径失败，随后用完全相同的代码、音频、checkpoint、seed 在同一
新任务中跑 A/B/B/A，允许单次失败后继续下一次。四次都在处理第 194 个
80ms 输入块时发生以下异常；此前 193 块的输入和输出 hash 四次完全一致：

```text
sampling reaches max_trials 50 and still get eos when ignore_eos is True, check your input!
```

调用链为 `NativeSession._finish_tick` → `FDBSpeechPlayback._synthesize_step`
→ CosyVoice `inference_salmonn_stream` → `sampling_ids`。这属于两条路径共享
的原生/TTS 路径异常，尚未证明是 checkpoint、TTS 配置还是其接入方式的问题；
不能归因于编排器，也不能算完整对话通过。本轮未改变 EOS 采样策略绕过异常。

复现任务 `99b5b46874b2530c`，证据目录
`data/simulations/control-signals/20261112-open-equivalence-salmonn-repro/`。

### Moshi：冷启动与预热分开

冷启动四次原始音频各 500 块完全一致。A0/B1 首次开口为 21250/250ms，
A3/B2 为 9630/250ms。直接路径首步计算分别含巨大的冷启动开销，不能用
这组差值代表稳态响应延迟。

原仓库 `moshi/server.py` 的 `ServerState.warmup()` 在接收会话前预热四帧，
再在连接时重置 mimi/LM 流状态。已追加独立预热对照：调用该原始函数体，
重置流状态和生成 seed 后才开始计时，不混入用户音轨。第一次入口尝试因
服务器模块在 import 时直接解析 CLI 而退出（0 个输入块），保留在
`20261112-open-equivalence-moshi-warm/`，不是模型运行结果。
修正为从 AST 提取未经修改的 warmup 方法并单独执行，避免启动服务器。
原函数文件路径和 hash 保存在 `warmup-source.json`。

有效预热补测任务 `2f4d9b1147006cef`，证据目录
`data/simulations/control-signals/20261112-open-equivalence-moshi-warm-v2/`。
预热后四次各 500 个输入块和原始输出块完全一致；可观察状态和文本也逐块
一致。A0/B1、A3/B2 首次开口都是 300/250ms，差值均 -50ms。六段有声区间
数量一致，所有起止边界的最大绝对差均为 60ms；两个 A 自身的边界最大差为
10ms。**这段音轨的预热后接入/编排对照满足 100ms 门槛**，不外推到全部任务
或独立原服务器一致性。汇总字段 `observed_replay_contract_pass=true` 表示
本例输入/原始输出一致且活动边界达标；`full_equivalence_pass=false` 保留
上述完整原生服务器与生产 provider 验收尚未闭合的限制。

### 本轮结论

- Moshi：原服务器预热后，本例时序和原始音频内容等价。
- MiniCPM、Freeze-Omni：原始内容一致，当前编排有效时间不等价，未达 100ms。
- SALMONN：原生/TTS 路径异常，两条路径均无法完成本例，不能验收。

本轮没有修改生产时钟或模型控制策略来让结果通过，也没有把输出音轨整体
平移。下一步需要解决冻结有效时间忽略模型计算等待的问题，并单独修复
SALMONN 共享 TTS 路径异常，随后复用此音轨和同一测试入口回归。

## 后续：100ms异步接入复测

已完成独立的100ms异步/绝对时钟对照，见
[复测报告](open-model-realtime-100ms-20261112.md)。原始输出逐块一致，主要
秒级提前误差收敛，但仍未全部达到100ms边界门槛；未更改默认生产配置。
