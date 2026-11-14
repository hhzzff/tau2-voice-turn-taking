# 100ms 异步开源接入对照

## 要验证的判断

当前闭源 OpenAI 适配器 `_execute_tick()` 用 `asyncio.gather` 并行发送
输入及收集窗口内事件，不等待模型整段生成结束。开源 `OpenModelAdapter`
则等待 `transport.request("step")` 返回，冻结有效时间，再播放结果。
因此两者不是同一个执行策略。此前 MiniCPM / Freeze-Omni 的提前开口是
**记录时间被压缩**，不是模型真的提前产生了音频。

本次实验增加 `scripts/control_signal_probes/realtime_adapter.py`：

- 实际 `FullDuplexOrchestrator.step()` / `DiscreteTimeAudioNativeAgent` 每
  100ms 运行；继承生产 adapter 的播放队列与控制信号处理。
- 输入队列与推理串行线程独立于 tick；到窗口末收集已完成结果，未完成不阻塞
  下一个 tick。音频不会在结果返回前进入队列。
- 原生块宽仍为 MiniCPM 1000ms、Freeze-Omni 160ms、Moshi 80ms。
  从连续输入无损切出完整原生块，按块结束时间提交；不把100ms片段强行当作
  一个模型原生调用。对不整除100ms的模型，完整块可用性会受传输粒度影响。
- 不新增 force_listen 或 set_break，不改变模型的自主控制策略。
- 仍使用生产 worker 返回边界。MiniCPM 的 finalize 后返回与直连 generate
  后即就绪的差异被保留，不能把100ms量化误差当作整个接入链路误差上限。
- 这只是实验 adapter，未修改默认生产配置；Moshi、Freeze 正式 provider
  注册及其共享参考运行时与原服务器一致性限制仍未闭合。

不是“每次阻塞推理后再 sleep 100ms”，而是“提交输入后，在独立的100ms
窗口内接收结果；无论推理是否完成，窗口结束后继续”。慢于实时的计算会形成
可观察的输入积压，而不会被冻结时钟隐藏。

## 复测方法

复用40秒英文固定音轨及原来的模型配置，同模型同一张 A100 上 A/B/B/A，
分别新进程初始化、固定seed。A 原生直连按输入原生块的结束时间送入，B 采用
100ms异步接入。Moshi使用已核验的原服务器预热方法。每步输入/输出hash、
原始文本/状态、真实返回时刻、实际编排播放音轨全部保留。

SALMONN此前四次均在第194块触发共同TTS异常，本轮不重复该失败样本，
也不将它计为通过。

本地测试：`test_open_realtime_replay.py` 三项（阻塞原生推理不阻塞tick、
实际step输入分块、调用间80ms开销不逐tick累计），加原冻结路径四项，
共七项通过。时钟累积测试先失败：5tick实际824ms；修复后通过。

证据目录：`data/simulations/control-signals/20261112-open-realtime-100ms/`。
独立冻结 snapshot 保留提交时的代码；旧 frozen 实验没有覆盖。

| 模型 | Merlin SID |
|---|---|
| MiniCPM | b61ca37df390e5f3 |
| Freeze-Omni | c908c315af690c36 |
| Moshi | 616a291164a9bad8 |

复用已授权 LQ cluster17/group301 队列
`compute-108-lq-cloudnative-ai-speech.vi-guarantee`，每任务1×A100_SXM_80GB，
镜像VID `d4p5eh3c77ube906ov60`，限时1小时。预检全部success；若error停止，
warn/hint需展示；保留式合并request_context.precheck_sid，30分钟内提交。

## 实测结果

### 第一版：每tick相对等待100ms

MiniCPM A0/B1首次开口5700/5420ms，A3/B2为5440/5320ms，分别提前
280/120ms；各40块输入和原始输出完全相同。整段单一有声区间起止边界的
最大差为800/250ms。A0/A3自身首次开口也相差260ms，最大边界差750ms。
不能把所有差异都归因于编排，也不能据此放宽用户的100ms判据。

Moshi两组首次开口300/370ms，6段语音区间一致，最大边界误差70ms，
各500块原始输出相同。

Freeze A0/B1首次开口5860/5370ms，提前490ms，各250块原始输出一致，
活动片段数25/29不一致，不能验收。

第一版暴露了额外的时间压缩：MiniCPM第346tick时，墙钟相对记录时间已多走
约5804ms。每个tick至少等待100ms，再叠加调用间编排/文件记录开销，会累积
漂移。不能把“run_tick每次等100ms”等同于“会话每100ms推进一次”。

### 第二版：按会话起点的绝对截止时刻

`realtime_adapter.py`按起点+n×100ms收集结果，扣除已花费的调用间开销。
为避免公共模板再次追加相对sleep，实验子类关闭模板的sleep分支（内部
`pacing="frozen"`仅抑制这次重复sleep）；实际运行仍由子类的墙钟截止时刻
推进，配置及orchestrator均为realtime，不等待推理完成。真实模型等待仍计时。

快照与证据独立保留于
`data/simulations/control-signals/20261112-open-realtime-100ms-absolute/`。

| 模型 | 第二版SID |
|---|---|
| MiniCPM | 6c325c5361c0b615 |
| Freeze-Omni | c06b5aef44881964 |
| Moshi | b3963f44b9e22f5a |

第二版12次实权重运行全部完成，无运行异常；均为真实step输出，没有事后
平移音轨。各模型四次的完整输入、原始PCM输出、文本和可观察控制状态逐块
完全相同（MiniCPM各40步、Freeze各250步、Moshi各500步）。

| 模型 | A0 / B1首次开口ms | A3 / B2首次开口ms | B−A两组 | 全段边界结果 |
|---|---:|---:|---:|---|
| MiniCPM | 5650 / 5520 | 5420 / 5620 | −130 / +200ms | 第一组活动区间1/3段；第二组最大差200ms |
| Freeze-Omni | 5690 / 5270 | 5240 / 5270 | −420 / +30ms | 两组活动区间25/30段，不能判全部边界通过 |
| Moshi | 300 / 370 | 300 / 370 | +70 / +70ms | 都为6段，最大边界差130ms，仍超100ms |

正值表示编排录音更晚开口。检测使用10ms RMS窗口、合并小于200ms间隙；
活动区间数变化说明播放间隙发生变化，不能解释为模型生成了不同语义内容。
Moshi第一版最大差70ms、绝对时钟版130ms，说明首次开口或一个版本的一段
音轨通过，不足以证明每个起止边界都通过。

400个输入tick内，六次B运行的累计墙钟/记录时间漂移为25.0–27.8ms，
第一版MiniCPM的秒级累计漂移已消除。此数值由末tick与首tick实际交付间隔
减去相应100ms音频间隔得到，和模型响应延迟是不同指标。

### 剩余差异的证据与解释范围

1. 原生重复自身有差异：MiniCPM A0/A3首次开口相差230ms；Freeze相差
   450ms，完整边界最大差770ms；Moshi首次开口相同、最大边界差10ms。
   这是同任务同GPU固定seed实测，不能以“GPU波动应该很小”替代数据。
2. Freeze第21个输入步含首次TTS，A0耗时1962ms，B1/B2为1481/1468ms，
   A3为1504ms。它解释了第一组开口差的主要部分；不是100ms时钟还能凭空
   把模型计算压缩掉。不能把所有剩余差异都归于网络或tick宽度。
3. MiniCPM A0的40步中7步超过原生1000ms，最长1171ms；B1最长1245ms。
   B侧中后段可出现队列供给不足/额外空隙。精确的每步耗时保留在
   `timing-diagnostics.json`及`native.jsonl`，不把慢推理期间静音删除。
4. 100ms边界收集、原生块宽、输出FIFO及接入/记录开销仍可能叠加，
   “100ms tick”不是“整条接入误差必然≤100ms”的保证。同步记录会影响运行
   开销，本实验保留了这些开销，未扣除或事后补偿模型输出时刻。
5. Freeze直连控制器处理`controller_stop_tts`，实验adapter仍受既有控制信号
   映射限制；其整个播放轨差异不能全部归于时钟。该绑定与原服务器并发模型
   的独立一致性也未验收。MiniCPM保留finalize后返回，未增加强制监听。
6. 此实验使用固定ReplayUser，未证明动态LLM/TTS用户模拟器的生成等待也能
   持续提供正确的音频/静音；生产实时接入需要另行处理这一层。

### 结论

用户提出的异步执行方向成立：发送与模型推理解耦，按独立墙钟窗口接收，
确实避免原冻结路径忽略计算等待所造成的主要1–2秒提前误差。必须按绝对
时间窗口推进，不能每步推理完成后等待100ms，也不能把循环开销逐步叠加。

**本轮仍未达到所有语音边界≤100ms的验收条件。** 原始模型内容保持相同，
但播放时间与间隙还不完全相同。默认生产OpenModelAdapter未切换；新增的是
可复现的实验异步适配器、回归测试和证据。SALMONN仍受之前共同TTS错误阻塞，
本轮不声称其通过。旧实验及两版新快照全部保留。

汇总命令：

```bash
.venv/bin/python scripts/control_signal_probes/summarize_open_equivalence.py \
  data/simulations/control-signals/20261112-open-realtime-100ms-absolute \
  --scheduled-mode realtime
```
