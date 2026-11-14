# 固定音频直接发送与 100ms 调度原型对照

> 历史探索记录：用户已指出闭源跨请求比较混入网络延迟，且此处没有运行实际
> 生产编排器。本报告不再作为公平性验收依据。最新开源实测见
> [四模型实际编排路径对照](open-model-equivalence-20261112.md)。保留原始失败证据。

日期：2026-11-12。**本轮不能判定通过 100ms 等价性验收。**

## 测试对象和限制

实测 MiniCPM-o-4.5、GPT Realtime 2.1、Gemini 3.1 Flash Live Preview、
豆包 S2S 1.2.6.1。GPT Live 沿用此前约定排除。本轮没有重测 SALMONN、
Freeze-Omni、Moshi。

这里测试的是独立的 **100ms 调度原型**，尚未经过生产
`FullDuplexOrchestrator.step()`、动态 user simulator 或生产 adapter。
输出音轨由真实接收时间和取消事件作因果回放生成，未做浏览器实际试听或
物理扬声器测量。因此即使原型通过，也不能直接证明现有 τ-voice 编排器通过。

## 固定输入和比较方式

新生成的英文音轨共 40 秒：

- 0–3.4126875s：“Hello. Please count from one to one hundred, slowly.”
- 10–15.20025s：“Stop counting. What is seven plus five? Answer with just the number.”
- 其余为静音。两条路径使用完全相同的 PCM，不根据模型回答修改输入。

使用算术问题避免时效知识和联网能力影响内容判断。固定音轨的“完整”指完整
对话输入；按真实时间逐包发送，不在开头一次上传未来 40 秒音频。

每个模型按直接 A → 调度 B → 调度 B → 直接 A 顺序运行，独立会话。
MiniCPM 每次新建进程，同一张 A100、同一 checkpoint/config、seed=42，
在模型准备之后再次重置随机数。直接路径每满 1 秒调用原始 SDK；调度路径
每 100ms 积累音频，满 1 秒交给独立串行推理线程。两条路径都在 generate
完成时记录音频可用时间，随后 finalize；原型不冻结或抹去推理耗时。
两边播放启动缓存统一为 0ms，均不包含原前端默认 200ms 启动缓存。

MiniCPM 两边使用相同的 pi-bench 外部打断策略（已知用户音轨区间触发
set_break/force_listen），**不能把停止说话解释为模型自主打断**。

闭源主对照两边均按 20ms 原始网络包节奏发送，B 将输出/取消事件的可见时间
向上取整到下一 100ms 边界。另保留一组 B 每 100ms 集中发送五个 20ms 包的
探索结果；不与主对照混合。每条路径都有完整原始接收事件和连续音频。
这两组均没有模拟生产闭源编排器的停发等待策略。

## 闭源主对照结果

下表是 B 相对配对 A 的差值，单位 ms，配对顺序为 A0/B1、A3/B2。
开口由输出 PCM 的 10ms RMS 窗口检测，阈值 100；短于 200ms 的静音合并。
第二次开口须在第二句话结束后，另以文本确认回答“12”。

| 模型 | 首次开口差 | 第二次开口差 | 内容 / 结论 |
|---|---:|---:|---|
| 豆包 S2S | -30 / +30 | +20 / +160 | 四次均回答 12；存在 >100ms 差异 |
| Gemini Live | -20 / +120 | -210 / +20 | 四次均回答 12；计数措辞/进度不同；存在 >100ms 差异 |
| GPT Realtime | +1280 / -360 | 无法计算 | 四次在插话后均未返回新回答；不能验收 |

直接路径 A0/A3 自身的首次/第二次开口绝对差：豆包 60/140ms，Gemini
10/200ms，GPT 740ms/缺失。这说明跨请求波动不能忽略；不能据此认定所有
差异来自调度，也不能据此豁免 100ms 门槛。样本数每模型每路径只有 2 次，
不构成稳定性或统计等价性证明。

同一模型四次的输入 PCM hash、2000 个发送包的 hash 序列完全一致，
主对照发送期限的最大迟到为 9.4ms。GPT 因重采样少 1 个采样点，各路径一致。
同一份接收轨迹采用即时播放与 100ms 播放回放时，B 首次开口差为 10–50ms；
这只验证回放量化没有增加整秒等待，不等于独立模型运行的延迟等价。

### GPT 缺失第二次回答

主对照四次均收到两次 `speech_started`，但只有第一次 `speech_stopped`、
`committed`、`response.created`，没有完成第二次响应。完整静音尾段仍发送到
40 秒，观察到 41 秒；没有报告 provider error。探索组仅一次直接路径收到
第二个回答，其余直接和调度路径也缺失。因此不能把该现象归因于 100ms
集中发包或模型能力；当前 gateway/protocol 路径尚未验证完整。

## MiniCPM 结果

| 运行 | 首次开口（音轨绝对 ms） | 插话后的新回答 |
|---|---:|---|
| A0 直接 | 5560 | 未观察到 |
| B1 调度 | 5360 | 未观察到 |
| B2 调度 | 5460 | 未观察到 |
| A3 直接 | 5330 | 未观察到 |

四次各 40 步，输入 hash、输出 PCM hash、文本、is_listen、end_of_turn、
force_listen **逐步完全一致**。均输出计数 1–9，插话后持续监听直至输入结束。
外部强制监听在用户结束后的首个静音块解除，后续仍未回答；这属于两条路径
共有的行为，尚不能确定来自参考控制策略还是模型，不能解释为自主打断成功。

B−A 首次开口差为 -200/+130ms；取消旧播放的执行时间差为约 -139.4/+99.7ms。
直接路径自身首次开口相差 230ms，原始音频就绪时间逐步最大相差 412.8ms。
所以“同硬件的计算波动可以忽略”没有被本次数据支持。原始内容一致是正面
证据，但尚未达到逐事件 100ms 门槛，也没有覆盖成功恢复回答的情况。

## 本地回归检查

`test_equivalence_timeline.py` 两项通过：1401ms 才生成的音频在 100ms
调度下只能从 1500ms 开始，不能回填输入起点；取消事件切掉旧残留和迟到包，
同时保留新响应。完整测试调用最终通过；另一次带 45 秒超时的重复调用在
加载公共 conftest 时超时，独立 `--noconftest` 调用也为两项通过。
这些测试验证回放逻辑，不验证模型语义。所新增实验脚本的 Ruff 检查通过。

## 可复现证据

- 固定输入、初次闭源探索、MiniCPM：
  `data/simulations/control-signals/20261112-equivalence/`
- 闭源同节奏主对照：
  `data/simulations/control-signals/20261112-equivalence-paced/`
- 各目录 `analysis.json` 保留逐次结果、配对差值、发送序列 hash、部分文本。
  `model/run-mode/played.wav` 为被测播放音轨；`arrival-playback.wav` 为同份
  接收轨迹的即时播放；`events.json`、`scheduler-events.jsonl` 或 `native.jsonl`
  为原始时序证据。取消事件切断旧队列，正常完成保留尾音。
- MiniCPM Merlin job：`e9895a0db8f9377f`；`jobs/` 保留提交、状态和四次日志，
  `gpu.csv` 保留硬件身份；`source/`、`minicpm-config.json`、`SHA256SUMS`
  保留冻结的实验版本和输入校验。
- `scripts/control_signal_probes/equivalence.py`：闭源采集和因果回放。
- `scripts/control_signal_probes/equivalence_minicpm.py`：原始 MiniCPM SDK 对照。
- `scripts/control_signal_probes/summarize_equivalence.py`：仅重算证据汇总，不发请求。

生产编排器的验收仍须使用公开 step/adapter 路径重复相同实验，不能把本次
原型证据标记为 `tau_adapter_live_validated=True`。
