# MiniCPM 原生 1000 ms 步骤计时

## 实测结论

在本次单张 A100-SXM4-80GB 上，MiniCPM-o-4.5 原始 SDK 通常能在
1000 ms 内完成当前输入块并返回完整音频，但不能保证所有步骤均满足。
同步 `finalize` 通常额外消耗约 40 ms。当前 SDK 不暴露单次调用内部
的音频流式回调；不能把“1000 ms 输入块”解释为“必定额外等待 1 秒
才返回输出”，也不能把计算速度合格解释为编排器透明性已验收。

## 方法

- Merlin JobRun：`1ffd969f23977182`，已核验状态 `done`。
- 设备：NVIDIA A100-SXM4-80GB，driver 535.261.03；GPU UUID 保存在原始产物。
- 权重：`/mnt/bn/audio-visual-llm-data6/base_models/MiniCPM-o-4_5`。
- 直接调用原始 `UnifiedProcessor`，没有经过 τ worker 或 adapter。
- 逐步执行 `prefill → generate → finalize`，在每段边界同步 CUDA。
- 固定 45 秒音轨：旅行请求从 0 秒开始，纠正从 20 秒开始，backchannel
  从 32 秒开始。使用已有通用合成音频，没有外部 API 调用。
- seed=42。前两次不加外部打断控制，第三次镜像 pi-bench 的
  `set_break` 和强制监听策略；第三次实际在第 20 步触发 break。
- 模型只加载一次，各次重新 prepare 并设置随机种子。该方式可用于
  各阶段计时；不能直接作为独立新进程的确定性基线。
- 启动、权重加载、prepare 和文件写入不计入步骤耗时。首个推理步骤保留。
  输入音轨不按墙钟实时播放，因此这不是完整实时服务的体验测试。

## 结果

每组 45 步，共 135 步，其中按整块 PCM RMS > 100 判定有声的步骤为
63 步。该活动门限仅用于分组，不表示模型处于 LISTEN 状态。

| 所有步骤的墙钟指标 | P50 | P95 | 最大值 | 超过 1000 ms |
| --- | ---: | ---: | ---: | ---: |
| 音频可用：prefill + generate | 375.3 ms | 889.9 ms | 1000.6 ms | 1 / 135 |
| 同步整步：再加 finalize | 416.4 ms | 930.1 ms | 1048.8 ms | 3 / 135 |
| finalize 单段 | 40.3 ms | 44.2 ms | 62.2 ms | 0 / 135 |

总体分位数包含监听/静音步骤，不可替代有声步骤统计：

| 运行 | 有声步骤数 | 有声音频可用最大耗时 | 有声同步整步最大耗时 |
| --- | ---: | ---: | ---: |
| native_1 | 16 | 881.8 ms | 921.7 ms |
| native_2 | 27 | 983.2 ms | 1023.8 ms |
| reference_barge_in | 20 | 1000.6 ms | 1048.8 ms |

超时步骤：native_1 的 tick 10（1009.1 ms，整块活动未过门限）、
native_2 的 tick 43（1023.8 ms）、reference_barge_in 的 tick 37
（1048.8 ms）。没有删除超时点。

原生有声返回音频长度为 480–1160 ms。非 1000 ms 的四块均标为
`end_of_turn=True`：native_1 的 tick 21 / 32 分别 480 / 1160 ms，
native_2 的 tick 20 / 33 分别 880 / 480 ms。尤其 1160 ms 的自然结束块
存在 160 ms 尾音；它是否被保留或被错误取消，须由播放路径对照验证。

## 不可据此宣称的内容

- 这不是直接运行与完整编排器的 100 ms 时间等价性对照。
- 同进程重复两次，输入 hash 全部相同，但 26 / 45 个步骤的文本不同、
  27 / 45 个步骤的原始音频 hash 不同。不能把这两次当作确定性重复
  已通过，也不能仅凭该差异归因于 GPU 波动；需隔离准备、缓存与随机状态。
- 参考打断组使用外部策略，不能报告为模型自主打断。
- 本次有限音轨不能证明所有输入、所有 GPU 都满足 1000 ms 实时期限。

## 复现及证据

脚本：[minicpm_timing.py](../scripts/control_signal_probes/minicpm_timing.py)。
计时统计接口先经历 failing → passing 测试；Ruff lint/format 及
`git diff --check` 通过。额外 adapter 回归运行未完成并已中断，不计为通过。

原始产物（本地、不提交音频）：
`data/simulations/control-signals/20261112-minicpm-native-timing/`。
其中保存固定脚本及 SHA256、SDK 文件 SHA256、GPU 身份、完整配置、
预检与提交响应、结束状态、输入 PCM、逐步输出 PCM、JSONL 和 summary。

预检 `data.status=error` 时停止提交；本次为 success，将返回的
`precheck_sid` 合并到 request_context 后在 30 分钟内提交。
