# 双工语音测试 bench 交接

更新日期：2026-10-30。工作区：`D:\code\tau2-voice-turn-taking`。

## 目标与当前状态

输入双工事件类型的子集，由项目内的 `gpt-6.1-terra` 生成话题，并在实际对话中决定事件顺序、触发时机和具体台词；被测语音模型接收自然对话与音频。不要把固定台词、固定顺序、时间表或评审标准注入被测模型。

已实现动态用户模拟器、事件目录、已播放音频 ASR 观察、时序与语义评审、独立第三者音轨和标签，以及跨轮记忆回答保护。默认测试入口是 `experiments.tau_voice.duplex_bench`；原有第三者专项实验仍可单独运行。

当前 GitHub 同步范围为 Git 管理的源码、测试与文档，以及本次新增文件。`data/simulations/` 和 `artifacts/` 按既有 `.gitignore` 保留为本地产物，六组录音、HTML、完整调用日志及批次报告生成脚本不随源码提交。已有 `.env.example` 删除属于本轮之前的工作区改动；本次同步保留该状态。实际凭证由本地环境或未跟踪的 `.env` 提供，不写入文档或提交。

## 用户确认的事件定义

不含 proactive 时共 15 类，可从 `C(15,5)=3003` 个组合中抽取 5 类；每个被选事件最多执行一次，执行是否有效另行判定。

| 类别 | 事件 ID | 含义 |
| --- | --- | --- |
| interruption | `follow_up`, `denial`, `repeat_request`, `topic_switch`, `correction`, `impatience`, `shut_up`, `clarification` | 用户主动打断正在说话的助手：追问、否定、重复请求、话题切换、纠正、不耐烦、停止说话、澄清 |
| rejection | `backchannel`, `pause`, `third_party`, `talking_to_others`, `noise`, `contextual` | 用户附和、句中停顿、背景第三者、用户对旁人说话、噪声、根据上下文不回复 |
| turn_taking | `handoff` | 跨话轮记忆：回答时遵循先前事实与最新更正 |
| proactive（另 4 类） | `completion`, `backchannel`, `clarification`, `contextual_interruption` | 助手补全未完句、附和、完整但含糊请求后的澄清、按先前约定主动插话 |

表中 ID 需加类别前缀，例如 `interruption.follow_up`。

`rejection.third_party` 指背景第三者说话，不一定与主用户交谈；`rejection.talking_to_others` 指主用户对其他人说话。`rejection.contextual` 指根据先前实际说出的条件不回答某人或某类旁话，Terra 运行时选择背景或主用户声线。`rejection.backchannel` 是用户附和，预期助手保持当前话轮；`proactive.backchannel` 才是助手附和。Proactive 的助手行为由被测模型产生，模拟器只创造机会。

## 记忆任务的最新定义（后续修改必须遵循）

1. Terra 先用 `prepare_event` 自然说出具体事实或约束，经过至少一次间隔交流，再问依赖前情的问题。提问不能重述待测试的事实值，私有 `user_context` 不算助手听过的证据。
2. **不检查记忆提问是否打断助手。** 目录中 `turn_taking.handoff.trigger` 为 `any`；没有提问前的静默门槛。不要恢复 quiet handoff 校验，也不要因提问重叠把结果降为 `inconclusive`。
3. **只在记忆回答期间禁止用户打断。** 问题发出后，禁止后续用户发言或事件；固定观察窗口结束不能直接释放。先观察请求后的回答，再等待可听音频及携带话语 ID 的音频均结束后连续 1 秒静默。默认超时 120 秒，超时属于观察未完成，不能当作成功。
4. 独立 `duplex_memory_judge` 逐项比较先前事实、最新更正和实际完整回答，输出 `memory_assessment.facts`，包含原话、时间、回答引用、遵循/违背/未证明/无关及原因。
5. `memory_assessment.verdict` 直接作为事件结果；不再生成 `handoff_assessment`。原始音频重叠指标可保留作记录，不参与记忆评分。
6. 回答无需逐字复述前情；具体建议遵循约束即可。实质冲突应失败；不能凭助手自称满足就证明预算、精确用时等数值。事实缺失或语义前提无效时才使用证据不足。

## 主要代码与 prompt 入口

| 文件 | 职责 / 排查入口 |
| --- | --- |
| `src/tau2/user/duplex_events.py` | `EVENT_CATALOG`、事件触发条件与预期行为、场景配置、话题生成、Terra 参数适配 |
| `src/tau2/user/duplex_scenario_user.py` | `RUNTIME_POLICY` 是运行时用户指令；`DuplexController` 管理事件归属与观察保护；`_decide()` 组装 Terra 输入；`_speech()` 保存原始 TTS 证据 |
| `src/tau2/user/duplex_perception.py` | `PlayedAudioPerception`，仅识别实际交付音频前缀；同一片段的新假设替换旧假设 |
| `src/tau2/user/duplex_evaluation.py` | `assess_report()` 时序评审、`judge_report()` 语义评审、`MEMORY_JUDGE_POLICY` 与 `judge_memory_event()` 独立逐事实记忆评审 |
| `src/experiments/tau_voice/duplex_bench.py` | `list / prepare / run / assess` CLI；普通对话环境与产物保存 |
| `src/tau2/user/third_party_simulator.py` | 第三者专项模拟、双事件调度与主用户纠正 |
| `src/experiments/tau_voice/third_party_interference.py` | 第三者专项实验入口 |
| `src/tau2/voice/utils/audio_activity.py` | 依据实际音频判断可听活动 |
| `src/tau2/voice/synthesis/conversation_builder.py` | 保留安静帧里的文本、按话语组织标签、分离第三者标签，取消 200 字截断 |
| `src/tau2/voice/audio_native/openai/live_discrete_time_adapter.py` | 工作区已有的 GPT-Live 实验适配器；8 kHz μ-law，仅音频、无工具；与 gpt-realtime 路径分开 |
| `src/tau2/config.py` / `src/tau2/registry.py` | 默认模型、记忆保护参数，以及 `duplex_user_simulator` / `third_party_voice_user` 注册 |

其他现有更新包括空用户 LLM 回复的静默处理，以及从自由对话助手策略中移除测试脚本与期望行为。详细操作说明见 [DUPLEX_BENCH.md](src/experiments/tau_voice/DUPLEX_BENCH.md) 和 [THIRD_PARTY_INTERFERENCE.md](src/experiments/tau_voice/THIRD_PARTY_INTERFERENCE.md)。

## 运行与验证

在已有 Python 3.12 环境中可用以下命令；首次安装参照根目录 `AGENTS.md` 的 voice、dev extras。运行真实对话或指定 `--judge-model` 会调用远程 API。

```powershell
# 仅列事件，不调用模型
.venv/Scripts/python.exe -m experiments.tau_voice.duplex_bench list

# 真实英文样本；只指定类型，台词与时机由 Terra 在对话中决定
.venv/Scripts/python.exe -m experiments.tau_voice.duplex_bench run --events interruption.impatience,rejection.third_party,interruption.follow_up --language English --model gpt-realtime --assistant-observation played_audio --output data/simulations/duplex_new_sample

# 重评已有运行；使用新规则时应备份旧 assessment，并保留原调用日志
.venv/Scripts/python.exe -m experiments.tau_voice.duplex_bench assess --output data/simulations/duplex_new_sample --judge-model gpt-6.1-terra

# 本次新增功能的离线回归：58 项通过
.venv/Scripts/python.exe -m pytest tests/test_streaming/test_duplex_scenarios.py tests/test_streaming/test_duplex_perception.py tests/test_streaming/test_third_party_simulator.py tests/test_voice/test_audio_activity_labels.py -q

# 提交前要求：make check-all；当前 Windows 没有 make，等价执行
.venv/Scripts/ruff.exe check .
.venv/Scripts/ruff.exe format .
```

测试已有 `audioop` 弃用、pytest asyncio 配置两条环境警告。离线测试不证明所有提供方在线可用；涉及提供方协议时仍需验证对应 API 与 provider suite。

本次同步前还执行了 `tests/test_voice/test_audio_native/run_provider_suite.py`。在线结果为 OpenAI **10 通过、1 失败**，其他提供方因缺少相应凭证或启用开关共 **66 跳过**。失败项 `test_barge_in_baseline` 两次均出现单个 tick 超过 300 ms 的耗时阈值，分别为约 444 ms、323 ms；连接、多轮、工具调用及其余打断测试通过。这项实时耗时问题尚未解决，不能宣称在线套件全绿。最新明细已保存至 `tests/test_voice/test_audio_native/provider_suite_results.txt`。

套件脚本另有 Windows 覆盖已有结果文件时的 `Path.rename()` 错误，已改为 UTF-8 写入并使用 `Path.replace()`；复查确认结果文件正常更新。执行全仓库 Ruff 格式化时也产生了少量既有文件的纯格式调整。

## 六组真实英文对话（本地批次）

目录：`data/simulations/duplex_random_15c5_20261030/`。`manifest.json` 是最终清单来源，保留：`group_01`、`group_02_retry`、`group_03_new`、`group_04_new`、`group_05_new`、`group_06_new`。每组 5 项，合计 30 项；29 项实际观察完成，1 项 correction 未形成真实重叠。最新结果：**20 通过、8 未通过、1 证据不足、1 未评估**。

| 含记忆任务的样本 | 结果 | 关键证据 |
| --- | --- | --- |
| 2 | 未通过 | 记住 chickpeas→lentils 更正，但为芝麻过敏者推荐 tahini |
| 4 | 通过 | 无芝麻、素食、可重复使用容器、冰袋、更正后的 9 升冷藏箱 |
| 5 | 通过 | 5 人、纯素、不用烤箱、避开蘑菇与豆腐 |

部分预算、人数、用时等事实仍标记为 `not_demonstrated`，通过不代表所有数值均已证明。

主要本地产物：

- `report.html`：可独立打开的总报告，包含六组样本、内嵌音轨、事件、ASR 与 Terra 调用；`index.html` 和各组 `sample.html` 为分页面版本。
- 各组 `scenario.json`、`events.json`、`assessment.json`、`simulation.json`、`perception.json`、`llm_calls/`、`audio/`：配置、执行、评审、轨迹、观察、完整模型输入输出及录音。
- `memory_user_audio.json` 与 `memory_user_audio/*.wav`：原用户录音的回溯 ASR；优先于计划台词作为已说出事实的证据。
- `assessment_before_memory_fact_review.json`、`assessment_before_memory_task_definition.json` 等：历次评审备份；`memory_fact_review.json`、`memory_task_definition_update.json` 记录当前判定及规则变更。最新规则更新复用已有语义结果，没有重跑对话或付费评审。
- `build_reports.py` / `report_ui.py`、`validate_final.py`、`check_ui.py`、`check_single_report.py`：当前批次专用报告生成与校验脚本；仅存在本地忽略目录。重建命令为 `.venv/Scripts/python.exe -X utf8 data/simulations/duplex_random_15c5_20261030/build_reports.py`。

旧 `group_03`、`group_03_retry` 已从最终清单、统计与报告排除。历史物理删除被自动审批拒绝，原目录仍留在本地；不要据目录枚举重新纳入统计。`group_02` 首次运行仅保留为诊断样本，不计入六组。

## 已查明问题与边界

- Realtime transcript 的接收时间不等于词在音轨中的播放时间。原比例分配会让 Terra 误以为助手没说过某词；默认观察已切换到实际播放音频 ASR。ASR 是同步前缀重识别，仍可能误识别并增加墙钟延迟；不是词级强制对齐，也不是流式 ASR。
- Audacity labels 不再因低能量帧漏词，并区分主用户与背景第三者，但文本标签不能证明 TTS 实际逐词读全。
- 新样本 3 在约 42.2 秒的计划台词含 “Is there another option?”，原录音实际止于 “fabric”。4.85 秒、25 个块全部发出，未发生发送队列截断；两次重合成都完整。当时原始 TTS 响应未保存，无法进一步区分合成漏读或响应获取异常。证据见 `group_03_new/user_audio_diagnosis/`。新运行已在 `synthesized_audio/` 和 `synthesis.json` 保存原始 TTS、输入文本、时长与哈希；尚未自动校验用户侧每句 TTS 的 ASR 完整性。
- 记忆保护和结束录音均使用音频及话语 ID 后的持续静默推断，不应将单个低能量帧解释为语义停顿。不要把提问前静默门槛重新加回记忆任务。
- 历史 Terra 日志保持原样，可能含旧的安静交接要求；当前代码、当前 `assessment.json` 和本文定义优先，不能改写原日志伪造历史输入。

## 后续工作的入口

优先把批次专用报告生成器整理为仓库内通用模块，以便新 checkout 无需本地忽略目录即可生成相同结构的单文件报告。其他可改进项是用户 TTS 完整性检查、异步流式 ASR 与更可靠的回答完成信号。除非用户要求重新生成，不要为更新展示或聚合而重跑真实对话；先复用原音轨、ASR 和逐项评审。
