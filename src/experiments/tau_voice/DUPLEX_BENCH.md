# 动态双工事件 bench（实验版）

输入是一组事件类型。默认使用项目里的 `gpt-6.1-terra` 自动生成 Topic，并在实际对话过程中决定何时说话、执行哪个待测事件、具体说什么。没有预写事件台词、固定事件顺序或“说够多少字就打断”的事件时间表。

```powershell
uv run python -m experiments.tau_voice.duplex_bench run --events interruption.follow_up,rejection.talking_to_others,proactive.clarification
```

英文样本使用 `--language English`。例如，对 gpt-realtime 各触发一次否定、话题切换和用户附和，并显式启用已播放音频 ASR：

```powershell
uv run python -m experiments.tau_voice.duplex_bench run --events interruption.denial,interruption.topic_switch,rejection.backchannel --language English --model gpt-realtime --assistant-observation played_audio
```

这里的 `rejection.backchannel` 是用户对助手的简短附和，预期助手保留话轮继续讲。每个所选事件最多执行一次；具体时机、台词和顺序由 Terra 在运行时决定。

去掉 `proactive` 后共有 15 类任务，选 5 类共有 `C(15,5)=3003` 种组合。`turn_taking.handoff` 包含跨话轮记忆：Terra 先用 `prepare_event` 自然说出具体事实或约束，经过其他交流后，再询问依赖这些信息的问题，提问时不重复待记忆值。提问可发生在助手正在说话或安静时，不设置提问前的静默门槛，也不评估提问是否打断助手。提问后保护完整回答，期间禁止后续用户发言或事件；观察到回答后，至少连续 `response_quiet_seconds`（默认 1 秒）没有可听或携带话语 ID 的助手音频，才推断回答结束。超时保留为未完成观察。没有建立前情或没有间隔交流，不能当成有效记忆测试。

记忆评审使用独立的 `duplex_memory_judge` 调用，逐项比较先前事实、后来更正与完整回答，输出 `memory_assessment.facts`，记忆判定直接作为事件判定。提问与先前助手语音的重叠不参与评分，不生成 `handoff_assessment`。回答是否逐字复述前情不是标准；评审检查实际建议是否遵循相关约束，并区分违背、未证明和无关。预算、用时等不能仅凭助手自称满足就判为已证明。

`prior_user_turns` 是用户计划台词，不保证 TTS 全部读出。若提供了 `prior_user_spoken_evidence`（用户原录音的回溯 ASR），记忆评审优先使用它；标准运行尚不自动进行用户侧 TTS 的 ASR 完整性核验。私有 `user_context` 不能证明助手听过事实。新运行会在 `synthesized_audio/` 和 `synthesis.json` 保留原始 TTS 音频、输入文本及重采样前后的时长与哈希，以便排查漏读。

`rejection.contextual` 仍计为一类，运行时可由 Terra 选择背景第三者说话（`contextual_speaker: "background"`），或主用户对旁人的话（`"primary"`）。两者都必须符合之前实际说出的不回复条件。

结束录音时，仍携带话语 ID 的音频帧（包括低能量帧）会阻止结束；完成所有观察后，至少连续一秒没有助手音频交付，才允许 Terra 选择结束。此保护用于避免单个低能量帧截断录音，不把能量阈值解释为真实语义停顿。

结果自动保存到 `data/simulations/duplex_<时间>/`。默认被测语音模型沿用 `config.py` 的 OpenAI 模型配置；用户决策、Topic 生成和语义评审默认使用 Terra，TTS 使用 OpenAI TTS。使用项目现有 `.env` 中的凭证。可用 `--model` 改被测模型，`--user-model` 改用户决策模型，`--provider` 切换已有语音提供方。

## 运行逻辑

1. Terra 根据所选事件生成 Topic 和私有用户背景，不生成事件剧本。
2. 模拟器接收被测助手的实时音频活动与文本片段。
3. 在可决策时，Terra 同时获得当前对话、助手是否正在发声、未覆盖事件和已建立的上下文条件，选择下一步。
4. 模拟器执行 Terra 的动作：等待、正常交谈、铺垫上下文，或立即触发某个事件。所有说出的内容均在这次决策中生成，再交给 TTS。
5. 记录实际播放、音频重叠和助手响应。完成本事件的观察窗口后，再让 Terra 选择下一步。输入事件的排列不构成执行顺序。

执行层只约束物理条件和事件归属。例如用户插话必须发生在助手发声期间；背景第三者使用另一种声线，“用户对旁人说话”使用主用户声线；噪声由 Terra 选择种类，再生成实际非语音音频。没有合适机会时 Terra 可以等待或继续正常交流，不会强制制造不合上下文的台词。

对于 contextual rejection / contextual proactive interruption，Terra 先自行选择时机、生成自然语言建立条件，再在后续对话中决定何时触发。Proactive 四类只生成用户侧的机会，助手的补全、附和、澄清和插话是否实际发生，由被测模型决定。

`decision_interval_seconds` 默认 1 秒，是调用决策模型的检查间隔，不是事件触发时间。检查结果可以一直为等待。播放用户语音和观察事件期间不会重复调用决策模型。Terra/TTS 调用仍是同步的，模型调用耗时会影响实时运行；事件日志保留调用耗时，实际重叠以音频轨迹为准。

Terra 的助手侧对话文本默认来自模拟器**已经收到并播放的助手音频**：每到决策点，对当前语音片段前缀做一次 ASR，并以新识别假设替换旧假设。不会把 gpt-realtime 尚未发出/播放的 transcript 或未播放音频交给 Terra。ASR 是分段重识别基线，不是流式词级识别/强制对齐；会增加决策墙钟延迟，且识别仍可能有误。原始音频前缀保存在 `heard_audio/`，识别时间、文本、错误和延迟记录在 `perception.json`。API 或 ASR 失败时，本次不会静默退回可能错位的比例文本。可以用 scenario 的 `assistant_observation: "proportional"` 显式关闭这一行为；这种模式下的助手台词仅供近似上下文参考，缺词不能用来断言助手漏答。

## 支持的事件类型

运行 `uv run python -m experiments.tau_voice.duplex_bench list` 查看完整目录。

| 类别 | 事件 ID 后缀（需要加类别前缀） |
|---|---|
| `interruption` | `follow_up`, `denial`, `repeat_request`, `topic_switch`, `correction`, `impatience`, `shut_up`, `clarification` |
| `rejection` | `backchannel`, `pause`, `third_party`, `talking_to_others`, `noise`, `contextual` |
| `turn_taking` | `handoff` |
| `proactive` | `completion`, `backchannel`, `clarification`, `contextual_interruption` |

这里 `rejection.backchannel` 指用户附和时助手继续当前话轮，不把附和当成新的请求；`proactive.backchannel` 指用户说话期间助手给出简短附和。`rejection.third_party` 是背景第三者对别人说话；`rejection.talking_to_others` 是主用户对旁人说话；`rejection.contextual` 是遵从之前的上下文指令不回答某人。

## 可选的配置文件

也可以先保存事件选择和运行参数，准备操作不调用任何 API：

```powershell
uv run python -m experiments.tau_voice.duplex_bench prepare --events interruption.correction,rejection.contextual,proactive.contextual_interruption --output data/simulations/duplex_policy.json
uv run python -m experiments.tau_voice.duplex_bench run --scenario data/simulations/duplex_policy.json
```

配置文件只有 `selected_events`、模型、语言等参数，以及可选的 Topic / 用户背景，没有事件内容或时间线。Topic 为空时在运行时由 Terra 自动生成。固定配置和 seed 不保证模型生成相同的对话；每次实际决策均单独保存。

## 结果与评测

- `scenario.json`：所选类型、实际生成的 Topic、用户背景和运行参数。
- `events.json`：Terra 每次决策及理由、是否接受、实际触发的内容、顺序、播放时间和助手观察记录。
- `simulation.json` / `ticks.json`：完整对话与逐 tick 轨迹；失败运行尽量保留部分轨迹。
- `audio/`：双方音频及对齐信息。`user_labels.txt` 只标主用户，`third_party_labels.txt` 单独标背景第三者，`assistant_labels.txt` 标助手。第三者仍在输入音频中供模型辨别；标签按语句 ID 聚合已播放片段的文字，时间边界依据音频活动，文字使用时长比例估算，不是逐词精确对齐。
- `assessment.json`：逐事件的时序证据、语义评审和覆盖统计。

评审附带本次已保存的 ASR 识别证据。普通事件仅使用原观察窗口内的识别；记忆任务另外用 `memory_response_review` 标出延长至下一次用户发言或结束决策的回答审查范围，保留原始时序指标和观察边界。延长审查属于事后评测，不会把未来音频提供给运行中的 Terra。
- `llm_calls/`：每次话题生成、实时决策和评审的实际请求及原始返回文本，包括 prompt、模型参数、usage、费用和耗时；同类调用全部保留。此目录只在模型调用成功返回时写入，连接失败而没有返回的请求不在其中。早期未启用日志的运行无法补回原始请求/返回。

待测事件只有在实际语音已合成并开始播放后才被记录为触发；仅仅提出一个决策不算覆盖。未触发、运行中断、缺乏真实重叠或观察未结束的事件不会算作模型通过，也不会当成模型失败。

静默测试可以直接依据完整音频活动窗口判断。助手在背景语音前就已经说话时，继续原来的回答不能直接判为错误回复；这种情况交给语义评审。插话后“又说了话”也不等于正确回答，需要检查具体内容。模型评审会先检查用户实际生成的刺激是否符合该事件；无效刺激或证据不足保留 `inconclusive`。这仍是实验性的评测，需要结合音频抽查。

需要只保存时序证据时使用 `--no-semantic-judge`。已有结果可重新评测：

```powershell
uv run python -m experiments.tau_voice.duplex_bench assess --output data/simulations/你的运行目录 --judge-model gpt-6.1-terra
```

语音观察以 tick 为粒度，助手输入相对用户输出有一 tick 的传递延迟。事件报告保留这一说明，不能把自然停顿直接当成由用户造成的取消。第三者刺激当前在主用户没有播放语音时发出，支持与助手语音重叠；不包含第三者与主用户同时讲话的混音测试。

离线验证（不调用外部模型或 TTS）：

```powershell
uv run pytest tests/test_streaming/test_duplex_scenarios.py -q
```
