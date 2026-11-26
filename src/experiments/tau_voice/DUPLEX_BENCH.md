# 动态双工事件 bench（实验版）

## 合并后的语音与声学配置

默认调度 tick 为 100 ms，用户新决策机会默认仍间隔 1 秒；开源模型保持独立的 100 ms 异步入口和原生音频块缓存。

用户 TTS 可用 `--tts-provider volcengine|openai|elevenlabs` 显式选择。普通双工事件默认保留 Volcengine Seed TTS 2.0 及原主用户/背景音色；声学任务默认使用 OpenAI；`--byted-gateway` 使用 ElevenLabs、Scribe 转写及远端原有网关配置。不同提供方的 model/resource ID 和 voice ID 需要匹配；ElevenLabs 的背景人物事件需显式配置另一个可用 voice ID。`run_config.json` 会保存实际 TTS 提供方。

Volcengine 的停顿和 CTC 仍在单次 SSML 请求中插入 750 ms 停顿；显式选择其他 TTS 时使用分段合成与 PCM 静音，并在刺激播放前测量波形中的停顿。两条合成路径不可作为完全相同的输入条件混用比较。

声学任务可独立运行，也可与事件组合。没有普通事件的独立声学任务不自动加入附和或 CTC；已有语义事件默认仍补充这两个机会，并支持 `--no-proactive-opportunities` 关闭。组合场景中的用户可在非记忆事件观察期间回答澄清追问；记忆回答仍保护至播放完成，助手的简短附和不会截断用户长陈述。

ASR 既支持场景的 `transcription_model`（含 Gemini），也支持注入完整 `TranscriptionConfig`；显式完整配置优先。网关继续使用 Scribe，普通路径使用场景模型。关键词标注、实际声学扰动和最终评审证据保持隔离。

## 三模型同顺序、同剧情比较

场景模板 `scenarios/picnic_matched_three_models.json` 用于 Grok、Gemini Live、Doubao Seeduplex。`event_order` 必须是全部覆盖目标（含附和、CTC）的完整排列；控制器拒绝越序触发。`conversation_outline` 给出共同主题阶段，`event_guidance` 约束各事件的语义目标，台词和插入时机仍由用户 LLM 根据实际已播放音频决定。`prepare_event` 可提前建立记忆事实，不推进事件顺序。背景问话固定为向 Jamie 询问门口钥匙；CTC 使用此前共享的 sesame 过敏事实，不能只凭私有计划判断刺激有效。

此模板显式使用 `gemini/gemini-3.8-flash` 作为用户模拟器和综合 Judge、`gemini-3.8-flash` 作为音频转写，解决当前 OpenAI HTTP 接口鉴权失败的依赖。Gemini 转写只接收实际已播放音频，不接收剧情或预期补全；可能修订部分转写，仍需结合原音频审查。转写凭证支持 `GEMINI_API_KEY` 或 `GEMINI_LIVE_API_KEY`；LiteLLM 文本调用使用 `GEMINI_API_KEY`，启动进程时可从现有 Live Key 映射，不要把凭证写入模板或报告。用户 TTS 仍为豆包 Seed TTS 2.0，两个停顿事件仍为单次 SSML `<break time="750ms"/>`。

```powershell
uv run python -m experiments.tau_voice.duplex_bench run --scenario src/experiments/tau_voice/scenarios/picnic_matched_three_models.json --provider xai --judge-model gemini/gemini-3.8-flash --output data/simulations/matched_grok
```

分别替换 provider 为 `gemini`、`doubao` 并使用新目录，建议串行运行。三者共用输入条件，但不能保证助手回答、总话轮和事件时刻相同。与历史 Terra/Whisper 批次比较时要注明模拟器、转写和评审均已变化；每模型一个样本用于案例检查，不能推断总体排名。Judge 与 Gemini 被测模型来自同一提供商，也应结合音频人工核查其判断。

## 统一评审与停顿校准

每个事件由一次 LM 评审结合回答内容、说话时机和音频活动证据，输出一个 `event_judgment`，其 `verdict` 和 `reason` 即最终结论。规则计算出的时序指标只作为 `timing_evidence` 输入，不再与 LM 结果分别计分。记忆事件沿用逐事实检查，输出同一格式的事件结论；请求与上一轮助手的重叠本身不影响记忆判定。未成功调用 Judge 的事件保留为 `inconclusive`。

句内停顿与 CTC 都以豆包单次 SSML 的 `<break time="750ms"/>` 合成。750ms 是请求值，实际波形需测量：`pause_audio_evidence` 保存 10ms 帧 RMS 检测出的唯一匹配静默区间。`marker_seconds` 应指向这个区间起点，不能使用包含前缀、停顿、后续内容的整句音频末尾；无法唯一定位时不播放该刺激，事件保留待重试。该方法是声学检测，不是逐词强制对齐。

历史比较曾混入 OpenAI TTS 的 1.5 秒分段拼接样本，而 Grok/Gemini 已使用豆包 750ms SSML，不能将这两种输入条件直接比较。已保存的 SSML 音频可用下列命令测量和校正时间戳，再重新评审；原始 `events.json` 首次修改前保存在 `events.before_pause_alignment.json`，录音与历史 LM 日志保持原样。

```powershell
uv run python scripts/audit_duplex_pauses.py data/simulations/你的样本目录 --apply
uv run python -m experiments.tau_voice.duplex_bench assess --output data/simulations/你的样本目录 --judge-model gpt-6.1-terra
```

## 豆包 Seeduplex 实时语音与本地对比

豆包测试入口为 `--provider doubao --model 1.2.6.1`，采用 Seeduplex 3.0 的原生 JSON WebSocket 全双工协议，而非角色助手接口。环境变量保存在本地 `.env`：`DOUBAO_REALTIME_API_KEY`；若使用旧式 Access Key，还需 `DOUBAO_APP_ID`，适配器自动使用 AppID/Access-Key 请求头；新式 API Key 不设置 AppID。不要在源码、报告或日志中写入实际密钥。

协议依据：[全双工文档](https://docs.volcengine.com/docs/doubao-voice/end-to-end-real-time-voice-full-duplex-version?lang=zh)。输入 16 kHz PCM、输出 24 kHz PCM，通过共享转换器接入框架的 8 kHz μ-law 通路。音频按 20 ms 发送；模拟器生成等待时发送 mute，恢复时 unmute。本实验适配器仅支持无工具的音频对话。

当前统一条件的四模型本地比较批次为 `data/simulations/voice_compare_20261103_003/`。以 `manifest.json` 明确选择样本，使用 `python scripts/render_voice_comparison.py data/simulations/voice_compare_20261103_003` 重建中文 HTML。音轨、事件、ASR、评审与调用证据保留在各样本目录中；生成目录被 Git 忽略。`voice_compare_20261103_002/comparison.html` 也已更新为相同样本集。

运行先保存录音和轨迹，再调用语义评审；也可用 `--no-semantic-judge` 先录制，之后单独 `assess`。网络断连产生的部分录音保留为诊断，未观察完的事件不算模型失败。模拟时间在同步 LLM/TTS/ASR 调用时暂停，远端语音服务仍运行，不能据此比较真实端到端延迟。GPT-Live 的连续静音保留时间位置，但不携带话语 ID，避免记忆回答永不结束。所有剩余事件均要求助手安静时，助手说话期间直接倾听，避免反复调用 LLM 得到相同的等待决策。
当前评估范围与 Topic 覆盖解释见 [HTML 报告](../../../docs/duplex-evaluation-report.html)。
报告已接入同一主入口，可以离线重新生成；不会启动对话或付费评审：

```bash
python -m experiments.tau_voice.duplex_bench report --output docs/duplex-evaluation-report.html
```

需要统计已有运行时，对每个最终有效清单中的目录显式追加 `--run-dir PATH`。
默认只做源码审计，不扫描历史／重试目录，也不将交接文档的历史结果当成本次实跑统计。
同时输出 `.evidence.json`，记录源码哈希、事件目录、明确纳入的运行和 Topic 字符串去重口径。

输入是一组事件类型。默认使用项目里的 `gpt-6.1-terra` 自动生成 Topic，并在实际对话过程中决定何时说话、执行哪个待测事件、具体说什么。没有预写事件台词、固定事件顺序或“说够多少字就打断”的事件时间表。

用户模拟器的打断决策必须以已听到的具体内容为依据：先听清相关建议、主张或表达，再在自然的分句边界插入；仅检测到助手正在出声、事件待执行或顺序轮到该事件，都不足以触发打断。私有 `reason` 必须引用已听内容，并解释为什么此刻需要打断；证据不足时选择 `wait`。普通衔接和记忆请求应等待当前回答结束，背景第三者及噪声仍可独立重叠。这是运行时 prompt 约束，控制器不新增固定等待秒数或语义硬门槛；实际自然度仍需新录音验证，历史结果保持原样。

对比页模板位于 `scripts/templates/voice_comparison.html`，使用明色布局。transcript 支持说话人筛选、搜索、点击时间跳转、播放高亮及重叠标记；助手文本可切换音轨标签与去重后的已播放音频 ASR。ASR 是检查点前缀识别，可能误识别或缺少后续尾句，不能当作逐词对齐。可在批次根目录放置 `review_notes.json`，按模型目录名和事件 ID 记录评审争议；页面会标出待复核项，保留原始 LM 分数。

```powershell
uv run python -m experiments.tau_voice.duplex_bench run --events interruption.follow_up,rejection.talking_to_others,proactive.clarification
```

英文样本使用 `--language English`。例如，对 gpt-realtime 各触发一次否定、话题切换和用户附和，并显式启用已播放音频 ASR：

```powershell
uv run python -m experiments.tau_voice.duplex_bench run --events interruption.denial,interruption.topic_switch,rejection.backchannel --language English --model gpt-realtime --assistant-observation played_audio
```

这里的 `rejection.backchannel` 是用户对助手的简短附和，预期助手保留话轮继续讲。每个所选事件最多执行一次；具体时机、台词和顺序由 Terra 在运行时决定。

## 原选定事件之外的附和与 CTC 机会

新建 `prepare` 配置或直接 `run --events` 时，默认保留原来的事件选择，并额外加入 `proactive.backchannel` 和 `proactive.completion` 各一次。例如从原 15 类中选 5 类，仍然保存为 5 个 `selected_events`，实际覆盖目标为 5 + 2 个；显式选过的 proactive 类型不会重复添加。Terra 根据实际对话选择穿插位置和台词，不固定放在开头、末尾或某个事件之后。

- **助手附和**：用户讲述一段与当前话题相关、保持话轮的较长经历或约束，给助手说“嗯哼 / mm-hm”之类简短回应的机会。默认要求合成音频至少有 8 秒可听活动（`backchannel_min_speech_seconds`），不能靠补静音满足长度；过短的合成不会播放，也不计覆盖，Terra 会收到反馈重选表达。用户不会因为助手的简短附和而截断自己的发言。
- **CTC 补全**：用户在已经有明确语义线索的句子中途卡壳，`text` 只含说出的前缀，`pause_seconds` 统一为 750 ms，`continuation` 必须为空。停顿通过豆包 TTS 单次 SSML 请求中的 `<break time="750ms"/>` 插入，随后观察窗口继续保持倾听。私有 `expected_completion` 保存预期缺失短语，只供评审参考，不进入 TTS 或被测助手策略；同义补全也可通过。

CTC 与 `rejection.pause` 的区别在于语义：前者具备可预测的补全，后者是不能据此推断缺失内容的思考停顿，预期助手等待用户继续。两者由 Terra 按实际共享上下文设计，并由语义 Judge 检查刺激是否有效；不会把普通附和误算成补全。用户模拟器统一使用豆包 TTS，CTC 和 `rejection.pause` 都在同一次请求中用 `<break time="750ms"/>` 插入停顿，不把停顿拆成后处理静音。

CTC 要求缺失含义由已说出的事实或固定表达基本确定，允许同义措辞；仅仅符合话题不足以构成有效样本。例如，只提到野餐，再说 “head to the…” 并不能确定是 park，也可能是 beach 或 lake。应先自然明确目的地，如用户说 “Let's make Riverside Park our destination.”，助手确认 “Riverside Park it is.”，随后用户说 “Great, after lunch we'll take the bus straight to…” 并停顿，此时补全 “Riverside Park” 才有明确依据。示例不固定写入运行剧本。模拟器在私有 reason 中引用依据；评审同时获得先前双方对话，不把私有预期答案当作共享知识。若仍有多个含义不同的自然补全，该刺激判为无效（inconclusive），不能因猜中预设词而通过，也不能因助手等待而判失败。

评估单独记录 `opportunity_window`：附和要求发生在用户长段发言期间；CTC 检查卡壳之后、下一次用户动作之前的观察窗口。时序命中还需语义评审确认是简短附和或相关补全。`summary.selected` 是原选定数量，`supplemental` 是额外数量，`total` 是合计目标，`selected_observed` 和 `supplemental_observed` 分别记录完成观察的数量。记忆回答的保护期间不会插入这两类机会。

示例（5 个原事件 + 2 个额外机会）：

```powershell
uv run python -m experiments.tau_voice.duplex_bench run --events interruption.follow_up,interruption.denial,rejection.pause,rejection.third_party,turn_taking.handoff --language English --model gpt-realtime
```

使用 `--no-proactive-opportunities` 可关闭额外机会。读取缺少 `include_proactive_opportunities` 字段的旧配置时，默认保留原有覆盖范围，以免重评历史录音时多出未运行事件；新运行若复用旧配置，增加 `--proactive-opportunities` 可启用。更新展示或重评不会自动重跑原对话。

去掉 `proactive` 后共有 15 类任务，选 5 类共有 `C(15,5)=3003` 种组合。`turn_taking.handoff` 包含跨话轮记忆：Terra 先用 `prepare_event` 自然说出具体事实或约束，经过其他交流后，再询问依赖这些信息的问题，提问时不重复待记忆值。提问可发生在助手正在说话或安静时，不设置提问前的静默门槛，也不评估提问是否打断助手。提问后保护完整回答，期间禁止后续用户发言或事件；观察到回答后，至少连续 `response_quiet_seconds`（默认 1 秒）没有可听或携带话语 ID 的助手音频，才推断回答结束。超时保留为未完成观察。没有建立前情或没有间隔交流，不能当成有效记忆测试。

记忆评审使用独立的 `duplex_memory_judge` 调用，逐项比较先前事实、后来更正与完整回答，输出 `memory_assessment.facts`，记忆判定直接作为事件判定。提问与先前助手语音的重叠不参与评分，不生成 `handoff_assessment`。回答是否逐字复述前情不是标准；评审检查实际建议是否遵循相关约束，并区分违背、未证明和无关。预算、用时等不能仅凭助手自称满足就判为已证明。

`prior_user_turns` 是用户计划台词，不保证 TTS 全部读出。若提供了 `prior_user_spoken_evidence`（用户原录音的回溯 ASR），记忆评审优先使用它；标准运行尚不自动进行用户侧 TTS 的 ASR 完整性核验。私有 `user_context` 不能证明助手听过事实。新运行会在 `synthesized_audio/` 和 `synthesis.json` 保留原始 TTS 音频、输入文本及重采样前后的时长与哈希，以便排查漏读。

`rejection.contextual` 仍计为一类，运行时可由 Terra 选择背景第三者说话（`contextual_speaker: "background"`），或主用户对旁人的话（`"primary"`）。两者都必须符合之前实际说出的不回复条件。

结束录音时，仍携带话语 ID 的音频帧（包括低能量帧）会阻止结束；完成所有观察后，至少连续一秒没有助手音频交付，才允许 Terra 选择结束。此保护用于避免单个低能量帧截断录音，不把能量阈值解释为真实语义停顿。

结果自动保存到 `data/simulations/duplex_<时间>/`。默认被测语音模型沿用 `config.py` 的 OpenAI 模型配置；用户决策、Topic 生成和语义评审默认使用 Terra，用户模拟器 TTS 统一使用豆包 Seeduplex TTS。使用项目现有 `.env` 中的凭证。可用 `--model` 改被测模型，`--user-model` 改用户决策模型，`--provider` 切换已有语音提供方。

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
