# User 音频扰动与 clarification 数据生成：设计讨论

日期：2026-11-02。

状态：核心流程、第 1、2 轮决策、ElevenLabs 对齐与对话结束后评估已确认；本文记录需求、代码接入依据及待实现的局部改动。尚未实现或进行性能测量。

收敛后的实施规格见 [clarification-user-audio-spec.md](clarification-user-audio-spec.md)，
已发布为 [GitHub Issue #7](https://github.com/hhzzff/tau2-voice-turn-taking/issues/7)，
标记 `ready-for-agent`。用户已确认测试边界。

用户确认四项拆分后，已发布并核对以下子 Issue，均标记 `ready-for-agent`：

| 子任务 | 原生阻塞关系 |
| --- | --- |
| [#8：完整 TTS 关键词 hard dropchunk 接入](https://github.com/hhzzff/tau2-voice-turn-taking/issues/8) | 无，可先实施 |
| [#9：完整 TTS 关键词 hard DNS noise 接入](https://github.com/hhzzff/tau2-voice-turn-taking/issues/9) | #8 |
| [#10：完整澄清评估与离线重评](https://github.com/hhzzff/tau2-voice-turn-taking/issues/10) | #8 |
| [#11：真实调用、可复现数据与扰动耗时验收](https://github.com/hhzzff/tau2-voice-turn-taking/issues/11) | #9、#10 |

子任务包含后续讨论确认的要求：复用现有话轮机制、由 agent 发起追问，
两种追问时机分别统计但采用相同语义标准。父 Issue 正文和状态保持原样。

## 用户提出的目标

将 `/mnt/bn/audio-visual-llm-data6/terumi/natural_conversation_data/clarification`
中针对 user 输入加噪声的流程引入本仓库，支持添加噪声或 dropchunk，
用于生成 user simulator 的 clarification task 数据。

用户要求加噪声或 dropchunk 的处理耗时尽可能小；这不是逐 chunk 执行扰动的要求。

## 已确认的设计边界（第 1 轮答复）

### 处理顺序与性能目标

1. 复用现有 LM 动作决策与文本生成调用，根据当前动作和 context 同时返回完整回复文本与关键词。
2. 复用现有完整 utterance 的 TTS 生成。
3. 使用 ElevenLabs `POST https://api.elevenlabs.io/v1/forced-alignment`
   将完整 TTS 与原始回复文本对齐，定位 LM 同时生成的关键词，
   再在目标片段施加噪声或 dropchunk。
4. 处理完毕后，复用现有切片与发送机制，按 tick 交付受损音频。

不新增一次独立的关键词选择 LM 调用，不重写文本生成、TTS 或 tick 发送流程。

不采用每个 tick / chunk 单独决定并执行扰动的方案。
本需求优先优化加噪或 dropchunk 的处理耗时。
关键词在现有 LM 调用中一并生成；音频对齐与实际加噪或 dropchunk 的处理耗时分别列出；
本轮明确优先优化的是后者，同时保留 TTS 完成到受损音频可交付的总耗时，
避免用纯计算时间替代全部等待时间。

### 信息可见范围与交互

- User simulator 知道自己本来想表达的内容，但不知道哪一句、哪个词被破坏，
  也不接收噪声类型、位置、难度或预定澄清结果等提示。
- LM 生成关键词是对自身回复的语义标注，不是对已发生扰动的知情。
  实际是否执行扰动、使用哪种方式及修改哪些采样，由处理层记录；
  不将这些结果注入 user simulator 的后续 context。
- Agent 接收实际受损音频，并自主决定是否请求澄清。
- User simulator 根据实际收到的 agent 打断或追问决定是否回复，
  不能因处理层已注入扰动而提前补充或解释。
- 评估器保留原始文本、目标字词、对应时间范围和实际扰动记录，
  用来判断 agent 的追问及后续交互是否合理。
- 只采用原始 hard 难度；不生成 mild / medium 档。
- 每个 clarification task 只扰动一条目标 utterance；追问后的重复、解释或补充回复
  使用正常 TTS，不再次破坏。
- Agent 在目标 utterance 结束前打断追问，或在结束后追问，都可以是有效澄清。
- 对话结束后统一评估完整交互，运行过程中只记录实际交互与音频证据，
  不逐回复进行 pass / fail 判断，也不将评估结果回流给 user simulator。

这里只确认交互与信息边界，不预设每次扰动都一定导致澄清。

## 文档用途

本文件保存这个主题的讨论过程与设计结论。已确认的领域术语写入根目录
`GLOSSARY.md`；符合项目标准的重要决策再单独记录到 `docs/adr/`。
本文件本身不表示已经接受某个架构决策。

## 当前已核查的约束

- 项目区分 simulation tick、simulation time 与 tick computation time。
  因此音频扰动的额外 wall-clock 处理耗时应单独衡量，不能从模拟音频时长推算。
  参见 `../GLOSSARY.md` 和 `adr/0001-frozen-simulation-time-for-open-model-mvp.md`。
- 当前工作区存在其他未提交修改；本次讨论先新增本文，保留已有修改。
- 本轮仅只读核查代码，未运行生成任务、外部模型调用或耗时基准测试。

## 源流程核查

以下源文件路径相对于
`/mnt/bn/audio-visual-llm-data6/terumi/natural_conversation_data/clarification/`。

- `clarification/noising/dns_noise.py:79` 的 `mix_noise_in_span`：
  在已知时间片段内混入噪声，采用 NumPy，保持音频时长。
- `clarification/noising/dropchunk.py:36` 的 `apply_dropchunk_in_span`：
  使用 Torch / SpeechBrain 将选定片段中的部分采样替换为静音，保持时长，
  不是从时间轴删除样本。当前接口面向完整波形，且使用全局随机种子。
- `clarification/pipeline/demo_pipeline.py:460-620` 的完整离线流程：
  ASR、关键词选择、对齐、扰动、生成澄清与修复对话、可选 TTS。
  其中澄清行为由生成策略安排，不能当作 agent 实际听音后的行为证据。
- `scripts/CORRUPTED_QUESTIONS.md` 描述更小的 question-only 流程：
  用已有问题文本选择关键词并强制对齐，然后执行 CPU 扰动，无需重新做 ASR 或 TTS。
- `clarification/alignment/mms_fa.py:30` 提供已知文本强制对齐，但仍需完整音频
  和对齐模型；当前实现仅支持英语，CJK 返回不可用。
- `clarification/pipeline/corrupted_question_pipeline.py:1000-1022`
  对缺失或不合理对齐拒绝样本；完整离线流程另有固定位置回退策略。
  引入本仓库时必须明确选用哪种失败语义。
- 当前 question-only 的 `_keyword_selector_prompt` 已偏向 word-searching / hmm 场景，
  且提示词允许 1–4 词，校验器只允许 1–2 词，不能直接照搬。
  通用关键词提示词位于 `clarification/llm/prompts.py:1-46`。
- 当前 question-only 校验要求关键词唯一出现；底层对齐匹配返回首个同名片段，
  未提供 occurrence index。重复关键词需要明确选择策略，不能静默取第一次。
- `clarification/config.py:2` 的 DNS 默认素材目录为
  `/mnt/bn/audio-visual-llm-data3/datasets/DNS/datasets_fullband/noise_fullband`。

## 本仓库核查

- `src/tau2/user/user_simulator_streaming.py:1300` 的 `_apply_chunk_effects`
  已提供 user 音频逐 chunk 处理链：噪声混音、可选电话编码、frame drop；
  记录 effect timeline，并支持 clean / noise / user output 等音轨采集。
  这是已有能力的参考；用户已明确本次扰动在完整 TTS 上完成，
  不将此逐 chunk 入口作为本次扰动的实施位置。
- `src/tau2/voice/synthesis/audio_effects/noise_generator.py` 中，背景噪声可在初始化时
  加载并按 chunk 从内存切片；burst 噪声当前仍可能在触发时读盘和重采样。
- `src/tau2/voice/synthesis/audio_effects/processor.py` 的 frame drop 保持 chunk 长度。
  当前一次处理最多覆盖当前 chunk；若指定时长跨多个 chunk，需要显式维护剩余范围。
  timeline 上声明的范围还需要与实际被破坏的采样核对。
- 现有 `proactive.clarification` 是语义歧义场景，不能直接代表听不清引发的澄清。
- `src/experiments/tau_voice/duplex_bench.py:70` 的 `build_dynamic_user`
  为保证事件归因，显式关闭随机噪声和 frame drop。新任务需要有目标的扰动计划，
  不能仅将全局随机开关打开后宣称完成。
- `src/tau2/user/duplex_scenario_user.py:472` 保存原始 TTS、脚本和哈希，
  但脚本不等于已核验的实际转录。评估时需要将干净意图与 agent 实际收到的
  受损音频分别作为证据，避免让干净脚本掩盖音频中的信息缺失。
- `tests/test_voice/test_audio_effects.py` 已覆盖混音、frame drop、随机种子与 timeline
  等基础机制；尚未发现声学扰动触发澄清的行为验收或延迟基准。

## 速度方面的初步判断（未测量）

单纯混音和静音替换的计算与关键词选择、音频对齐是不同阶段。
关键词已在原有 LM 回复中生成，音频时间定位仍需要对齐；
不能仅以混音计算快就宣称首次音频输出也快。
噪声预加载、预选有效素材和完整波形上的局部向量运算是候选优化，
尚未确认毫秒级指标。

## 已选定的音频定位接口

采用 ElevenLabs 的
[`POST /v1/forced-alignment`](https://elevenlabs.io/docs/api-reference/forced-alignment/create)。
已于本轮核对官方接口文档，未提交实际音频或调用付费接口。

用户已确认凭据在本仓库 `.env` 中，变量名为 `ELEVENLABS_API`。
新增对齐调用从这个变量读取密钥，不要求用户另配 `ELEVENLABS_API_KEY`；
文档与评估产物不保存密钥值。

- 使用 multipart/form-data 提交 `file`（干净、完整的 TTS 音频）和
  `text`（这段 TTS 对应的原始回复文本）。
- 响应包含字符级 `characters` 与词级 `words`，两者均有 `text`、`start`、`end`；
  响应还包含 `loss`，词级条目也包含 `loss`。
- 根据 LM 标注的关键词及其在回复中的具体出现位置匹配对应字符或词，
  再映射为完整波形中的采样范围。保持原音频时长，之后继续使用现有 tick 切片。
- 保存原始对齐响应及实际使用的范围，供评估复查。对齐失败或匹配不可靠时
  明确记录失败，不随机选一段冒充关键词位置。
- 对齐请求耗时与本地 hard 扰动处理耗时分别记录；复用网络连接和预加载噪声素材
  是实现时可采用的优化。

本方案不依赖源目录英文限定的 MMS 对齐入口；不同语言的实际对齐质量仍需验证，
不因接口提供字符级结果就宣称已完成中文验证。

## 讨论收敛状态

当前无阻塞实施的设计问题。后续按上述已确认流程完成接入与验证：

- 同一次现有 LM 调用生成回复及关键词标注。
- 完整 TTS 上进行 ElevenLabs 对齐与原始 hard 档扰动，随后复用 tick 发送。
- 私有受损记录供评估使用，不进入 user simulator 的后续 context。
- 支持真实追问与修复交互，对话结束后统一评估完整轨迹。
- 记录对齐和扰动的独立耗时；验证长度不变、目标范围及未修改部分、
  实际发送音频、信息隔离及澄清后的修复行为。

本节表示需求讨论已收敛，不表示代码已实现、接口已实测或行为验收已通过。

## 最高难度的源实现事实

- DNS noise 的原始 `hard` 档为目标片段 SNR = -20 dB
  （`clarification/pipeline/demo_pipeline.py:558-560`）。这表示相对信号的噪声强度，
  不是把整句音量调到最大；最终输出还有 clipping。
- 原始 `hard` dropchunk 在目标关键词范围里选两个静音块，
  每个块的目标长度为该范围的 65%–90%，二者可以重叠
  （`clarification/noising/dropchunk.py:79-83`）。
  它不等于确定性地将整个关键词 100% 置零。
- 实际掩盖范围、clipping 和听觉上的不可辨识性不能仅由 `hard` 名称保证。
  评估器应记录实际修改的采样范围，不把目标字词范围直接当作逐字可听性结论。

## 第 2 轮结论及用户修正

### Q3：沿用原始 hard 定义（已确认）

DNS noise 采用 -20 dB；dropchunk 沿用原始 hard 的静音块规则，
不额外改变成关键词全覆盖。

### Q4：关键词随现有回复生成（用户修正后确认）

现有 LM 已根据当前动作与 context 生成回复，直接扩展其结构化输出，
在同一次调用中返回回复文本与关键词。音频处理层只负责定位和扰动，
不再调用独立关键词选择器。使用已知 TTS 文本对齐音频，不重新做 ASR。
对齐和扰动记录仅交给执行层与评估器。

### Q5：扰动次数与修复回复（已确认）

每个 clarification task 只选一条目标 utterance 执行扰动；
追问后的重复、解释或补充回复使用正常 TTS，不再次破坏。

### Q6：澄清发生的时机（已确认）

Agent 在目标 utterance 结束前打断追问、或在结束后再追问，
都可以是有效的澄清；不以必须发生重叠打断为唯一合格条件。

## 直接复用现有实现的接入依据

- `src/tau2/user/duplex_scenario_user.py` 的 `_decide`：
  当前动作、pending events 和实际对话 context 已在一次 LM 调用中产生
  `DuplexDecision`。关键词应扩展在这次输出里。
- `src/tau2/user/duplex_events.py` 的 `DuplexDecision`：
  已有 `action`、`event_id`、`text`、`continuation` 等字段和 Pydantic 校验，
  可增加关键词语义标注并校验其与实际输出文本一致。
- `DuplexUserSimulator._generate_full_duplex_voice_message`：
  已调用 `_speech` 得到完整 TTS 波形，再返回包含完整音频的 `UserMessage`。
  应在完整 TTS 就绪、消息交给切片器之前插入关键词对齐与 hard 扰动。
- `VoiceStreamingUserSimulator._perform_turn_taking_action`：
  已调用 `_create_chunk_messages(full_message)`，再放入 `output_streaming_queue`。
  这一切片和 tick 输出机制继续复用。

## 核查后确需处理的局部问题

### 让 agent 的真实追问能够进入现有交互链

`DuplexUserSimulator._next_turn_taking_action` 目前在输出队列未空时
继续 `keep_talking`，只对 `proactive.contextual_interruption` 有提前停止分支；
事件 active 时也会等待观察窗口结束再让 LM 决策。
直接新增音频扰动不会自动得到“agent 打断追问后 user 判断是否回复”的闭环。

通用 `VoiceStreamingUserSimulator` 已调用 `basic_turn_taking_policy`，
具备被 agent 打断时返回 `stop_talking` 的策略；`_perform_turn_taking_action`
已在该动作下清空未播放队列，常规回复继续由原有 LM/TTS 生成。
这些是本次应复用的基础能力，不是需要重新开发的 clarification 功能。

Agent 是澄清追问的发起方：它可以在 user 发言中追问，也可以等 user 说完后追问。
User simulator 只是基于实际请求和原意决定回复，不承担主动澄清受损信息的任务。

接入时检查 duplex 封装覆盖通用话轮策略及 active observation 阻塞的局部位置，
仅做复用已有能力所必需的调整；不新增专用澄清识别器或整套交互状态机，
不依据私有受损标记强制回复。保留其他事件原有观察保护。

这里的运行时观察用于收集实际语音和管理话轮，**不等于运行时评分**。
当前 `src/experiments/tau_voice/duplex_bench.py:233` 先执行 `orchestrator.run()`，
然后在 `finally` 的 252–254 行才调用 `assess_report` / `judge_report`；
不是每收到一条 agent 回复就调用评估器。

本需求继续采用对话结束后统一评估，澄清评估读取目标话语、实际发送的受损片段、
agent 追问、user 实际修复及后续回应，覆盖完整交互。
固定 observation window 到期不能代替澄清闭环完成，也不能让运行时评分阻塞回复。
当前 runner 在异常退出时也会走 `finally`；这类轨迹需保留中断或不完整标记，
不能当作完整对话成功结果。

### 私有标注不回流到 user simulator

`_decide` 当前将 pending `EventSpec` 的完整内容（包含 `expected_behavior`）发给 LM。
不能把具体受损字词、方式或难度加在这里再发回给 user simulator。
后续 context 还包含 executed event IDs 和 execution feedback，也需检查是否暴露扰动结果。

`DuplexController.history` 当前只记录显式列出的对话字段；保留这个边界。
LM 输出的关键词和实际扰动记录可以放在私有执行状态及评估 artifact 中，
不加入后续 `recent_conversation`，也不把扰动异常作为“你刚才的话受损”反馈给 LM。

### 关键词时间定位与失败语义

本仓库当前 TTS 返回的 `AudioData` 不含词级时间戳，已有字符分配到 chunk 的逻辑
是均匀分配，不是强制对齐。因此 LM 同时返回关键词可以省掉一次选择调用，
但不能省掉准确定位关键词实际发音的步骤。不能用文本长度比例冒充词级音频定位。

现有 `DuplexScenario.language` 同时支持 Chinese / English，默认 Chinese。
已按用户选择采用 ElevenLabs forced alignment，接入时保留现有语言配置，
并对未能可靠定位的样本明确标记失败，
不能随机破坏其他位置后仍标成指定关键词受损。

重复关键词也需明确到文本中的具体出现位置；实际评估记录应关联该位置、
对应音频时间范围及最终实际修改的采样，而不是仅保存一个可能重复的词字符串。

## 两类 agent 澄清的评估细化（已确认）

本节回应用户对 agent 中途打断澄清与 user 说完后澄清的评估问题，
已由用户确认；两者不是要求 user simulator 主动执行的动作。

### 构造有效性、时机与语义分开

1. 先核查目标片段实际被扰动并交付，而非只存在于尚未播放的完整 TTS 文件。
   目标未交付或轨迹中断时明确记录，不归为有效受损输入上的澄清成功或失败。
   hard 参数证明执行了指定处理，不能直接证明人或模型完全无法辨认。
2. 在对话结束后，由实际 agent 音频的追问内容识别澄清请求，
   不能把首次发声、附和或普通回答当成澄清。
3. 根据该追问的实际音频起点与目标 user 话语的时间线分类：
   user 尚持有话轮时发生的是中途澄清，自然说完后发生的是说完后澄清。
   另记录实际语音重叠；句内静音不必然等于 user 已结束。
4. 保留目标 user 话语原定自然结束位置与实际停止位置。
   若 user 因 agent 打断而提前取消余音，不用取消后的结束位置反过来把该追问算成说完后澄清。
   追问具体起点必须与已确认的请求语句关联；只有 tick 级证据时标明分辨率，
   在边界附近不能可靠分类则保留不确定类别，不以 ASR 完成或收到网络事件的时刻代替发声时间。

### 两种时机使用相同的语义标准

- 追问是否为合理澄清：在该追问发生之前实际交付的音频与已有 context 下，
  是否确有需要补充或确认的信息，是否是在追问缺失内容而非无依据地假设答案。
- 追问是否相关：允许语义等价请求，例如出发城市受损时问“从哪里出发”，
  不要求 agent 说出它没有听到的目标词。
  泛化的“请再说一遍”是否足够合理取决于缺失范围，不机械判为通过或失败。
- 修复后是否恢复理解：根据 user 实际提供的清晰回复，检查 agent 后续是否正确使用该信息。
  请求本身合理与最终修复完成分别记录；user 未回复、音频缺失或对话中断不能一概算作 agent 追问失败。
- 判断追问合理性时只提供截至该时刻已交付的 context；未来还未播放的话语不能倒灌为 agent 已知信息。
  判断后续修复时才提供之后的真实交互。

### 结果与时延

分别记录输入构造/交付有效性、是否发起澄清、时机类别、追问合理性/针对性、
修复完成及后续信息使用情况。未追问时还需区分基于已有信息合理回答与无依据猜测，
不强制把每个 hard 样本都判成必须追问。

时延作为独立描述，不能用更早追问自动换成更高语义分数。
中途澄清报告追问与已交付受损范围的时间关系及实际重叠；说完后澄清可报告
追问起点减去 user 自然发言结束时刻。两类结果分别汇总，并明确有效样本分母、
未追问、边界不确定、无效输入与不完整轨迹，不仅给一个总成功率。
