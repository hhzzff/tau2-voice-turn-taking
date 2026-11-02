# 在完整 user TTS 上执行关键词 hard 扰动，生成声学澄清任务

## Problem Statement

需要在现有 τ-voice 对话生成中构造 Acoustic clarification task：user 正常表达完整意图，
其中一个关键字词的音频被强噪声或 dropchunk 破坏，agent 根据实际听到的信息决定是否追问，
user 再根据真实追问自然决定是否回复。已有离线 clarification 工程具备关键词音频扰动能力，
但不能整套搬入它的额外关键词选择、ASR、对话脚本生成及难度遍历流程。

当前仓库已经采用“基于动作与 context 的 LM 文本生成 → 完整 TTS → 切为 simulation ticks”的流程。
需求是复用该流程，在完整 TTS 就绪后、切片前插入最低可行开销的关键词扰动，
而不是重新设计流式 TTS 或在每个 chunk 单独施加扰动。

数据需要保留原始意图、实际损坏位置与真实交互证据，使评估器能够在对话结束后判断澄清行为。
User simulator 必须不知道自己的语音实际被破坏；仅有“已加噪”记录不能证明 agent 已完成澄清。

## Solution

沿用现有动作和 context 驱动的 LM 调用，同时返回自然回复与关键词语义标注。
沿用完整 TTS 生成，使用 ElevenLabs forced alignment 对齐干净音频与原文，
定位指定关键词的具体出现位置，再在内存中执行一种原始 hard 扰动：
DNS noise 的目标 SNR 为 −20 dB，或原始 hard dropchunk 静音块规则。
保持时长和采样数，然后使用现有切片、队列和 tick 交付机制发送给 agent。

每个任务只扰动一条目标话语，后续修复回复保持清晰。
Agent 可以中途打断追问，也可以等用户说完再追问；user simulator 根据实际对话决定回应。
实际扰动记录仅交给执行层及评估器。对话结束后，评估完整的目标话语、追问、修复及后续反应。
独立记录对齐耗时、本地扰动耗时及 TTS 完成后的总等待时间，优先优化本地扰动耗时。

## User Stories

1. As an 数据生成者, I want 复用现有动作和 context 驱动的 user simulator, so that 新任务仍然是真实动态对话。
2. As an 数据生成者, I want 原有 LM 调用同时返回回复与关键词, so that 不增加一次关键词选择模型调用。
3. As an 数据生成者, I want 关键词指向当前回复中的具体出现位置, so that 重复词不会被静默定位到错误位置。
4. As an 数据生成者, I want 使用现有完整 TTS 输出, so that 不需要更换 TTS 或重写生成链路。
5. As an 数据生成者, I want 使用 ElevenLabs 对齐干净 TTS 与已知原文, so that 无需为了定位关键词重新识别文本。
6. As an 数据生成者, I want 延续现有 Chinese / English 配置, so that 新任务不因旧对齐工具只支持英文而静默失效。
7. As an 数据生成者, I want 在完整音频切片前完成扰动, so that 跨 tick 的关键词破坏保持连续。
8. As an 数据生成者, I want 在目标关键词范围施加 hard DNS noise, so that 可以构造严重声学遮蔽。
9. As an 数据生成者, I want 使用原始 hard dropchunk 规则, so that 新任务与已有扰动定义一致。
10. As an 数据生成者, I want 仅生成 hard 难度, so that 不消耗资源产生本次不需要的 mild / medium 数据。
11. As an 数据生成者, I want 每个任务只破坏一条目标话语, so that 澄清原因可以归因到指定输入。
12. As an 数据生成者, I want 修复回复保持清晰, so that 对话有机会通过真实追问恢复缺失信息。
13. As an 数据生成者, I want 保持完整音频时长与非目标范围, so that 既有 simulation time 和 tick 调度继续成立。
14. As an 性能验证者, I want 单独测量本地扰动耗时, so that 能直接衡量本次要求优化的处理开销。
15. As an 性能验证者, I want 另外报告对齐请求与 TTS 后总等待耗时, so that 网络等待不会被混同为本地扰动计算。
16. As an user simulator, I want 保留自己的原始意图并只依据实际收到的 agent 对话回应, so that 我的行为不被私有损坏记录预先引导。
17. As an agent, I want 只收到实际交付的受损用户音频, so that 无法通过干净文本或私有元数据绕过声学任务。
18. As an user simulator, I want 能在 agent 真实追问时让出话轮并决定是否回应, so that 固定观察窗口不会阻塞自然修复。
19. As an 评估者, I want 中途追问和话语结束后的追问都保留证据, so that 不把时机形式当成语义完成。
20. As an 评估者, I want 知道目标字词、对齐范围和实际修改的采样范围, so that 能判断追问是否针对缺失信息。
21. As an 评估者, I want 对照原始与实际发送的音频, so that 不把文本脚本当成已核验的听音证据。
22. As an 评估者, I want 在对话结束后统一评估完整轨迹, so that 评分不会干预运行时 user 行为。
23. As an 评估者, I want 未追问、合理推断、错误猜测及未完成修复均有实际证据, so that 不因注入了噪声就自动判为澄清成功。
24. As an 数据维护者, I want 对齐失败、损坏未实施和中断轨迹显式标记, so that 不完整证据不会混入成功样本。
25. As an 数据维护者, I want 保留噪声来源、随机种子、hard 参数及产物哈希, so that 可以审计和复现数据生成。
26. As an 仓库维护者, I want 未选择声学澄清任务时保留现有事件行为, so that 新能力不改变其他任务的含义。
27. As an 仓库维护者, I want 自动测试无需真实 API key, so that 常规回归不依赖外部服务或产生调用成本。

## Implementation Decisions

1. **复用链路。** 扩展现有 duplex runtime decision、完整 TTS 之后的音频准备阶段和离线评估。
   不替换现有 LM 生成、TTS、音频消息、切片器、tick 队列或 orchestrator 的时间推进方式。
2. **同次 LM 标注。** 在现有结构化决策 schema 和 prompt 中增加关键词及文本出现位置的语义标注，
   同步校验标注与输出文本一致。不增加独立 selector 调用，也不把关键词当成 spoken text 发送给 TTS。
   语义标注使用中性要求，不能以“你这句话将被损坏”提示 user。
3. **任务配置与执行隔离。** 使两种声学扰动可明确选择、可复现，并与原有语义歧义 clarification
   和用户打断助手的 clarification 区分。保留已有事件标识与语义；私有执行配置确定哪条话语实际被扰动。
   仅因 LM 产生关键词不代表必定执行扰动；执行结果不回流至 LM context。
4. **对齐服务。** 使用 ElevenLabs `POST https://api.elevenlabs.io/v1/forced-alignment`，
   multipart 请求提交完整干净 TTS 的 `file` 和对应原文 `text`；读取字符及词的 `text/start/end`。
   不需要额外 ASR，也不改用源工程英文限定的本地 MMS 对齐。
5. **凭据。** 官方直连使用现有 dotenv 中的 `ELEVENLABS_API`。
   实施时用户补充确认：本环境使用 `TERUMI_BT_AK` 和已有
   `tutorials/byted-elevenlabs.md` 教程，通过 ModelHub 调用 ElevenLabs；
   其对齐请求仍为 multipart `file` / `text`，网关使用 `ak` 查询参数鉴权。
   同一 AK 可调用用户指定的 OpenAI 网关
   `https://aidp.bytedance.net/api/modelhub/online/v2/crawl`，文本模型使用 `gpt-5.6-luna`。
   不记录凭据、认证头或含 AK 的完整 URL；文本模型调用与语音 agent 接入分别验收。
6. **精确范围。** 文本位置必须匹配到对齐响应的具体出现位置，再映射到用于切片的完整波形采样范围。
   时间戳在重采样前后指向同一段实际发音；明确边界取整和裁剪，禁止按字符长度比例估计音频位置。
   对齐响应和 loss 作为审计证据保存，不将未经验证的阈值当作语言质量保证。
7. **语言和失败处理。** 保持已有 Chinese / English 配置。关键词不存在、出现位置含混、时间范围无效、
   服务失败或音频中未可靠定位时，标记失败并保留诊断；不随机挑其他片段冒充该关键词。
   不将失败样本纳入有效扰动或行为成功分母。重试必须有界，不因异常向 user 透露损坏信息。
8. **完整音频扰动。** 在完整 TTS 就绪、创建 chunk messages 之前执行且仅执行一次目标扰动。
   静音块跨越多个 ticks 时，由已处理波形自然覆盖，不在逐 chunk 路径重算或重复注入。
9. **Hard DNS noise。** 使用原始 hard 的目标 SNR −20 dB，即目标噪声 RMS 约为干净目标信号 RMS 的 10 倍。
   沿用局部混音和输出范围裁剪语义，记录噪声文件、截取位置、缩放、目标 SNR 及 clipping 情况。
   这是相对信号强度定义，不是将整句音量调至最大，也不保证关键词完全不可辨识。
10. **Hard dropchunk。** 保持原始规则：目标范围内两个静音块，每个目标长度为该范围的 65%–90%，
    可重叠，按原始长度约束裁剪；保留采样数，不删除时间。
    不将该模式改成关键词 100% 确定性置零。记录实际静音区间、并集覆盖和种子，不能把两个块长度之和当作实际覆盖。
    移植或优化实现必须保持该规则；不能未经验证就声称不同随机实现逐采样等价。
11. **减少本地开销。** 启动或准备阶段加载、解码并重采样噪声素材，准备可用片段；
    扰动时优先内存中的局部数组运算，不逐句扫描素材目录、不重复读盘、初始化模型或下载资产。
    随机状态属于任务实例，避免修改进程全局随机状态导致并发任务相互影响。
12. **一次损坏与清晰修复。** 每个 Acoustic clarification task 只扰动一条实际目标话语。
    后续重复、补充或解释继续走正常 TTS；不预写修复对话，不因私有损坏状态强制 user 发言。
13. **交互调度。** Agent 发起澄清追问，user simulator 根据实际请求自然回应。
    复用现有被打断后停止发言、清空未播放队列及 LM/TTS 回复能力；
    仅针对 duplex 封装覆盖通用策略和观察期阻塞的情况做必要局部接入，
    不新增专用澄清识别器或重新实现整套交互机制。
    保留已经交付的部分及截断位置，不依据私有受损状态强制 user 回应。
14. **观察与评分分离。** 调整该任务的 active observation 限制以允许修复交互，保留其他事件的观察保护。
    固定窗口到期既不表示 agent 已澄清，也不表示对话已完成；使用实际对话与现有有界终止约束结束轨迹。
    运行中允许话轮决策与证据收集，但不调用语义评分器或逐回复评分。
15. **信息隔离。** user 保留干净意图；后续 context 不包含实际扰动方式、难度、目标位置、执行成败或评估结果。
    审查 pending event 描述、expected behavior、executed event IDs、history 和 execution feedback，
    防止它们透露私有执行信息。Agent 的输入也不能夹带干净答案、关键词标注或评估提示。
16. **证据记录。** 以任务和 utterance ID 关联原始文本、关键词文本及字符位置、干净 TTS、原始对齐响应、
    目标时间及采样范围、实际修改区间、扰动模式及参数、噪声来源或随机种子、处理后音频及哈希。
    另外关联实际交付区间、截断或取消部分、双方真实音频与识别证据。
    不把准备完成但未播放的关键词当作 agent 已听到的输入。
17. **评估时机。** 沿用 runner 退出对话后再调用评估器的顺序；当前实现已满足“评分发生在对话之后”。
    增加声学任务所需的完整轨迹证据，覆盖目标话语、agent 是否及如何追问、user 实际修复与后续反应。
    异常、超时轨迹可生成诊断结果，但不能冒充完整对话成功。
18. **行为结论。** 单独记录是否实际请求澄清、追问是否针对缺失内容、修复是否完成及证据不足情形。
    对有效利用上下文的回答与无根据猜测按完整证据区分；hard 标签不自动意味着必须追问或已经通过。
    音频交付、时间线正确性和模型行为有效性分别报告。
19. **性能口径。** 使用 wall clock 测量本地扰动处理、alignment 请求和 TTS 后准备总耗时。
    记录被测音频时长、目标范围、模式、设备、预加载状态、样本数及汇总统计。
    Simulation tick 与 simulation time 保持原定义；不得把 frozen simulation time 当成 realtime 性能证明。
    当前没有已确认的毫秒 SLA，不编造阈值或达标结论。
20. **窄范围集成。** 将必要的音频处理能力纳入本仓库，通过现有配置来源提供默认值。
    不依赖运行时导入外部 clarification 工作目录，不整体搬入原离线脚本、额外对话生成和训练导出流程。

## Testing Decisions

1. 主验收边界是**现有 user simulator 对外产生的 tick 音频与对话结束后的 assessment**。
   优先在这个现有高层边界验证整条流程，不为内部每个方法创建独立测试入口。
   外部 LM、TTS 与 alignment 在自动测试中使用受控替身，保留真实切片、调度和记录逻辑。
2. 好的测试证明用户可观察结果：重组实际输出 ticks 后，只有指定关键词范围被处理，
   长度及非目标部分保持一致，实际交付与私有证据对应，而不是仅断言某个内部函数被调用。
3. 同次 LM 回复含关键词标注，关键词与位置校验有效，没有独立 selector 调用；
   缺失、重复或不一致标注无法被静默标成有效损坏样本。
4. 在受控对齐响应上覆盖字符、词、多次同名出现、中英文、重采样与跨 tick 边界，
   核对具体被修改的采样。服务失败或无效响应必须留下明确失败记录，禁止随机回退。
5. 两种 hard 模式都经由完整 TTS → 处理 → tick 输出验证。
   DNS noise 对照已知信号核对 clipping 前的目标比例与实际裁剪，dropchunk 核对原始 hard 规则及实际区间并集；
   不用输出难度名称替代音频验证。必要的数值参考校验可以在该集成测试内进行。
6. 受控对话分别让 agent 中途追问、说完后追问、不追问，核对 user 的真实决策机会与修复后清晰音频。
   追问时不因 observation window 被阻塞；中途停止后未交付的音频不再被发送。
7. 对比每次后续 LM 请求和 agent 输入：实际损坏类型、位置、执行状态和评估结果不能泄露；
   变化私有扰动结果时，不应凭空改变传给 user 的干净历史。
8. 对话运行中不发生 judge 调用；退出后使用完整修复轨迹评估。
   中断、超时、对齐失败、目标音频未交付及 judge 失败不能自动判为通过。
9. 保留未选择新任务时的回归，尤其是已有 semantic clarification、contextual interruption 和 memory observation。
10. 既有先例包括 duplex scenarios 的运行时动作、私有事件隔离、观察保护和失败 judge 测试，
    以及 audio effects 的 frame drop、SNR、种子和 timeline 测试。扩展这些现有测试模式。
11. 真实 ElevenLabs 验证与默认自动测试分开，按现有 live API 测试约定控制调用。
    验证 multipart、凭据变量、真实返回的字符/词位置和失败处理；不把受控替身的通过说成真实服务通过。
12. 性能验证单独测量本地扰动，报告预热后样本分布，噪声加载成本单列。
    同时给出 alignment 与准备总耗时；不以不稳定的 CI wall-clock 阈值替代可复查的基准产物。
13. 实际模型行为需通过真实交付音频与整段对话验收，不能仅由绿灯测试、hard 参数或已完成请求推断。

## Out of Scope

- 改写现有完整文本、完整 TTS 或 tick 发送架构；逐 chunk 决定并执行本次扰动。
- 单独关键词选择 LM 调用，为已知 TTS 文本重新做 ASR，或复用英文限定 MMS 作为本任务的对齐依赖。
- Mild / medium 难度、自动扫难度、搜索保证触发澄清的噪声强度。
- 将 hard dropchunk 改成关键词全覆盖静音、真正不发包或删除采样压缩时间。
- 脚本化 agent 追问或 user 修复；向 user 注入受损提示；为了每条出现追问而强制模型行为。
- 运行时逐回复评分、以评分驱动 user 下一句内容，或重构所有事件的调度与评估体系。
- 保证所有受损话语都会产生澄清，或预先宣称中文对齐质量、模型行为和毫秒性能已通过验证。
- 整体搬迁原离线数据生成、难度遍历或训练导出系统；发布网页、部署服务或修改无关工作区变更。

## Further Notes

- 本规格来自本会话已确认的设计讨论；仓库讨论记录为 `docs/clarification-user-audio-design.md`，
  后续实施以本规格为收敛后的需求。
- 术语使用根目录 GLOSSARY，遵守已有 frozen simulation time 决策：模拟音频时间与真实处理耗时分开。
- ElevenLabs 接口依据[官方 forced-alignment 文档](https://elevenlabs.io/docs/api-reference/forced-alignment/create)。
  配置变量名由用户明确指定为 `ELEVENLABS_API`；本文不含密钥。
- 原始 hard 行为依据现有 clarification 工程的 DNS-noise 和 DropChunk 实现；
  移植前后要核对实际规则，不能只复制 hard 名称。
- 本规格本身不是实现结果。代码接入、真实 API、延迟数据和实际模型澄清行为仍需分别验证。
- 用户已确认上述测试边界；本规格发布到本仓库 GitHub Issues，标记 `ready-for-agent`。
