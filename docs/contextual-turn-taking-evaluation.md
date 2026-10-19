# 上下文相关 Turn-Taking 评估

这套流程用于评估自由讨论场景中的 full-duplex 打断。用户模拟器先围绕一个具体主题发起对话，过程中在合适位置插入相关追问、否定、重复请求、话题轻微转移、静默/停止等 interruption，最后回到主话题或自然结束。

核心要求：

- 不使用原 tau-bench 的工具调用、订单查询、账号查询等业务任务。
- 一轮完整对话可以包含多个双工事件。
- 可以规定一轮对话中的 interruption 类型顺序，例如先 `follow_up`，再 `denial`。
- 每个双工事件单独评估被测模型回复质量。
- 整轮对话结束后，再评估用户模拟器自然度。

## 1. 评估对象

### A. 事件级模型回复质量

对一轮对话中的每一个双工/打断事件分别调用 A prompt。A prompt 只评估被测 assistant/model，不评估用户模拟器质量。

输入上下文包括：

- 打断前后局部 tick 窗口
- 用户打断内容，仅作为模型输入上下文
- assistant 被打断时正在说的内容
- assistant 打断后的回复
- 自动统计指标，如是否成功让渡、停止延迟、回复延迟

主要指标：

- `assistant_response_correctness`
- `assistant_conversational_flow`
- `assistant_response_relevance`

### B. 会话级用户自然度

一轮完整对话结束后，对整场对话调用一次 B prompt。B prompt 只评估用户模拟器，不评估 assistant 回复质量。

输入上下文包括：

- 完整对话 transcript
- 主题计划
- 本轮所有已观测 interruption event

主要指标：

- `overall_user_naturalness`

## 2. Prompt 位置

实现文件：

- `src/experiments/tau_voice/contextual_turn_taking_judge.py`

两个 system prompt 常量：

- `EVENT_ASSISTANT_QUALITY_PROMPT`：A prompt，每个双工事件调用一次，只评模型回复质量。
- `CONVERSATION_USER_NATURALNESS_PROMPT`：B prompt，每轮对话结束后调用一次，只评用户自然度。

对应 prompt 构造函数：

- `build_user_prompt()`：构造 A prompt 的 user message。
- `build_conversation_user_naturalness_prompt()`：构造 B prompt 的 user message。

## 3. A Prompt 评估规则

A prompt 的判断边界：

- 用户插话只作为输入上下文，不给用户自然度打分。
- 不评价工具调用、任务完成、业务规则，除非它们直接影响 assistant 对插话的回复。
- 重点看 assistant 的回复是否正确、相关、对话流是否自然。
- streaming transcript 可能出现切词和额外空格，judge 应先在心里归一化这些分片伪影。

输出 JSON 示例：

```json
{
  "sample_id": "sim-id:269",
  "interruption_type": "topic_switch",
  "scores": {
    "assistant_response_correctness": 4,
    "assistant_conversational_flow": 4,
    "assistant_response_relevance": 5
  },
  "overall_decision": "pass",
  "failure_modes": [],
  "notes": "The assistant yielded, answered the side question, and returned to the main meal-prep discussion."
}
```

## 4. B Prompt 评估规则

B prompt 的判断边界：

- 只评用户模拟器，不评价 assistant 回答是否正确。
- 重点看用户是否像真人一样推进一场围绕具体主题的语音对话。
- 打断应出现在合理时机，内容应和主话题或相邻子话题有关。
- 如果用户转到相关侧话题，之后应自然回到主话题。
- 自由讨论场景中，用户不应突然进入订单、账号、工具查询、客服任务框架。
- streaming transcript 可能出现切词和额外空格，judge 应先在心里归一化这些分片伪影。

输出 JSON 示例：

```json
{
  "simulation_id": "sim-id",
  "task_id": "0",
  "scores": {
    "overall_user_naturalness": 4
  },
  "overall_decision": "pass",
  "failure_modes": [],
  "notes": "The user stayed on the meal-prep theme, asked related side questions, and returned to the main planning topic."
}
```

## 5. 工作流

1. 生成主题计划：每轮测试先拟定一个具体主话题、相关侧话题、对话弧线和打断策略。
2. 运行 full-duplex 对话：user simulator 用 LLM 生成文本，用 TTS 合成音频；被测模型使用 GPT-realtime。
3. 抽取 interruption events：保存 `events.csv`，包含类型、时间、停止延迟、回复延迟、打断前后 transcript。
4. 事件级评估：对 `events.csv` 中每个 event 调用 A prompt，写入 `event_records`。
5. 会话级评估：按 `simulation_id` 聚合事件，对每轮完整对话调用一次 B prompt，写入 `conversation_records`。
6. 汇总指标：分别汇总 `summary.event_assistant_quality` 和 `summary.conversation_user_naturalness`。

## 6. 类型序列规范

如果需要规定一场对话内多个双工事件的类型顺序，使用 `--type-sequence`：

```powershell
.venv\Scripts\python.exe src\experiments\tau_voice\interruption_type_experiment.py run `
  --type-sequence follow_up,denial `
  --samples-per-type 1 `
  --contextual-max-interruptions 2 `
  --topic-source discussion `
  --audio-native-model gpt-realtime
```

执行语义：

- 第 1 个真实 interruption event 使用 `follow_up`。
- 第 2 个真实 interruption event 使用 `denial`。
- 超过序列长度后不再触发新的规范 interruption。
- `events.csv` 中每个事件的 `interruption_type` 会写入实际消费的类型，而不是统一写成 `sequence`。
- topic plan 会显式写出 `Interruption sequence`，让 user simulator 的 LLM 能看到“先追问、再否定”的流程要求。

支持的类型：

- `follow_up`
- `denial`
- `repeat_request`
- `topic_switch`
- `silence_stop`

## 7. 运行示例

只生成 prompt，不调用 LLM：

```powershell
.venv\Scripts\python.exe src\experiments\tau_voice\contextual_turn_taking_judge.py `
  --events-csv data\simulations\contextual_discussion_sample_20260819_001\events.csv `
  --output data\simulations\contextual_discussion_sample_20260819_001\contextual_judge_dryrun_ab.json `
  --interruption-type topic_switch `
  --max-events 2 `
  --dry-run
```

调用 LLM judge：

```powershell
$env:OPENAI_API_KEY = "<your key>"
$env:HTTP_PROXY = "http://127.0.0.1:7890"
$env:HTTPS_PROXY = "http://127.0.0.1:7890"
$env:ALL_PROXY = "http://127.0.0.1:7890"

.venv\Scripts\python.exe src\experiments\tau_voice\contextual_turn_taking_judge.py `
  --events-csv data\simulations\contextual_discussion_sample_20260819_001\events.csv `
  --output data\simulations\contextual_discussion_sample_20260819_001\contextual_judge_ab.json `
  --interruption-type topic_switch `
  --max-events 2 `
  --review-model openai/gpt-4.1-mini
```
