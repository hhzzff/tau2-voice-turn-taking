# User Simulator Turn-Taking Flow

这份文档说明当前 voice user simulator 在 full-duplex 测试里如何判断：

- 是否应该打断 agent
- 是否应该正常回话
- 用什么 prompt 参与判断或生成

示例参数来自样本：

`data/simulations/sequence_followup_denial_sample_20260821_002/`

## 1. 基本时间单位

这次样本的实际配置：

- `tick_duration_seconds = 0.2s`
- `interruption_check_interval_seconds = 2.0s`
- `wait_to_respond_threshold_other_seconds = 1.0s`
- `wait_to_respond_threshold_self_seconds = 5.0s`
- `yield_threshold_when_interrupted_seconds = 1.0s`
- `integration_duration_seconds = 0.5s`

含义：

- 系统每 `0.2s` 推进一个 tick。
- 当 agent 正在说话时，user simulator 大约每 `2.0s` 检查一次是否要打断。
- 当 agent 停止说话后，user simulator 需要等 agent 静默超过 `1.0s`。
- 同时 user 自己也需要静默超过 `5.0s`。
- 两个静默条件都满足后，才触发一次正常 user 回复。

## 2. 是否应该打断

判断入口：

- `src/tau2/user/user_simulator_streaming.py`
- 函数：`user_interruption_policy()`

底层调用位置：

- `src/tau2/agent/base/streaming.py`
- 函数：`basic_turn_taking_policy()`

### Force 模式

这次样本使用的是 force 模式：

```powershell
--type-sequence follow_up,denial
--force-interruptions
--contextual-interruption-rate 1.0
--contextual-min-agent-words-before-interrupt 12
```

实际逻辑：

1. agent 正在说话。
2. 每 `2.0s` 检查一次。
3. 当前 agent 已经说了至少 `12` 个词。
4. 当前 sequence 里还有未消费的类型。
5. `contextual-interruption-rate = 1.0`，所以命中后一定触发。
6. 第一次 interruption 消费 `follow_up`。
7. 第二次 interruption 消费 `denial`。
8. 序列消费完后不再触发规范 interruption。

这次样本结果：

- 第 1 次：`follow_up`，发生在 `41.8s`
- 第 2 次：`denial`，发生在 `55.8s`

### LLM 判断模式

如果不启用 `--force-interruptions`，是否打断由 LLM 判断。

Prompt 常量：

- `INTERRUPTION_DECISION_PROMPT`

位置：

- `src/tau2/user/user_simulator_streaming.py`

核心 prompt：

```text
You are analyzing a conversation to decide if the user should interrupt the agent.

Conversation history (most recent at bottom):

<conversation_history>
{conversation_history}
</conversation_history>

The agent is CURRENTLY speaking (you can see their ongoing speech in the conversation above).

Based on the conversation so far, should the user interrupt the agent NOW?

Consider:
- Has the user heard enough to understand what the agent is asking or saying?
- Has the user heard enough to have a response, question, or correction ready?
- Did the agent just complete the sentence which has all the pertinent information the user was looking for?
- Do NOT repeatedly interrupt the agent if it has spoken only a few words (say less than 5 words).

Respond with ONLY "YES" if the user should interrupt now, or "NO" if they should keep listening.
```

如果存在 contextual topic plan，还会追加：

```text
Contextual topic plan for this test:
{topic_plan}

Only answer YES when the interruption would be a natural related side question
inside this plan. Prefer waiting until the agent has said enough substantive
content to make the interruption specific. Do not interrupt just to satisfy the
experiment condition.

For a contextual turn-taking test, one well-timed side-topic interruption is
expected after the conversation has a concrete main topic. If the agent is now
giving a long explanation, listing requirements, confirming details, or moving
through next steps, answer YES when the user can naturally ask one specific
related question from the topic plan.
```

如果当前 sequence 指定了类型，还会追加该类型的判断指令，例如：

- `follow_up`
- `denial`
- `repeat_request`
- `topic_switch`
- `silence_stop`

## 3. 如何生成打断内容

当打断动作已经被选中后，user simulator 会调用 user LLM 生成具体 utterance。

生成入口：

- `src/tau2/user/user_simulator_streaming.py`
- 方法：`_generate_full_duplex_voice_message()`

生成时会给 LLM：

- user simulator 的 system prompt
- 当前对话历史
- 角色提醒
- topic plan
- 当前 interruption type 的 generation instruction

类型生成指令来自：

- `INTERRUPTION_GENERATION_INSTRUCTIONS`

示例：

```text
follow_up:
You are interrupting the agent. Generate a short follow-up or clarifying
question about what the agent is currently saying.

denial:
You are interrupting the agent. Generate a short correction, rejection,
or disagreement with what the agent just said.
```

因此 sequence 模式下：

- 第一次生成时使用 `follow_up` 的生成指令。
- 第二次生成时使用 `denial` 的生成指令。
- 事件记录中的 `interruption_type` 会写入实际消费的类型。

## 4. 是否应该正常回话

正常回话不是用 LLM prompt 判断“现在该不该回”。

它由规则触发：

- `src/tau2/agent/base/streaming.py`
- 函数：`basic_turn_taking_policy()`

规则是两个静默阈值都满足：

```text
time_since_last_other_talk > wait_to_respond_threshold_other
time_since_last_talk > wait_to_respond_threshold_self
```

这次样本对应：

```text
agent 静默 > 1.0s
user 自己静默 > 5.0s
```

两个条件都满足后，动作变成：

```text
generate_message
```

然后才调用 user LLM 生成用户下一句正常回复。

## 5. 总结

当前 user simulator 的 turn-taking 行为可以概括为：

- 是否打断：
  - force 模式下由规则决定。
  - 非 force 模式下由 `INTERRUPTION_DECISION_PROMPT` 让 LLM 判断 YES/NO。
- 打断类型：
  - 单类型模式使用 `TAU2_INTERRUPT_EXPERIMENT_TYPE`。
  - 序列模式使用 `TAU2_INTERRUPT_EXPERIMENT_SEQUENCE`，例如 `follow_up,denial`。
- 打断内容：
  - 由 user LLM 生成。
  - 会加入当前 interruption type 的 generation instruction。
- 是否正常回话：
  - 由静默时间规则决定，不由 LLM 判断。
- 正常回复内容：
  - 由 user LLM 根据 system prompt、对话历史和 topic plan 生成。
