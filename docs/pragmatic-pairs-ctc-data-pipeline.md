# Pragmatic Pairs + CTC Data Pipeline

本文档汇总当前仓库里用于合成 full-duplex 语音对话数据的两层流程：

- Pragmatic pairs：把一场对话设计成一个或多个有语用功能的双工事件序列，例如 `follow_up -> denial`。这里的 pair 不是一个单独脚本名，而是通过 `--type-sequence`、topic plan 和 interruption type instruction 组合出来的数据设计单元。
- CTC：contextual turn-taking conversation。每条样本先拟定一个具体讨论主题，用户模拟器围绕主话题自然对话，在合适时机插入相关追问、否定、重复请求、话题轻微切换或静默/停止，再回到主话题。

目标是让另一台机器可以从环境配置开始，跑出完整的语音对话、事件表、指标汇总和 LLM judge 结果。

## 1. 代码入口

主要文件：

- `src/experiments/tau_voice/interruption_type_experiment.py`
  - 生成 typed interruption / pragmatic pair / CTC 样本。
  - 调用 `tau2 run --audio-native` 真实跑 full-duplex 对话。
  - 输出 `topic_plan.*`、`results.json`、simulation ticks、audio taps、`events.csv` 和 `summary.json`。
- `src/tau2/user/user_simulator_streaming.py`
  - 用户模拟器的 LLM turn-taking 判断和打断内容生成。
  - 读取 `TAU2_CONTEXTUAL_TURN_TAKING_PLAN`、`TAU2_INTERRUPT_EXPERIMENT_SEQUENCE` 等环境变量。
- `src/experiments/tau_voice/contextual_turn_taking_judge.py`
  - 对生成后的事件和整轮会话做 LLM judge。
  - A prompt：逐个双工事件评价 assistant 回复质量。
  - B prompt：整轮会话结束后评价 user simulator 自然度。

参考文档：

- `docs/contextual-turn-taking-evaluation.md`
- `docs/user-simulator-turn-taking-flow.md`

## 2. 环境准备

在新机器上安装依赖：

```powershell
git clone <repo-url>
cd tau2-bench
uv sync --extra voice --extra dev
```

Linux/macOS 可以使用同样的 `uv sync --extra voice --extra dev`。

根目录创建 `.env`，至少配置：

```env
OPENAI_API_KEY=<openai_or_openai_compatible_key>
ARK_API_KEY=<doubao_user_llm_key>
VOLCENGINE_TTS_API_KEY=<volcengine_tts_key>
```

如果使用 ElevenLabs TTS，则改用：

```env
ELEVENLABS_API_KEY=<elevenlabs_key>
```

如果使用代理：

```powershell
$env:HTTP_PROXY = "http://127.0.0.1:7890"
$env:HTTPS_PROXY = "http://127.0.0.1:7890"
$env:ALL_PROXY = "http://127.0.0.1:7890"
```

如果 judge 或普通 chat completion 走 OpenAI-compatible base URL：

```powershell
$env:OPENAI_BASE_URL = "http://101.6.64.59:3011/v1"
```

注意：Realtime 被测模型走 WebSocket provider，默认 endpoint 在 `src/tau2/config.py` 的 `DEFAULT_OPENAI_REALTIME_BASE_URL`，不是这个 chat completion 的 `OPENAI_BASE_URL`。

## 3. 数据合成总流程

完整 pipeline：

1. 选择 interruption 类型或类型序列。
2. 为每条样本生成 contextual topic plan。
3. 将 topic plan 注入 agent policy 和 user simulator instructions。
4. 运行 full-duplex 语音对话。
5. 用户模拟器用 LLM 判断是否打断或正常回话，用 TTS 合成用户音频。
6. 被测模型使用 audio-native provider，例如 OpenAI Realtime。
7. 保存完整 simulation ticks 和 audio taps。
8. 从 ticks 分析 interruption events，生成 `events.csv`。
9. 汇总自动指标，生成 `summary.json`。
10. 对 `events.csv` 调 LLM judge，生成 `contextual_judge_ab.json`。

## 4. Pragmatic Pairs 配置

支持的 interruption types：

```text
follow_up
denial
repeat_request
topic_switch
silence_stop
```

单个 pragmatic pair 示例：

```text
follow_up -> denial
```

含义：

- 第一次双工事件：用户追问或澄清 assistant 当前解释里的具体点。
- 第二次双工事件：用户对 assistant 的合理建议提出阻力或反对。assistant 不必真的说错，用户可以因为麻烦、成本、偏好不匹配、无聊或不现实而否定。

对应参数：

```powershell
--type-sequence follow_up,denial
--contextual-max-interruptions 2
```

如果要批量覆盖更多 pair，可以分别跑：

```powershell
--type-sequence follow_up,denial
--type-sequence topic_switch,repeat_request
--type-sequence follow_up,silence_stop
```

当前脚本一次只接收一个 `--type-sequence`，多个 pair 需要多次调用或外部批处理。

## 5. CTC 样本生成命令

推荐使用非 force 模式，让 LLM 判断更自然的打断时机：

```powershell
$env:OPENAI_API_KEY = "<your openai key>"
$env:ARK_API_KEY = "<your ark key>"
$env:VOLCENGINE_TTS_API_KEY = "<your volcengine tts key>"

.venv\Scripts\python.exe src\experiments\tau_voice\interruption_type_experiment.py run `
  --type-sequence follow_up,denial `
  --samples-per-type 10 `
  --save-to ctc_followup_denial_001 `
  --domain retail `
  --disable-domain-tools `
  --audio-native-provider openai `
  --audio-native-model gpt-realtime `
  --user-llm openai/doubao-seed-character-260628 `
  --user-llm-api-base https://ark.cn-beijing.volces.com/api/v3 `
  --user-llm-api-key-env ARK_API_KEY `
  --tts-provider volcengine `
  --speech-complexity control_behavior `
  --topic-source discussion `
  --contextual-topic-plan `
  --contextual-interruption-rate 0.35 `
  --contextual-min-user-turns-before-interrupt 2 `
  --contextual-min-elapsed-seconds-before-interrupt 20 `
  --contextual-min-seconds-since-last-interruption 18 `
  --contextual-max-interruptions 2 `
  --max-steps-seconds 180 `
  --audio-taps
```

如果网络需要 Clash：

```powershell
  --proxy http://127.0.0.1:7890
```

如果只想快速验证覆盖率，可以打开 force 模式：

```powershell
  --force-interruptions `
  --contextual-min-agent-words-before-interrupt 24
```

force 模式会更容易保证生成指定事件，但可能造成“不等 assistant 说够就打断”的不自然样本。正式数据建议先使用非 force 模式，只有覆盖不足时再单独补样本。

## 6. Topic Plan 生成逻辑

`interruption_type_experiment.py` 会为每个 sample 生成：

- `main_topic`
- `related_side_topics`
- `interruption_sequence`
- `interruption_sequence_steps`
- `conversation_arc`
- `interruption_guidance`
- `style`

当 `--topic-source discussion` 时，脚本使用自由讨论主题，不围绕 tau-bench 工具调用或客服查询组织对话。

已有 discussion topic profile 包括：

- 工作日晚餐 meal prep
- 居家办公 routine
- 周末 hiking 选择
- 新项目组首次会议准备

也可以用 `--contextual-main-topic` 指定固定主话题：

```powershell
  --contextual-main-topic "help the user plan a realistic study schedule for a language exam"
```

每条样本会保存：

```text
data/simulations/<save-to>/sequence/sample_XX/topic_plan.json
data/simulations/<save-to>/sequence/sample_XX/topic_plan.txt
```

## 7. 用户模拟器工作方式

用户模拟器分两类行为：

### 是否打断

非 force 模式下，agent 正在说话时，用户模拟器约每 `interruption_check_interval_seconds` 检查一次，调用 LLM 判断是否打断。

相关判断 prompt 位于：

```text
src/tau2/user/user_simulator_streaming.py
INTERRUPTION_DECISION_PROMPT
```

判断会结合：

- 最近 conversation history
- agent 当前正在说什么
- topic plan
- 当前应该消费的 interruption type
- 最小 user turn 数
- 最小 elapsed seconds
- 距离上次 interruption 的最小间隔

### 打断内容生成

一旦决定打断，用户模拟器调用 user LLM 生成具体 utterance，再调用 TTS 合成音频。

生成时会注入：

- user simulator system prompt
- 对话历史
- topic plan
- 当前 interruption type 的 generation instruction

### 正常回话

正常回话不是 LLM 判断“现在是否该说话”，而是由静默阈值触发：

```text
time_since_last_other_talk > wait_to_respond_threshold_other
time_since_last_talk > wait_to_respond_threshold_self
```

触发后才调用 user LLM 生成下一句普通回复。

## 8. 输出目录结构

一次 `--type-sequence follow_up,denial --samples-per-type 10 --save-to ctc_followup_denial_001` 后，主要输出：

```text
data/simulations/ctc_followup_denial_001/
  events.csv
  summary.json
  sequence/
    sample_00/
      topic_plan.json
      topic_plan.txt
      results.json
      simulations/
        <simulation_id>.json
      artifacts/
        task_0/
          both.wav
          agent_output.wav
          user_output.wav
          user_tts-output.wav
          assistant_labels.txt
          user_labels.txt
```

关键文件：

- `both.wav`：双方合并后的完整对话音频。
- `agent_output.wav`：assistant 输出音频。
- `user_output.wav`：用户经过效果处理后的输出音频。
- `user_tts-output.wav`：用户 TTS 原始输出。
- `simulations/<id>.json`：完整 tick 序列。
- `events.csv`：所有识别出的 interruption events。
- `summary.json`：自动指标聚合。

## 9. 自动指标

`summary.json` 按 interruption type 汇总：

- `attempted_interruptions`
- `simulations_with_interruptions`
- `interruption_success_rate`
- `post_interrupt_answer_success_rate`
- `post_interrupt_stop_latency_mean_sec`
- `post_interrupt_stop_latency_samples`

`events.csv` 每行一个双工事件，核心字段：

- `interruption_type`
- `simulation_file`
- `interruption_start_sec`
- `user_transcript`
- `agent_overlap_transcript`
- `interruption_success`
- `post_interrupt_stop_latency_sec`
- `post_interrupt_answer_success`
- `agent_response_latency_sec`
- `agent_response_transcript`

## 10. LLM Judge

生成后运行：

```powershell
$env:OPENAI_API_KEY = "<your judge key>"
$env:OPENAI_BASE_URL = "http://101.6.64.59:3011/v1"

.venv\Scripts\python.exe src\experiments\tau_voice\contextual_turn_taking_judge.py `
  --events-csv data\simulations\ctc_followup_denial_001\events.csv `
  --output data\simulations\ctc_followup_denial_001\contextual_judge_ab.json `
  --max-events 20 `
  --review-model openai/gpt-4.1-mini
```

如果使用官方 OpenAI endpoint，可以不设置 `OPENAI_BASE_URL`。

输出结构：

```text
contextual_judge_ab.json
  mode
  review_model
  events_csv
  summary
    event_assistant_quality
    conversation_user_naturalness
  event_records[]
  conversation_records[]
```

A prompt 逐事件评分 assistant：

- `assistant_response_correctness`
- `assistant_conversational_flow`
- `assistant_response_relevance`

B prompt 整轮评分 user：

- `overall_user_naturalness`

## 11. 复用到别的机器的最小清单

需要带走：

```text
src/experiments/tau_voice/interruption_type_experiment.py
src/experiments/tau_voice/contextual_turn_taking_judge.py
src/tau2/user/user_simulator_streaming.py
src/tau2/voice/utils/volcengine_tts_utils.py
src/tau2/voice/utils/openai_tts_utils.py
docs/pragmatic-pairs-ctc-data-pipeline.md
```

更稳妥的方式是直接同步整个仓库和 `uv.lock`，然后在新机器上跑：

```powershell
uv sync --extra voice --extra dev
```

不要同步 `.env` 到公共仓库。新机器单独创建 `.env` 并填入自己的 key。

## 12. 快速验收

生成完成后检查：

```powershell
Get-ChildItem data\simulations\ctc_followup_denial_001
Get-Content data\simulations\ctc_followup_denial_001\summary.json
Get-Content data\simulations\ctc_followup_denial_001\contextual_judge_ab.json
```

确认：

- `events.csv` 至少有目标类型事件。
- `summary.json` 中目标 type 的 `attempted_interruptions` 大于 0。
- `both.wav` 可以播放。
- `contextual_judge_ab.json` 里 `event_records` 数量等于要评估的事件数。
- `conversation_records` 数量等于被评估的 simulation 数。

## 13. 常见问题

### 事件太少

提高：

```powershell
--contextual-interruption-rate 0.5
```

或者降低：

```powershell
--contextual-min-user-turns-before-interrupt 1
--contextual-min-elapsed-seconds-before-interrupt 15
--contextual-min-seconds-since-last-interruption 12
```

### 打断太生硬

降低 interruption rate，增加最小间隔，并保持非 force 模式：

```powershell
--contextual-interruption-rate 0.25
--contextual-min-user-turns-before-interrupt 2
--contextual-min-elapsed-seconds-before-interrupt 25
--contextual-min-seconds-since-last-interruption 20
```

### 想强制得到指定 pair

使用：

```powershell
--force-interruptions
--contextual-min-agent-words-before-interrupt 30
```

但这类样本应标注为 coverage 补样本，不建议和自然模式样本混在一起统计自然度。

### 不想要 tau-bench 工具调用

保持：

```powershell
--disable-domain-tools
--topic-source discussion
--contextual-topic-plan
```

脚本会设置 free conversation agent policy 和 user instructions，避免订单、账号、航班、工具查询等客服任务框架。
