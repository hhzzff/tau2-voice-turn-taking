# 八模型控制信号实测（2026-11-12）

已对八个目标执行真实端点调用或 GPU 权重推理。共 60 个探测 case，56 个完成并收到音频，4 个失败；这不是打断成功率。每项启用信号都能追溯到实际 trace 和 SHA256，见 [证据账本](control-signals-live-evidence.json)。未观测信号留在 `CONTROL_SIGNAL_CANDIDATES`，不作为已验证契约。

## 结果与播放语义

| 目标及实际版本 | 实测结果 | 接入处理与边界 |
| --- | --- | --- |
| GPT Live `gpt-live-1` | 6 次会话收到连续 PCM，包含静音；未收到离散取消事件，长静音段约 29 秒 | 保留静音的媒体时间；不伪造 LISTEN，不因静音清队列。已修复 agent 将非空静音包标为说话的问题 |
| GPT Realtime `gpt-realtime-2.1` | 插话后收到 `input_audio_buffer.speech_started`；`response.done` 实际出现 `completed` / `cancelled` | 前者按语音起点截断；completed 保留尾音，cancelled 只取消对应 response。缺 response ID 时不得清除新回复 |
| Doubao S2S `1.2.6.0` / `1.2.6.1` | 均观测到说话中插话后的 `conversation.item.input_audio_transcription.started`，以及 `response.output_audio.done` | ASR started 表示用户轮次开始，只有存在旧播放时才构成取消；done 表示生成结束，保留尾音 |
| Gemini Live `gemini-3.1-flash-live-preview` | 实际收到 `serverContent.interrupted` 和 `serverContent.turnComplete` | 前者取消旧播放；后者仅表示 turn 关闭，可出现在被打断后，不单独称为自然结束 |
| MiniCPM `MiniCPM-o-4.5` | 实测 `is_listen`、`end_of_turn`；显式 break 后及后续 8 个强制监听步骤均返回监听 | 未结束 turn 的 LISTEN 清残留，正常结束保留尾音；`set_break` / `force_listen` 标为外部策略，不能报告为模型自主打断 |
| SALMONN `checkpoint-44000`（下述完整路径） | 实际观测 assistant start；修复后自然播放结束、恢复监听、第二次回答均已出现 | EOS 是播放控制器补入的输入标记，不能等模型预测 EOS；原生用户打断转换本次未观测，继续保留为候选 |
| Freeze-Omni 原始 checkpoints | 实测 listening 阶段 `cl/ss`，generation 阶段 `cs/sl` | 必须带调用阶段解释 stat；`sl` 不等于打断。参考控制器在分离轮次测试中取消旧 TTS，属于外部策略 |
| Moshi：`Moshika-pytorch-bf16` | 实际逐帧 encode → step → decode，连续输出 PCM，包含静音 | 保留媒体时间；本轮没有建立离散取消信号，不把文本 EOS 或静音当作取消 |

SALMONN 使用用户指定权重：
`/mnt/bn/audio-visual-llm-data2/yuwenyi/playground/exp_results/cu12_stage2-0519_fc-180_aes_er0_chunk30_llama31/checkpoint-44000`。
Moshi 使用 `moshika-pytorch-bf16/model.safetensors`，不外推到其他 Moshi 变体。

## 关键差异与异常

- GPT Live 连续静音是有效媒体。agent 原先用“有字节”判说话，现用已有的共享音频活动判断，仍保留原音频和时长。
- SALMONN 原实现等待模型预测 `<|end_of_speech|>`，可能持续 THINK。原始 FDB 在文本 EOT 且实际播放耗尽后将该标记补入输入。已按此修复，先用不会预测 EOS 的边界模型复现失败，再跑 CPU 和实际权重。真实正常轮次 rendered=played=495040 个 16kHz 样本，取消 0；复合场景恢复监听并开始第二次回答。两种自然结束控制均标为 `playback_controller`，不冒充原生打断。
- Doubao 1.2.6.1 早期插话未触发 ASR started；等待模型说话 3 秒后，明确打断触发取消约 153 万字节旧音频。1.2.6.0 的打断与 backchannel 均曾触发取消。信号确实存在，但不保证每次插话都会触发，也不能据此归因于版本本身。
- MiniCPM SDK 在说话 turn 未结束时可能将预测 LISTEN 改成 tts_bos。强制监听验证通过只证明受控路径可用，不证明模型自主让出话权。
- Freeze-Omni 分离轮次测试等待首轮用户结束及 2.5 秒静音，再等待模型说话 3 秒；观察到状态转换和参考控制器取消。该探测没有验证原 socketio 服务端到端行为，不能把探测器的 stop_tts 写为原生模型信号。
- `gpt-realtime-1.5` 在当前网关于可用音频会话前 EOF，未获得正向证据。字典的实际模型版本门禁不将 2.1 的证据沿用到 1.5。
- 初次 SALMONN 探测读错输出采样率 provenance 字段，3 个 case 失败，已修正并重跑；失败 case 不进入信号准入。

## 验证范围

探测输入为仓库通用合成语音及新生成的固定旅行请求、打断、backchannel；未上传历史用户对话。实际闭源探测复用 pi-bench 的传输和播放队列，保留 raw events、raw PCM 与逐步取消记录。取消审计未发现取消后继续播放旧 response 的片段。

8 个目标的原生协议已实际探测；**不表示 8 个 τ-voice adapter 已完成端到端实测**。Doubao、Freeze-Omni、Moshi 仍为参考接入；GPT Live 为部分接入。账本及注册表明确保存此边界。本轮不验证动态用户模拟器质量、音频语义正确率或计算耗时小于时间片的实时保证。

复现入口与产物结构见 [探测说明](../scripts/control_signal_probes/README.md)。原始产物在 `data/simulations/control-signals/20261112-*`（不提交音频与权重）。字典回归同时检查每个启用信号都有实际 trace witness，未知或未测版本保持未知。

## 最终回归

- 聚焦回归 **104 passed**：控制字典及证据准入、开源 adapter/worker、Gemini parser、SALMONN adapter、离散时间 agent（含真实 GPT Live 静音片段）。
- SALMONN CPU 原生流契约通过；额外长序列验证 2 个新会话、3020 个因果 tick，覆盖跨 30 秒编码块。真实权重验证另计，见上文及账本。
- 按目录规则运行官方 provider suite：77 skipped，因该套件所用测试环境/开关未启用；不能算作在线通过。上文在线证据来自单独执行并保存原始 trace 的实际探测。
- Ruff lint/format、`git diff --check` 通过。
