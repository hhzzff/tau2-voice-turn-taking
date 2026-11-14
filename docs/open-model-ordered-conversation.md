# Shared English dialogue protocol

> 2026-11-14 决策：默认采用 **100ms 异步调度窗口**，不等待原生推理完成才推进。
> 原生输入块仍为 SALMONN 80ms、MiniCPM 1000ms，由接入层缓冲拼接。
> 注册的 SALMONN/MiniCPM agent 和 demo 默认采用此模式；`pacing="frozen"`
> （demo 的 `--pacing frozen`）仅用于显式复现旧实验。Moshi/Freeze-Omni
> 仍通过实验入口接入。本决策不表示已通过全事件误差 ≤100ms 的公平性认证。
> 下文原生 tick / 冻结时间的描述保留作历史方案记录。


`ordered_conversation` compares the [currently selected SALMONN checkpoint](open-model-contract-20261102.md) and MiniCPM-o-4.5 with the same
stage order and English-only user generation and agent instructions. Scheduling
uses actual played audio and heard ASR; stage changes do not rely on fixed
recording timestamps or future model output.

1. **Greeting:** the user greets and requests a relaxed three-day Beijing trip
   during initialization. This gives the user's track an explicit opening at
   the start, instead of waiting through an unsolicited model monologue.
2. **Interruption:** after the greeting finishes and at least one second of
   actual model audio has been heard and transcribed, the user changes the trip
   to two days and asks to revise. This is a user constraint change, not an
   assertion that the model made an error.
3. **User backchannel:** after the correction finishes, a stage-specific semantic
   listener classifier checks the subsequent heard speech for a complete useful
   idea. A recent user turn does not disqualify this scheduled action; the
   free-dialogue continuer restrictions do not apply here. Only a YES triggers the original backchannel generation and TTS path.
4. **Model backchannel opportunity:** at a pause or completed heard thought,
   the user supplies a continuous 60–90-word description of accessibility,
   budget, food and history preferences. The scheduled utterance is allowed to
   finish even if the model speaks. Model speech is retained and must be
   inspected; overlap alone is not labelled a successful model backchannel.
5. **Dynamic follow-ups:** the original user turn-taking policy resumes,
   selecting words and reactions based on actual heard audio.

The first four user actions follow the same protocol in both models; their
exact wording and timing depend on native model responses. Stage requests,
fresh post-user-turn heard snapshots and played utterance extents are exported separately from
semantic outcomes. A missing opportunity stays missing; no native model stop
or successful backchannel is manufactured to complete the checklist.

The scheduler still exchanges previous-tick audio with frozen simulation time
and native 80ms / 1000ms widths. Normal user synthesis, chunking, ASR and queues
are reused. The model owns its own playback and listening transitions.
