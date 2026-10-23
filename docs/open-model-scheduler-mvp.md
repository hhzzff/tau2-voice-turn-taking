# Open-model τ-voice scheduler MVP

Status: MVP scope agreed through the design interview. Runtime preflight and implementation remain outstanding.

## Confirmed scope

- Use frozen simulation time as defined in `../GLOSSARY.md` and `adr/0001-frozen-simulation-time-for-open-model-mvp.md`.
- Demonstrate each model interacting bidirectionally with the τ-voice user simulator; save both audio tracks and per-tick events for inspecting overlap and scheduling.
- Domain task completion with tool calls is outside the initial demo acceptance scope.
- Default the tick duration from the selected model. Accept an explicit duration only when it matches the model's native duration; reject mismatches.

| Model | Selected variant | Native tick |
| --- | --- | --- |
| SALMONN-Omni | before-SFT, `logs/cu12_stage2_v2_tfv2_short_sm1_sp_longs1/checkpoint-44000` | 80 ms |
| MiniCPM-o | 4.5 | 1000 ms |

Implementation references are the `inference/salmonn-omni/` and `inference/minicpm_o/` directories in `/mnt/bn/audio-visual-llm-data6/terumi/pi-bench`. The selected SALMONN variant differs from that adapter's default after-SFT variant; it must be selected explicitly.

## Audio exchange

Retain the current orchestrator handoff: each participant receives the other's previous tick output. Both inputs are captured before either participant computes its new output. The user explicitly accepted retaining this rule for the MVP.

A scheduler tick covers one interval on the simulated audio timeline; it is not an utterance or a WebSocket packet. A participant can generate audio spanning multiple ticks. The current adapter limits playback to one tick and buffers excess audio. Provider input/output concurrency within a tick is distinct from the orchestrator's cross-participant handoff boundary. Native tick duration determines exchange granularity, not guaranteed response or interruption latency.

## Demo scenarios

Run each selected model against all four scenarios:

| Scenario | Interaction to exercise | Observation |
| --- | --- | --- |
| Normal question and answer | User asks a question; model responds | Audio delivery, response, and turn transition |
| User interruption with correction | User speaks a substantive correction during model speech | Overlap, model stopping or continuing, and subsequent handling of the correction |
| User backchannel | User gives a brief acknowledgement during model speech without requesting the floor | Whether the model continues speaking or yields |
| Model backchannel | User continues a longer utterance with opportunities for a brief listener acknowledgement | Whether the model gives a brief acknowledgement, remains silent, or takes the floor |

Provide reproducible triggering conditions and record whether each intended event actually occurs. Model behavior is an observation, not a guaranteed consequence of successful scheduling. A failed or absent response must remain visible in the demo report; do not substitute prerecorded model output for live model behavior.

## Deliverables and acceptance

- One documented launch command with model selection and optional validated tick duration.
- Per-conversation stereo WAV with user and model on separate channels.
- Per-tick JSONL recording simulated time, actual computation duration, audio handoff, and relevant events.
- An HTML timeline showing both participants' speech intervals and overlap, with playable audio.
- Coverage of two models by four scenarios; report scenario triggering and observed model behavior separately from scheduler checks.
- Verify causal audio delivery, correct tick advancement, preservation of output across tick boundaries, and operation at both native durations. Review the user's subsecond timing thresholds at the 1000 ms tick duration.

## Implementation preflight

Inspect model runtime environments, available compute, and user simulator configuration before running the demo. Reuse pi-bench implementations as references while keeping MVP changes in this repository. The demo artifacts have not yet been produced.
