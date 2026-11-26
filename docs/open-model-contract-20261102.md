# Open model audio contracts — 2026-11-02

## Selected checkpoint

SALMONN now defaults to:

`/mnt/bn/audio-visual-llm-data2/yuwenyi/playground/exp_results/cu12_stage2-0519_fc-180_aes_er0_chunk30_llama31/checkpoint-44000`

The worker reads this checkpoint's own parent `config.json`. Its settings are Llama 3.1, 80 ms native stepping, `add_end_of_speech=true`, `simple_mode_echo_ratio=0`, and `encode_chunk_size=30` seconds. Encoder state resets every 3,000 Fbank frames. Speech projection groups are formed across encoder chunks; no zero padding is inserted at a chunk boundary. These 30-second encoder chunks are distinct from the scheduler's 80 ms ticks.

The initial decoder prefix now follows the upstream default Llama template: BOS, context, audio placeholder, user header, EOT. The previous worker placed a system header and EOT before the first audio placeholder. Speech-end handling appends the predicted speech-end token, think token, and next audio/user prefix, permitting the next assistant turn.

## Test evidence

Artifacts: `data/simulations/open-model-contract/20261102-selected-checkpoint/`.

- Selected-checkpoint configuration test: passed; checks checkpoint path, matching training config and echo ratio.
- Native CPU session contract: passed, two independent sessions and 64 causal ticks. Tiny external model/tokenizer/CosyVoice substitutes exercise the production worker, acoustic frontend, decoder stepping and playback; this does not establish behavior of the actual checkpoint.
- Encoder numerical oracle: streaming output matches the full external causal encoder, run independently for each chunk, with global projection grouping. Tested 24-frame and actual 3,000-frame chunks, with two boundaries each. The old unbounded encoder fails this comparison (6 tokens instead of 5 in the short case); corrected output has 5 and 749 complete tokens respectively. The incomplete final projection group remains buffered because stream end has not been declared.
- Long CPU contract: two independent 120.8-second sessions (3,020 ticks total) pass. Before the fix, the new chunk boundaries eventually caused a speech-embedding underflow. The worker now continues ingest and playback while waiting for the next complete projection group; each session records one frontend wait, without losing an input packet or fabricating a future embedding. This fix was made after freezing the 32-second GPU canary, so that canary does not verify this later branch.
- MiniCPM worker regression: the native SDK result's `is_listen`, `end_of_turn`, `current_time`, and `cost_all_ms` now survive the worker and adapter boundary. No forced user-onset playback cut was introduced.

## Clock and delivery semantics

`FullDuplexOrchestrator.step()` snapshots both previous chunks, generates the next user chunk synchronously, and then advances the agent with the previous user chunk. During user LLM/TTS computation, the model receives no additional native step. Simulation time is frozen; the audio interval is delivered after computation finishes rather than dropped. The scheduler contract test blocks user generation and checks the received native sample counts and literal decoded samples for both native widths.

Frozen simulation permits computation longer than the audio interval. It does not prove real-time operation. Real-time feasibility requires sustainable processing throughput and bounded buffering. The native canary records CUDA-synchronized step durations, startup time separately, and deadline misses against 80 ms / 1,000 ms.

## Real-weight verification

Merlin job: `dc5fc6bda9815095` completed (`done` / `finished`). Requested resource class: A100 SXM 80 GB; actual `torch.cuda.get_device_name(0)`: **NVIDIA A800-SXM4-80GB**. Startup/model loading is excluded from per-step durations.

This is a controlled replay of existing user audio, not a new dynamic dialogue. Both SALMONN phases use the same newly selected checkpoint and exact input packets; only the initial decoder template differs. MiniCPM uses a transparent native SDK recorder to compare each current input waveform and returned flags, text and PCM bytes.

| Replay | Native steps | Median / p95 compute | Steps over native deadline |
| --- | ---: | ---: | ---: |
| SALMONN, previous initial prefix | 400 (32 s) | 61.1 / 822.4 ms | 67 / 400 |
| SALMONN, upstream initial prefix | 400 (32 s) | 54.5 / 71.4 ms | 18 / 400 |
| MiniCPM, native SDK parity | 75 (75 s) | 793.0 / 976.2 ms | 4 / 75 |

Both SALMONN runs received the same 116 nonzero input intervals; activity spans 0–5.04 s and 7.04–11.28 s. Both predicted a listening transition at step 138 (11.04 s). The previous-prefix run started another reply at step 151; the upstream-prefix run did not resume within the 32-second replay. **The prefix candidate is not a demonstrated behavioral repair.** Green prefix and encoder tests establish matching implementation contracts, not better conversation behavior.

MiniCPM SDK parity passed for all 75 inputs and returned flags/text/PCM bytes. Listen state transitions were: listen at 0, speak at 4, listen at 13, speak at 14, listen at 35, speak at 65. The one-second input at every step matched the current recording block exactly. This establishes that native listening transitions can occur, not that interruptions are prompt or that corrections are understood.

These timings include the production native worker's CUDA-synchronized computation, excluding mailbox, telephony conversion, scheduler and external user-simulator costs. They do not certify end-to-end real-time performance.

Final-code long replay: Merlin job `41b81cc7141a4548`, artifacts in `data/simulations/open-model-contract/20261102-selected-long/`. It reuses 179 seconds of existing MiniCPM user input, losslessly rechunked to 80 ms, plus an explicitly recorded one-second silence tail (2,250 input intervals). This exercises multiple 30-second boundaries with the later frontend-wait repair. It is not a fresh dynamic user/model dialogue.

The job completed (`done` / `finished`) on an actual **NVIDIA A100-SXM4-80GB**. All 2,250 current input hashes, sequential input sample totals and emitted native audio hashes passed the post-run audit. No future Fbank frames were supplied. The frontend wait occurred once, at native step 1,125; the complete recording was preserved. Native assistant starts occurred at steps 59, 173, 811, 1,240 and 1,971; listening transitions occurred at 141, 359, 1,156 and 1,509. No native speech-end token was observed in this recording, so that branch is covered by CPU contracts, not by this real-weight replay.

Median compute was **56.8 ms**, p95 **89.7 ms**, with **147 / 2,250 (6.53%)** steps exceeding 80 ms. Total native computation was **242.55 s for 180 s input** (RTF 1.35), excluding the 120.08-second startup. This run does **not** meet sustained real-time throughput. The calculation excludes scheduler, transport and external user LLM/TTS/ASR costs.

Review artifacts (all under `data/simulations/open-model-contract/20261102-selected-long/salmonn-final/`):

- `audit.json`: hash, sample count, causal-feature and native control evidence.
- `summary.json` and `steps.jsonl`: exact model path, hardware and per-step timing.
- `aligned-stereo.wav`: 180-second native playback review, user left / model right. Empty output slots are explicitly padded with silence, preserving time.
- `user-input.wav` and `model-played.wav`: separate 180-second aligned tracks.

Native text is decoder emission, not acoustic ASR alignment. The model resumed after the correction and responded again later, but also repeated its final question during later silence. Correction understanding and natural backchannel behavior are not accepted by these mechanical checks. A fresh dynamic dialogue and semantic/audio review remain separate from the verified replay.

Public regression tests: 20 passed, 1 optional native-interpreter integration test skipped. The native CPU contracts were executed directly in the native interpreter separately. Ruff checks passed for touched code. Full-suite validation has not been performed.
