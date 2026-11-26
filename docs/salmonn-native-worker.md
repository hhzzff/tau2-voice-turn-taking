# SALMONN before-SFT native worker

The shared `scripts/open_model_workers/worker.py --model salmonn` entry point
loads `SalmonnWorker` in the configured SALMONN Python interpreter. It imports
no τ controller packages. Both subprocess and Merlin mailbox transports use the
same connect/step/disconnect protocol described in `open-model-adapter-contract.md`.

Only `cu12_stage2_v2_tfv2_short_sm1_sp_longs1/checkpoint-44000` is accepted.
Loading checks the training template, projection cadence, and native source
fingerprints. The runtime manifest must include the before-SFT training config,
model package, audio visual model dependencies, Matcha-TTS, CosyVoice checkpoint,
and speaker prompt. The training config also references base Llama weights on
`audio-visual-llm-data2`; mount that volume as well as data3, data5 and data6.

## Timing and causal input

Each step accepts exactly 1280 little-endian PCM16 samples at 16 kHz. Online
Fbank exposes only complete windows; causal convolution boundary handling and
transformer KV caches retain history without incorporating future audio. The
first 80 ms yields no complete projection group. That tick returns silence and
a `frontend_startup` event; later ticks yield one speech embedding apiece. This
startup delay is independent of computation time and the scheduler's previous
tick exchange rule.

The language decoder caches its prefix and consumes only newly appended audio
embeddings. A native start-header token can switch between listening and
speaking. The original one-text-token-per-80-ms cadence is retained, including
assistant role insertion at speech onset. No user-annotation-driven forced
interrupt is applied. When the model predicts a return to listening, unheard
playback is cancelled and the discarded sample count is recorded.

CosyVoice produces audio incrementally. Its codec offset and HiFT overlap cache
prevent replaying previously emitted samples. Playback retains the upstream
plain before-SFT policy (`clause_v2=False`), including its 400 ms initial lead.
The playback queue emits at most one 80 ms frame each tick. Completely zero
frames return an empty audio payload so listening is not mistaken for speech;
the adapter pads the conversation track. Every tick still consumes user audio
while speech is queued. Disconnect discards remaining queued audio and releases
acoustic, language and synthesis state. A new connection creates a fresh model
session. Context exhaustion raises an explicit error; no hidden history
truncation is performed.

Events retain acoustic sample counts, produced speech embeddings, native token
transitions, playback/cancellation counters, and synthesis errors. Native text
is generated text, not an ASR transcript of the rendered audio.

## Validation

`tests/test_voice/test_audio_native/test_salmonn/test_salmonn_worker.py` checks configuration
and lifecycle before loading GPU code. Setting `SALMONN_NATIVE_PYTHON` and
`SALMONN_RUNTIME_CONFIG` additionally runs the CPU contract in the native runtime.
That contract uses a tiny external model and TTS substitute while exercising
the actual stable Fbank, causal transformer, projection, session state machine,
playback queue, cancellation, and two fresh connections. It is not evidence of
checkpoint quality or GPU compatibility. Real checkpoint and two-way speech
acceptance remains the Merlin canary and demo matrix.

The public `OpenModelAdapter.connect/run_tick/disconnect` test separately checks
80 ms input size, startup silence, audio conversion, and reconnect isolation.

## Source provenance

`scripts/open_model_workers/SALMONN_PROVENANCE.json` records exact source files,
SHA256 values and adaptations. Stable causal frontend classes were adapted from
pi-bench; incremental playback was adapted from the local SALMONN FDB runtime.
The full-recording inference loop and continuation prefetch were excluded.
Neither local source checkout had a tracked LICENSE, COPYING or NOTICE file;
no upstream license grant is inferred or replaced by this repository's license.
