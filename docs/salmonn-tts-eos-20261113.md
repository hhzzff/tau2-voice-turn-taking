# SALMONN TTS EOS diagnosis (2026-11-13)

The user-provided worker 4526491 is reachable with
`NO_COLOR=1 TERM=dumb mlx worker login 4526491 -- <command>`.
The verified device is an A100-SXM4-80GB. This legacy worker uses `mlx`;
`merlin-cli` worker lookup returned 400 and its webshell execution returned 404.

Checkpoint remains:
`/mnt/bn/audio-visual-llm-data2/yuwenyi/playground/exp_results/cu12_stage2-0519_fc-180_aes_er0_chunk30_llama31/checkpoint-44000`.
No weights or shared external SDK files were changed.

## Reproduction and isolation

Evidence root: `data/simulations/control-signals/20261113-salmonn-tts-debug/`.
The fixed 40-second fixture failed at native block 194 (80 ms per block).
The captured call (`failure.pt`) has four text tokens and 12 previously
synthesized speech tokens. CosyVoice's default minimum is eight speech tokens.
Its final-drain loop starts its local counter at zero and rejects EOS for
another eight tokens, even though the streaming prefix already met the minimum.
The repeated-EOS sampler then throws after 50 retries.

Replaying the captured call with its original CPU/CUDA RNG state reproduces the
error. Permitting EOS only because the cumulative minimum is already met ends
normally, with finite input and five additional speech tokens. Seed 43 also
avoids the original failure, so this is a sampling-dependent trigger; previous
successful runs with the same weights do not disprove the bug.

## Narrow compatibility correction

`FDBSpeechPlayback._synthesize_step()` passes `min_token_text_ratio=0.0` only
when the final-drain branch is reached and existing speech tokens already meet
the SDK default 2x minimum. Prefix generation and below-minimum calls retain
SDK defaults. The retry limit, weights, seed, and normal playback queue are
unchanged. This does not implement a general SDK correction for the case where
only part of the cumulative minimum has been met.

CPU regression was red before the change and green after it. It covers the
already-met minimum, preservation of the below-minimum restriction, and full
normal waveform draining without cancellation. The actual captured GPU call
provides independent evidence beyond that test double.

The complete real-weight fixture verification is recorded separately as
`verify.log`, `verified-events.jsonl`, `verified.wav`, and `verified.json`.
Completion of that check is not completion of the 50-repeat timing experiment;
new timing runs must snapshot the repaired source for both direct and scheduled
arms. Existing immutable batches were not modified.

## Complete fixture result

On worker 4526491, the repaired direct runtime completed all 500 native input
blocks (40 seconds), seed 42, with no TTS errors in the recorded events. The
previously failing block 194 completed successfully. `verified.wav` concatenates
returned PCM packets (9.68 seconds); it is not a silence-padded conversation
clock and must not be used to infer onset latency. `verified.json` contains the
checkpoint/device provenance and final playback metadata.

The focused pytest run passed four tests with one skipped native-runtime test
when only SALMONN_NATIVE_PYTHON was configured. The cumulative-minimum regression
was included and passed. Ruff checks and diff whitespace checks passed.

The full targeted run with both SALMONN_NATIVE_PYTHON and
SALMONN_RUNTIME_CONFIG configured also passed: **5 passed** in 133.12 seconds,
including the existing two-fresh-episode causal CPU streaming contract. Warnings
were the existing audioop deprecation and unknown pytest asyncio config option.
