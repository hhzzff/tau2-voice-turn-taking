# Controlled local-user scenes

These are the 19 latest controlled scenes migrated from the local development
prototype on 2026-11-25. `migration.json` records the source suite and module
hashes. The scene IDs map to the existing τ² duplex event catalog. These fixtures
are stimuli, not completed experiments or evidence of target success.

The migrated policy retains audio-based floor permissions, one Qwen3.5-9B model
for semantic review and expression, and deterministic agenda/disclosure
accounting. Ordinary handoffs may be rule-constrained; record the model action,
effective action and stimulus source separately.

P01 uses the approved continuous spoken retrace, without an added synthetic
2.5-second gap. P03 is the two-expense settlement task; only fully transmitted
answers disclose private amounts. R06 plays the same background-speaker clip
before and after a spoken instruction. Memory probes must not reveal answers.

## Audio provenance

`audio/manifest.json` preserves the source hashes and processing parameters.
Only clips referenced by these scenes are included. Identical clips share a
single file. Do not regenerate stimuli when running target evaluations.

Most clips are synthetic Qwen3-TTS speech. The main voice uses the Chinese
Emilia demo reference (`emilia-zh-3`), selected during local listening review:
https://emilia-dataset.github.io/Emilia-Demo-Page/ . Third-party speech uses the
Qwen3-TTS CustomVoice Dylan voice. Model, seed and exact trim operations are
recorded per clip; text is the synthesis script, not a verified transcript.

The keyboard clip is derived from ESC-50 `1-137-A-32.wav` (Freesound source 137),
under CC BY-NC 3.0. Its source URL, original hash, crop and gain are retained in
the manifest. This material retains its own license, independently of the
repository's MIT code license. Source: https://github.com/karolpiczak/ESC-50 .
