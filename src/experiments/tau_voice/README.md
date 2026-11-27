# Voice interaction experiments

These are experimental entry points, not domain task evaluators. The local-user
route exercises 19 frozen voice scenarios using the migrated user simulator,
the existing OpenAI provider, and independent real-time send/playback clocks.
It does not award airline, retail, or telecom task rewards. Finishing a scenario
does not mean its semantic objective passed.

## Local-user experiment

Install with Python 3.12 (the repository's pinned version):

```bash
uv sync --extra local-user --extra dev
python -m experiments.tau_voice.duplex_bench local --help
```

The trial needs external X-ASR weights, a local chat-completions service and
the Qwen TTS service with the pinned reference speaker. Service defaults live
in `tau2/config.py`; runtime options are validated by `LocalUserConfig`.
Put `OPENAI_REALTIME_API_KEY` in the ignored `.env` file, then run:

```bash
python -m experiments.tau_voice.duplex_bench local \
  --asr-model-path /path/to/x-asr-160ms \
  --scenes P03 \
  --output data/simulations/local-user-smoke
python -m experiments.tau_voice.local_user_report \
  data/simulations/local-user-smoke
```

For other service addresses, pass `--config` with a validated JSON file; see
[the example](examples/local_user_config.json). `--asr-model-path` overrides the
file's ASR path, and may be omitted when the file supplies one. Unknown settings
are rejected. The [local-user package](local_user/README.md) documents ownership,
registration and the playback/delivery contract.

Omit `--scenes` to run all 19 scenarios. Output directories must be new. Trials
preserve partial audio and failure summaries where the filesystem is writable;
cleanup/export failures cannot count as successful trials. Cancellation cleans
up, attempts to save partial evidence, and propagates to the caller.

The report uses local ASR of actually played audio as the user's observation.
Server transcripts are audit evidence only. Delivery receipts denote submission
to the client transport, not acknowledgment that the remote model heard audio.

## Validation

```bash
uv run pytest tests/test_local_user_cli.py tests/test_streaming/test_local_user_*.py
make check-all
```

The CLI test also runs in the core test tier. Tests needing local audio packages
skip when those packages are absent; install `local-user` to exercise them.
Changes to shared voice components additionally require the repository's voice
tier (`uv sync --extra voice --extra local-user --extra dev; make test-voice`)
and provider suite. Live provider tests require credentials and must be reported
as skipped when unavailable.

See [integration and evidence](../../../docs/local-user-integration.md) for the
runtime contracts and [reviewed trials](../../../docs/local-user-integration-review.md)
for the existing recordings. Their provenance refers to the commits that ran
them; later code edits do not retroactively validate against those recordings.

## Other entry points

- [Dynamic duplex benchmark](DUPLEX_BENCH.md): event selection, preparation,
  execution, and assessment through the original dynamic-user path.
- [Third-party interference](THIRD_PARTY_INTERFERENCE.md): interference stimuli
  and their evaluation setup.
