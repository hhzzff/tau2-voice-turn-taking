# Local-user adaptation: pre-PR audit, 2026-11-26

## Scope and review status

This branch implements the controlled local duplex user experiment and its 19
staged voice scenarios. It does not implement native tau airline/retail/telecom
tasks or their rewards. The local scenario format and executor stay together;
tau supplies the user interface, provider and shared audio conversion.

The architectural and contribution-standard findings from the initial audit
have been addressed. The PR is a draft for user review. Full core/voice tiers
still have baseline failures, documented below; completing a live trial is not
proof of target semantic correctness. P03 X-ASR homophone errors remain normal
recognition limitations, and prompts/disclosure rules were not tuned to hide them.

- Comparison baseline: `0f634c66f9a90f573b6ca380229895fa992c391c`.
- Initial audited implementation: `f7df2b451edffba9d7bc016176e8b650d7aa4b70`.
- Final runtime and dependency commit: `0ae4b9199e2f25a2961ec414d4260749733d4cf1`.
- Branch: `experiment/local-user-simulator` in an isolated worktree. The original
  `tau2-voice-turn-taking` checkout remains clean on the baseline.
- Final evidence/test/documentation commits follow the runtime commit. Recordings
  are attributed to their actual clean runtime commit, not to a later PR SHA.

## Resolved findings

### Lifecycle and optional dependencies (`bd74848`)

- CLI construction no longer imports optional audio packages. Existing `list`
  and `--help` work without the local-user extra; execution imports it explicitly.
  Subprocess regressions block the optional audio imports.
- Each cleanup/export stage is attempted independently. Errors retain their
  original cause and stage; recorder shutdown runs in an outer `finally`.
  Cancellation exports partial evidence where possible and is re-raised.
  Failed durable journaling cannot yield a successful return.
- Runtime shutdown still stops background tasks and closes ASR if drain
  journaling fails. Native ASR worker tasks are shielded and joined before
  recognizer/stream handles are released; a blocked fake recognizer reproduces
  the original cancellation race.
- Tests requiring `soundfile`/`soxr` are dependency-gated. Policy, report and
  ASR worker tests remain usable without those optional packages.

### Experimental ownership (`aa30718`)

The local runtime, agenda, scenario stimuli/prompts, user participant, continuous
driver and OpenAI playback adapter now live under
`src/experiments/tau_voice/local_user/`. Core tau does not import the experiment.
`register_local_user()` registers the `FullDuplexUser` implementation only when
the experiment CLI is selected. Core registry defaults remain unchanged.

The real-time driver is intentional: its send/playback/receive clocks continue
while ASR, B and TTS run. This preserves the experiment's elapsed-time semantics;
it does not reinterpret local scenes as native tau domain tasks. Shared provider
options and PCM conversion remain in the core voice layer, with unchanged
existing defaults and live compatibility coverage.

### Models, interfaces and documentation (`58077c2`, `0ae4b91`)

- `contracts.py` defines outgoing packets and runtime/target callbacks.
- Six serializable observation/policy/ASR data classes use strict Pydantic
  models. Immutable snapshots are frozen, while playback retains shared item
  identity. `OutputBuffer` is a resource-owning class, not a data model.
- All non-private declarations in the local experiment modules have docstrings;
  an AST inventory found none missing. Defaults for voice, ASR model and reference
  hash come from `tau2/config.py`.
- Validated `--config` JSON supports external service settings; explicit
  `--asr-model-path` overrides the file. Unknown fields are rejected.
- The `all` dependency extra includes `local-user`; the lock update introduces
  no unrelated package-version changes. The branch uses the contribution guide's
  experimental prefix and conventional commit subjects.
- Each experiment has a README with ownership, lifecycle and execution guidance.

The scene assets and prompt constants are unchanged from the initial audited
implementation. The prompt source-file hash changes with docs/types/imports;
constant equality and fixture hashes are checked separately. Pydantic Frame and
FloorState construction measured about 1–2 microseconds per operation in local
batch means, not a general end-to-end latency guarantee.

## Validation

Tests used Python 3.12.13. Missing voice packages were installed into an isolated
lock-version overlay, with PortAudio/JACK libraries extracted locally. Neither
the original checkout nor system packages were modified for provisioning.
Baseline comparison used a complete Git checkout with the baseline and
`origin/main` refs, avoiding artificial failures from archive-only metadata.

| Check | Implementation | Baseline / interpretation |
| --- | --- | --- |
| Local-user and provider compatibility regressions | 42 passed | CLI, lifecycle, ASR cancellation, contracts, config, provider defaults/events |
| Core tier, valid OpenAI credentials | 193 passed, 12 failed, 1 xfailed | 187 passed, same 12 failed, 1 xfailed |
| Voice tier | 544 passed, 5 failed, 5 skipped, 83 deselected | 508 passed, same 5 failed, 5 skipped, 83 deselected |
| Required provider suite, real OpenAI | 11 passed, 0 failed, 66 skipped | Other-provider credentials unavailable |
| `make check-all`, `git diff --check` | Passed | Lint/format/whitespace checks, not runtime tests |
| Final 19-scene live replay | 19 normal completions, 47.89 minutes | Same clean runtime commit and matched case/config hashes; not 19 semantic passes |
| Live cancellation / disconnect | 2 passed | Partial audio/summary retained, ASR drained, resources released |

Initial core runs lacked a generic `OPENAI_API_KEY`. Re-running both checkouts
with a valid existing OpenAI credential raised the pass count. The remaining 12
failures request the retired `gpt-3.5-turbo` model, including a downstream empty
simulation assertion. No credential value was printed or committed.

The five voice failures are the missing machine-specific SALMONN runtime
`/mnt/bn/audio-visual-llm-data6/terumi/pi-bench/inference/salmonn-omni/runtime.json`
and four existing third-party simulator assertions. The baseline and PR failure
IDs are identical in both tiers. This is evidence of no new failing test ID in
these runs, not a claim that the full required suites passed. See the validation
JSON for the exact failing IDs.

The required provider suite executed all 11 available OpenAI cases through the
repository's existing test interface. The other 66 cases were skipped, not
counted as passing. Its result file was redirected to the ignored audit directory
so the source tree stayed clean during the real-time trials.

## Live evidence and limitations

All 19 trials completed normally with their planned stages sampled, observations
drained, and no target response marked `incomplete`. The replay used the same
clean runtime commit throughout. Each case/config hash matches the corresponding
previously selected recording. Total recorded trial duration is 47.89 minutes.
The listening artifact contains 19 hash-bound notes and 95 valid local audio
files; HTML data and audio headers were checked without opening a browser.

The two opt-in live fault tests passed in 54.45 seconds. Each waited for actual
target playback before cancellation or WebSocket closure. Cancellation preserved
a `cancelled` summary and re-raised `CancelledError`; disconnection preserved an
`error` summary. Both retained partial audio, drained observations, closed the
provider, stopped recorder/runtime workers and released native ASR handles.
The final committed test is byte-identical to the externally executed test.

P03 received both missing amounts in one delivered answer. The target's final
57/96-yuan transfers match the independent ledger. Its normal ASR homophone and
arithmetic-transcription errors remain recorded. M01 completed 16 stages in
952.8 seconds and answered all three recall questions correctly in the local
transcripts; the questions themselves supplied no answer. The repeated-budget
limitation below still applies. Its maximum B input was 1688 tokens and maximum
target input was 5315 tokens; it is not a 9B long-context limit benchmark.

A simple local-ASR scan found no 8–80-character substring repeated consecutively
three or more times. This does not rule out other repetition, hallucinations or
factual errors. For example, I01's airport metro claim needs separate fact
checking. The full per-scene notes, counts, failure IDs and source hashes are in
[the portable validation evidence](local-user-pr-validation.json).

The recorded case/config hashes are compared with the corresponding previously
selected trial, including its explicit runtime and output-token budgets. Every
new run uses a new directory; historical failures and selected recordings remain
intact. Normal end, complete agenda and B's `met` are distinct from semantic pass.
Log-derived review notes bind to the exact summary SHA256; final listening remains
with the user. The page shows local ASR of actually played audio separately from
service transcripts retained only for audit.

The local B model can deviate from a stage's expression constraints. In this
M01 replay it repeated the initial 6000-yuan budget in the scope discussion,
although that stage asks not to repeat numbers. This limits one-exposure memory
claims and is annotated as stimulus quality, separately from target recall.
R01/R03/R04 likewise preserve target behavior that did not meet the scenario
expectation. No prompt changes or selective reruns were used to remove these
observations.

## Reproduction and artifacts

Use the [experiment README](../src/experiments/tau_voice/README.md) for service
configuration and the [integration guide](local-user-integration.md) for replay
commands. The opt-in live fault tests require `LOCAL_USER_LIVE_TEST_ENABLED=1`,
`LOCAL_USER_ASR_MODEL_PATH`, local B/TTS services and OpenAI credentials. They are
marked `full_duplex_integration` and do not add live API work to default tests.
With the enable flag unset, the final committed test reports 2 skipped and makes
no live API calls. Final lint/format and whitespace checks passed; runtime and
fixture/dependency file hashes still match the recording commit.

Ignored machine-local audit artifacts remain under
`/cpfs/user/jingpeng/workspace/tau_pr_audit_20261126/`:

- `contracts.log`, `source-invariants.json`.
- `core-pr-authenticated.log`, `core-baseline-authenticated.log`.
- `voice-pr.log`, `voice-baseline-complete.log`, `pr-baseline-comparison.json`.
- `provider-suite-pr.log`, `provider-suite-pr-results.txt`.
- `live-standard.log`, `live-extended.log`, `live-memory.log`, `live-faults.log`.

Historical pre-fix reproduction logs are retained in the same audit directory.
The committed validation JSON contains portable counts, identities and hashes;
large recordings and raw service journals are intentionally local artifacts.
