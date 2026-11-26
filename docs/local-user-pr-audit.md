# Local-user adaptation: pre-PR audit, 2026-11-26

## Verdict and scope

The controlled local-user migration is working, but this branch cannot yet be
described as fully compliant with the source repository's contribution rules
or as having passed every required test. This audit fixed reproducible failure
paths and added regression coverage. Architectural placement, API documentation,
and the remaining full-suite failures still need explicit disposition before a
merge-ready PR. Passing Ruff or completing 19 trials is insufficient evidence.

Scope is the migrated local duplex user plus the 19 controlled voice scenarios.
Native airline/retail/telecom tasks and rewards are not this phase's acceptance
criteria. The P03 X-ASR homophone error is a normal recognition limitation, not
an adaptation defect; this audit does not change its prompts or disclosure rules.

Reviewed baseline: `0f634c66f9a90f573b6ca380229895fa992c391c`.
Reviewed implementation: `f7df2b451edffba9d7bc016176e8b650d7aa4b70` plus the
uncommitted fixes and tests from this audit. The original `tau2-voice-turn-taking`
checkout remains clean on the baseline. Nothing was pushed and no PR was opened.

## Confirmed defects fixed in this audit

1. **Optional dependencies leaked into the existing CLI.** Building the parser
   imported the local runtime and `soundfile`, so even `list` and `--help` failed
   without the local-user extra. Registration now lives in the small
   `local_user_cli.py` module; execution imports the runtime only when selected.
   Three subprocess regressions explicitly block optional audio imports.

2. **One cleanup/export failure skipped remaining cleanup and evidence.** A
   failure in playback finalization, runtime close, provider disconnect, or track
   export could prevent later cleanup and `summary.json`; provenance failure
   also left the recorder worker running. Each finalizer is now attempted,
   failures retain their stage and original error, and recorder close runs in an
   outer `finally`. Cancellation saves partial evidence where possible and is
   re-raised. Journal failures cannot produce a successful return. Runtime
   shutdown also cancels its tasks and closes ASR if observation journaling fails.

3. **Cancelling an ASR coroutine released handles while its worker still ran.**
   `asyncio.to_thread` cancellation does not stop native decoding. The old
   `close()` cleared the recognizer/stream before the thread's subsequent access.
   Worker tasks are now retained and shielded; shutdown awaits them before
   releasing the handles. A blocked fake recognizer reproduces the old race
   without requiring model weights or a live service.

4. **The voice test tier implicitly required a separate extra.** Local runtime
   and converter tests imported `soundfile`/`soxr` unconditionally, although the
   documented `voice` extra does not install them. Relevant tests now skip at
   collection without those optional packages. They execute with `local-user`
   installed; policy, report and ASR worker tests remain independent of them.

An experiment README was also added, as required by `CONTRIBUTING.md`.
Provider compatibility tests cover the three existing modality defaults and
lossless raw response-status events. No model behavior was tuned in this audit.

## Outstanding repository-standard findings

### Experimental placement needs a clear boundary

`CONTRIBUTING.md` requires experimental code to be self-contained under
`src/experiments/`. The runner/report follow that layout, but the frozen scene
controller, scene audio, scenario agenda and scene-specific prompts remain in
`src/tau2/user/local/`. `UserRuntime` directly constructs `ScenarioAgenda` and
the new core OpenAI playback adapter depends on local-user observation classes.
These are observable dependencies, not merely a directory naming preference.

Before treating this as a general core component, isolate scenario-specific
policy/stimuli in the experiment and define the reusable runtime/driver contracts,
or obtain an explicit maintainer decision accepting this experimental component
in core. This audit did not perform an unreviewed relocation of the migration.

The continuous driver itself is intentional: the existing discrete-time adapter
freezes simulated time during work, whereas this experiment measures real elapsed
time. Its separate clock is not evidence that domain adaptation is unfinished.
The existing provider is reused, event handling lives in `_process_event`, and
PCM conversion stays in the shared converter module.

### Public API documentation and model conventions are not fully met

`AGENTS.md` and `CONTRIBUTING.md` require docstrings for public APIs and complex
functions. Examples still missing them include `ScenarioRuntime.control`,
`ScenarioAgenda.update_review`, `FloorController.evaluate`, and the LLM `review`
function. The audit inventory finds 101 undocumented non-private declarations
in the added local/continuous modules, including simple classes and accessors;
this is an inventory, not 101 independent correctness defects. Ruff's configured
rules do not check this requirement.

`AGENTS.md` specifies Pydantic `BaseModel` for data classes. Configuration uses
it, but seven observation/policy/recording/buffer classes use standard dataclasses.
Their internal runtime role may justify an exception, but none is documented.
Resolve this convention explicitly instead of claiming blanket compliance.
Some defaults also remain local (for example target voice `marin` in the
continuous adapter), despite the central-configuration convention.

The `codex/local-user-simulator` branch name is outside the recommended branch
prefixes. This is a minor submission-hygiene issue, not a runtime defect; the
existing seven commits already use conventional commit subjects.

### Required tests are not all green

Core and voice tests were run on the implementation and the same baseline using
the same interpreter. Missing audio dependencies were installed into an isolated
audit overlay at their `uv.lock` versions, with PortAudio/JACK libraries extracted
locally. The working virtualenv, system packages, and original checkout were not
changed to provision these tests.

| Check | Observed result |
| --- | --- |
| Local-user tests including CLI | 34 passed |
| Added provider compatibility tests | 4 passed |
| Optional-package absence simulation | 1 passed, 3 modules skipped; no collection errors |
| `make check-all` | Passed; this runs lint and format, not tests |
| Core baseline | 182 passed, 17 failed, 1 xfailed |
| Core after fixes | 185 passed, same 17 failed, 1 xfailed |
| Voice baseline | 508 passed, 5 failed, 5 skipped, 83 deselected |
| Voice after fixes | 543 passed, 5 failed, 5 skipped, 83 deselected |
| Required provider suite | 77 skipped, 0 executed; credentials unavailable |

The 17 core failures are authentication failures or their downstream empty-result
assertion. The voice failures include a missing machine-specific SALMONN runtime
file and four assertions in the existing third-party simulator tests. They must
not be represented as passing just because they also reproduce on the baseline.
The final baseline and implementation have identical failing test IDs in each
tier; no newly failing core or voice test was observed in this comparison.
Provider tests skipped for missing credentials are not provider compatibility
proof; the added mock tests cover defaults and payload preservation only.

The historical 19 recordings remain valid evidence for their recorded commits.
They were not regenerated after this audit's lifecycle fixes, and are not an
end-to-end result for the modified working tree. This audit's new regressions
exercise the modified paths offline.

## Evidence and next acceptance gate

Local artifacts are in
`/cpfs/user/jingpeng/workspace/tau_pr_audit_20261126/`:

- `regression-before.log`: 10 failures and 2 passes on the original code.
- `runtime-cleanup-before.log`: the original close method fails the recorder-error
  cleanup regression (loaded separately without reverting the working tree).
- `local-after.log`, `provider-compatibility.log`, `optional-gates.log`.
- `core-baseline.log`, `core-after.log`, `voice-final.log`,
  `voice-baseline-complete.log`, `provider-suite.log`, `check-all-final.log`.
- `missing-public-docstrings.txt`, `dataclasses.txt`, and `deps/locked-overlay.txt`.
- `comparison.json`, `reviewed-files.json`: failure-ID comparison and SHA256
  inventory of the reviewed files, including uncommitted fixes.

The initial archive-only baseline had one extra report-test failure because that
test requires Git metadata. A separate baseline Git checkout was established
inside the audit directory, including the original `origin/main` reference, for
the final comparison; the original clone was not used as a mutable test workspace.

A merge-ready PR should resolve the module boundary and documentation/model
conventions, show the required checks in a suitable environment or explicitly
track accepted baseline failures, and bind any claimed live validation to the
final submitted commit. The audit supports a concrete review; it does not certify
that all semantic scenarios or all repository checks have passed.
