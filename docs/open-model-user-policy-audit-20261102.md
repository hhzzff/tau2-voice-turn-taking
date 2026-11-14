# User simulator interruption parity audit — 2026-11-02

Status: factual audit and design discussion. No production runtime changes in this audit. Findings are reproduced locally without model/API/TTS/ASR calls. The user invoked `grill-with-docs`; implementation decisions remain to be confirmed after reviewing these differences.

## Repository and behavior forks

GitHub metadata reports `hhzzff/tau2-voice-turn-taking` as `isFork=false`, with no parent. The current branch is `openFDB-time-froze-MVP`. This does not establish historical source ancestry.

There is a behavioral divergence: `OpenModelScenarioUser` inherits `VoiceStreamingUserSimulator` but overrides the turn-taking policy. Inheriting its synthesis and chunking does not imply inheriting its interruption behavior.

## Confirmed findings

### 1. The latest SALMONN replay has no online user simulator

`data/simulations/open-model-contract/20261102-selected-long/run_phase.py` loops over recorded input packets and calls the native worker. The 180-second test contains 179 seconds of existing MiniCPM user input plus one second of silence. New SALMONN speech cannot change this fixed user waveform, stop it, or generate a new reply. The website labels this recording as controlled replay; it is not evidence of dynamic-user interruption correctness.

### 2. Scheduled scenarios bypass normal yielding

`src/tau2/user/open_model_scenarios.py:287` returns `keep_talking` whenever `output_streaming_queue` is nonempty. This precedes the `free_dialogue` branch, so ordinary follow-up utterances are also forced to finish. The analogous branch at line 432 affects `qa`, `correction`, `user_backchannel`, and `model_backchannel`.

The standard API user path is:

1. `src/tau2/runner/build.py:327` constructs `VoiceStreamingUserSimulator` with explicit timing settings.
2. `src/tau2/agent/base/streaming.py:1758` receives the agent's previous audio chunk, updates speech counters, then chooses an action.
3. `basic_turn_taking_policy` at line 2428 decides whether to yield based on current overlap and who initiated it.
4. `src/tau2/user/user_simulator_streaming.py:1498` executes `stop_talking`: clears unsent user speech and emits non-speech for this interval.

The open-model scheduled queue branch skips step 3. It therefore never reaches step 4 while queued speech remains. The `long_conversation` branch does delegate ordinary playback to the base policy.

### 3. Shared single-chunk backchannel flag leak

`user_simulator_streaming.py:1555` takes the first backchannel chunk from the queue, then unconditionally sets `is_backchanneling=True`. If this was the only chunk, the queue is empty but the flag stays true. The next ordinary generated utterance does not clear that flag, and the shared base policy treats the whole utterance as a protected backchannel.

This is a shared implementation bug; it can affect API users too. A one-second MiniCPM tick makes one-chunk acknowledgements common. Fixing only the scenario override would leave this defect in place.

The constructor's default `yield_threshold_when_interrupted` is **2 ticks**, not `None`. Passing `None` explicitly disables yielding; the lower-level policy function has a separate `None` default. The audit distinguishes these defaults.

### 4. Coarse ticks silently change the physical threshold

The configured default for an agent interrupting ordinary user speech is 1 second; for a user holding the floor after interrupting the agent it is 5 seconds (`config.py:115`). `build_demo_user` applies `max(2, ticks)`:

| Model | Requested interrupted threshold | Converted ticks | Demo threshold |
| --- | --- | --- | --- |
| SALMONN | 1 s | 13 × 80 ms | 1.04 s |
| MiniCPM | 1 s | 1 × 1000 ms | 2 s |

The base simulator's validation also requires at least two ticks. Passing the standard one-second setting directly to a one-second-tick user would raise, rather than restore parity. This exposes a fine-tick assumption in the original implementation; removing the demo clamp alone is insufficient. A packet marked as speech is counted as a full tick, so short model backchannels inside a one-second packet also need explicit coverage.

### 5. Stage requested is currently mistaken for stage completed

`_emit_ordered` advances `_ordered_phase` before the utterance has been delivered. `_ordered_after_output` waits for the queue to become empty. If yielding is restored, cancellation also empties the queue, so a cut-off correction could be treated as completed and skipped.

Stage request, actual delivery, interruption, and completion must be recorded separately. User constraints that were never delivered must not be treated as heard by the model.

### 6. An unrelated shared tool-speech cleanup defect

`stop_talking` does not clear `delivering_tool_result_speech`. After a tool-result utterance is cancelled, a later ordinary utterance may inherit the high tool-speech threshold. This is reproduced, but **does not explain the current demo**, which rejects tools.

## Reproduction evidence

Persisted under `data/simulations/open-model-contract/20261102-user-policy-audit/`, with hashes of all audited production source files.

- `tau_user_interrupt_diff.py`: 24 comparisons using real `build_demo_user` state, base/scenario policies, and the production action executor. Only audio-effect postprocessing is replaced. Eight ordinary-speech cases differ; both native widths and both overlap initiators are covered.
- `tau_user_interrupt_execution_diff.json` / `.log`: base policy clears three pending chunks and emits zero PCM; ordered/qa emits nonzero PCM and leaves two chunks under the same state. Backchannel-tagged samples remain protected in both policies, as intended.
- `tau_user_bc_flag_repro.py` / `.log`: executes the unmodified production action/policy function bodies with external synthesis/effect doubles. Confirms the single-chunk backchannel flag leak and tool-speech flag leak.
- Existing `test_turn_taking_policy.py`: 7 tests passed. Those tests did not cover either leaking-state sequence and did not compare the overriding scenario policy.

Example local reproduction:

```bash
LITELLM_LOCAL_MODEL_COST_MAP=True UV_NO_SYNC=1 \
  .venv/bin/python /tmp/tau_user_interrupt_diff.py
.venv/bin/python /tmp/tau_user_bc_flag_repro.py
```

## Intentional differences

Frozen simulation time, native 80 ms / 1000 ms model input, process isolation, and using ASR of delivered model audio are not themselves reasons to disable user yielding. In the original API path, cancellation is of queued playback; synchronous full-utterance LLM/TTS generation is not a cancellable streaming operation.

## Proposed correction, pending design confirmation

- Use one shared playback/yielding policy for API and native models. Scenario logic supplies goals and eligible listener actions, and must not force ordinary utterances to finish.
- Preserve the distinction between ordinary speech and a genuinely current short backchannel; clear state when it ends or is cancelled.
- Distinguish partial delivery and cancellation from completed scenario stages.
- Reconcile second-based thresholds with coarse native ticks explicitly, including short model backchannels, silent gaps and previous-tick handoff.
- Retain replay as diagnostics; behavioral acceptance needs fresh dynamic interactions with each model.
- Add regressions for actual output cancellation, no replay of cancelled audio, one-/multi-chunk acknowledgements, both overlap initiators, and cut-off stage handling before real-model reruns.

Open decision: after a correction or long explanation is interrupted, should the user dynamically decide whether and how to restate its unheard content, rather than automatically replaying the old TTS tail? Recommended: yes; do not count the incomplete stage as delivered.
