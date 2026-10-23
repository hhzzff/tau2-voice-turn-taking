# Open-model τ-voice frozen-time scheduler MVP

## Problem Statement

τ-voice needs to evaluate bidirectional voice interaction with open models, starting with SALMONN-Omni and MiniCPM-o, in addition to API-backed models. The existing configurable tick duration does not by itself supply a stateful, causal local-model execution interface. The models operate at different native audio granularities, and inference may take longer than the audio represented by one step.

We need an inspectable demo that separates scheduling correctness from model conversation behavior and actual inference speed. It must expose normal turn taking, user interruption with correction, user backchannel, and model backchannel opportunities.

## Solution

Extend the existing τ-voice full-duplex scheduling flow with local-model adapters and an explicit frozen simulation time policy. Complete both participants' work before advancing simulation time by one simulation tick, recording tick computation time independently.

Select SALMONN-Omni before-SFT checkpoint-44000 at 80 ms and MiniCPM-o-4.5 at 1000 ms. Default the duration from model selection; reject explicit durations that do not match the selected model. Preserve the existing previous-tick audio handoff between participants.

Provide one documented launch interface and eight demo runs covering both models across four scenarios. Each run produces a stereo WAV with separate participant channels, per-tick JSONL, and a playable HTML timeline. Report both successful and absent model behaviors honestly.

## User Stories

1. As a researcher, I want to select SALMONN-Omni before-SFT, so that I evaluate the agreed checkpoint rather than the adapter's after-SFT default.
2. As a researcher, I want to select MiniCPM-o-4.5, so that I can exercise its native duplex interface.
3. As a researcher, I want the scheduler to select the model's native tick duration automatically, so that ordinary runs need no manual timing configuration.
4. As a researcher, I want to specify a matching tick duration explicitly, so that my invocation documents the timing configuration.
5. As a researcher, I want incompatible durations rejected before inference, so that an apparently valid run cannot silently violate model granularity.
6. As a researcher, I want simulation time to wait for each step's computation, so that slow inference does not distort the simulated conversation timeline.
7. As a researcher, I want actual tick computation time recorded separately, so that I can inspect runtime cost without confusing it with audio time.
8. As a researcher, I want both participants to receive the other's previous-tick output, so that neither gains same-step information because it executes second.
9. As a researcher, I want only already-delivered user audio exposed to the model, so that future input cannot influence its current response.
10. As a researcher, I want conversation state retained across model steps, so that the model can respond coherently while listening and speaking.
11. As a researcher, I want state reset between conversations, so that one demo cannot contaminate another.
12. As a researcher, I want audio longer than one tick carried across subsequent playback ticks, so that generated speech is preserved on the timeline.
13. As a researcher, I want silence and short output represented with the correct duration, so that both audio tracks stay aligned.
14. As a researcher, I want a normal question-and-answer scenario, so that I can inspect basic audio exchange and turn transition.
15. As a researcher, I want the user to attempt a substantive correction during model speech, so that I can observe overlap, stopping, and handling of the correction.
16. As a researcher, I want the user to give a brief acknowledgement during model speech, so that I can observe whether the model continues or yields unnecessarily.
17. As a researcher, I want a longer user utterance with listener-response opportunities, so that I can observe model backchannel behavior.
18. As a researcher, I want the same scenario definitions used for both models, so that differences are interpretable despite their different tick durations.
19. As a researcher, I want scenario triggering distinguished from the model's response, so that a missing opportunity is not reported as a successful behavior test.
20. As a researcher, I want failed or absent behaviors retained in the report, so that the demo does not conceal model limitations.
21. As a demo user, I want a documented launch command, so that I can reproduce a run using the required runtime and assets.
22. As a demo user, I want user and model audio on separate WAV channels, so that I can listen to each participant or their overlap.
23. As a demo user, I want a playable timeline with both speech intervals, so that I can inspect the timing of interruptions and backchannels.
24. As a developer, I want machine-readable per-tick evidence, so that I can trace the audio delivered and produced at every scheduler step.
25. As a developer, I want deterministic scheduling tests without loading model weights, so that failures can be diagnosed cheaply.
26. As a developer, I want real-model smoke runs in addition to deterministic tests, so that passing test doubles does not stand in for successful model integration.
27. As a developer, I want model identity, timing configuration, scenario, and runtime settings recorded with each run, so that artifacts can be attributed and reproduced.
28. As a researcher, I want user timing thresholds interpreted sensibly at both granularities, so that conversion to 1000 ms ticks does not silently erase subsecond behavior settings.

## Implementation Decisions

- Reuse the existing full-duplex orchestrator, streaming participant contracts, audio-native agent/adapter boundaries, and configurable tick plumbing. The feature is local-model stepping with an explicit clock policy, not a second independent scheduler.
- Model selection fixes both identity and native duration: SALMONN-Omni before-SFT checkpoint-44000 uses 80 ms; MiniCPM-o-4.5 uses 1000 ms. SALMONN's after-SFT default must never be selected implicitly.
- Frozen simulation time advances only after the current step completes. For example, three seconds spent computing a one-second tick advances simulation time by one second. Wall-clock receive deadlines and pacing intended for API providers must not define local inference completion or impose realtime execution on this mode.
- Preserve previous-tick handoff: snapshot both incoming chunks before computing either participant's new output, then retain the new outputs for the next step. Define initialization using the existing orchestrator's initial chunks. Native granularity is not a promised response or interruption latency.
- Keep model state alive across steps and clean it up between conversations. Adapt the model-specific streaming operations from pi-bench, with changes scoped to this repository. Runtime/process isolation is an implementation choice to resolve during environment preflight, not a requirement to merge incompatible model dependencies.
- SALMONN must receive audio causally through incremental processing. Full-file feature extraction or preloaded future user audio cannot be used as the interactive model input. MiniCPM must expose its stateful prefill/generate/finalize progression through the shared step boundary.
- Separate model generation size from playback duration. Reuse existing audio conversion, output capping, overflow buffering, and silence handling where applicable. Preserve audio order and duration across ticks, and reflect any supported cancellation explicitly rather than silently losing buffered audio.
- Audit user-simulator threshold conversion at 1000 ms; choose and document behavior that does not silently truncate a positive subsecond setting to zero. Preserve the intended distinction between continuing speech, acknowledging, and requesting the floor.
- Use four scenario definitions for each selected model: normal question and answer; user interruption with correction; user backchannel; and model backchannel opportunity during continuing user speech. Make triggering conditions reproducible and report whether they actually occurred.
- A backchannel is a brief listener acknowledgement without an attempt to take the floor. A substantive correction is a floor-seeking interruption. The scenario report must distinguish these intentions from observed model behavior.
- Deliver a launch interface, aligned stereo WAV, per-tick JSONL, and playable HTML timeline. Record model/variant, native duration, scenario identity, relevant settings, tick index, simulated interval, actual computation duration, and audio handoff/event evidence sufficient to inspect the run.
- Record intended scenario, observed trigger, and observed model response separately. Do not turn absent model backchannel or unsuccessful interruption handling into scheduler success claims, and do not fabricate model responses to fill the demo matrix.
- Preserve existing API-backed and half-duplex operation. Local frozen-time execution is an explicit mode with model-specific capability validation.

## Testing Decisions

The confirmed primary test seam is the existing full-duplex orchestrator's public simulation-step interface, with controlled streaming participants or model-runtime doubles substituted at existing adapter boundaries. Assert emitted audio, delivered inputs, simulation advancement, and recorded artifacts; avoid assertions about private helper calls or internal implementation order.

- Exercise both native durations at the same seam. Supply distinguishable audio chunks to prove previous-tick delivery and absence of future or same-step input leakage.
- Vary computation duration independently of audio duration and verify that simulation time advances by exactly one configured interval per completed tick. Avoid slow timing-dependent sleeps where a controlled runtime/clock can establish the behavior.
- Verify long-output preservation, short-output silence handling, track alignment, and overlapping output. Use sample counts and declared audio formats as the duration reference, not transport packet counts.
- Check model selection and duration mismatch through the public configuration/launch boundary. Include protection against unintended SALMONN after-SFT selection.
- Verify conversation reset and cleanup, and observable handling of unsupported or unavailable runtimes without reporting an unexecuted run as completed.
- Exercise relevant user turn-taking behavior at 80 ms and 1000 ms, including positive subsecond thresholds and the distinction between backchannel and interruption.
- Validate generated JSONL/WAV/HTML from a controlled short run: model provenance, complete tick sequence, matching simulated audio durations, separate participant tracks, and usable playback references.
- Add only necessary model-adapter contract checks at the existing public connect/run-tick/disconnect seam to expose integration faults that scheduler doubles cannot detect. Existing audio-native provider smoke tests offer prior art for lifecycle, fixed played-audio duration, and interruption probes; do not inherit their tool-call requirements for this MVP.
- Existing streaming integration tests, discrete-time agent tests using mock adapters, user turn-taking tests, and the shared provider suite provide testing patterns. Prefer these established boundaries over new test-only interfaces.
- Run the two-model by four-scenario demo matrix with real model inference and the user simulator. Report runtime failures, missing scenario triggers, and unsuccessful model behaviors explicitly. Deterministic tests establish scheduling mechanics; real runs establish integration and supply behavioral evidence.
- Complete repository-required checks appropriate to changed code; run the required lint/format gate before any eventual commit. Publishing this spec does not itself constitute test execution or demo completion.

## Out of Scope

- Completing τ domain tasks that require model tool calls, or adding a new tool-call protocol to these models.
- Wall-clock realtime guarantees, throughput optimization, or interpreting frozen-time response intervals as physical end-to-end latency.
- Arbitrary or aggregated tick durations beyond the two selected native durations.
- Within-tick cross-participant feedback or a new audio exchange protocol.
- Additional models, SALMONN after-SFT comparisons, training, or tuning models to force desired behavior.
- A new benchmark leaderboard, automated semantic judge pipeline, or statistical claim about relative model quality.

## Further Notes

The design follows the agreed frozen simulation time ADR and project glossary. The working branch is `openFDB-time-froze-MVP`. Existing code already supports a variable tick duration; acceptance requires actual incremental open-model operation, not merely exposing another duration flag.

Runtime environments, model assets, compute availability, and user simulator dependencies require preflight before demo execution. Record any resulting limitation explicitly. “Eight scenarios attempted” is not equivalent to eight successful behavioral demonstrations: the report must expose both coverage and outcomes.

This specification synthesizes the agreed conversation. The user accepted the testing seam conditional on reusing τ-voice's existing implementation boundaries. Source verification confirms that the orchestrator already exposes step(), the adapter exposes run_tick(), and an existing streaming test invokes orchestrator.step(). The proposed frozen-time assertions extend coverage at these existing boundaries. No implementation or inference has been performed as part of specification preparation.
