# Acoustic clarification tasks

An acoustic task uses the existing live user conversation and complete TTS
pipeline. The same user model decision includes a neutral keyword and its exact
character occurrence. The private audio executor aligns the complete clean audio,
degrades the selected span once, and passes the waveform to the existing tick
slicer. Later user turns are clear. The agent decides whether and when to ask for
clarification; the user model receives no degradation state.

Prepare a standalone task without selecting a semantic ambiguity event:

```bash
uv run python -m experiments.tau_voice.duplex_bench prepare \
  --acoustic-clarification dropchunk --language Chinese \
  --output /tmp/clarification-scenario.json
uv run python -m experiments.tau_voice.duplex_bench run \
  --scenario /tmp/clarification-scenario.json --max-seconds 120 \
  --output /tmp/clarification-run
```

The same option can accompany `--events` or override a loaded scenario when
running. Existing event IDs retain their meanings. In particular,
`proactive.clarification` is an ambiguous user intent, rather than acoustic damage.

Alignment uses ElevenLabs `POST /v1/forced-alignment` with a WAV `file` and original
`text` as multipart fields. When `TERUMI_BT_AK` is set, requests use the configured
ModelHub ElevenLabs gateway with `ak` authentication. Otherwise they use the
official endpoint and `ELEVENLABS_API` as `xi-api-key`. The runner loads `.env`;
direct simulator callers must configure their process environment. Neither
credential is included in private reports or model execution feedback.

Hard dropchunk applies two silent gaps, each sampled between 65% and 90% of the
target span length, with the original integer length clamps. Gaps may overlap;
this does not guarantee that the whole word is removed. A local NumPy RNG makes
replays deterministic without changing global random state. Alignment must map
the exact keyword occurrence to actual character or word boundaries; there is
no timing estimate or random fallback. Alignment failures and unchanged audio
are failed inputs. Only one utterance is attempted per task, including failures.

The acoustic task builder disables incidental random effects and telephony
compression so output ticks carry the resampled PCM waveform. Other duplex
modes retain their channel configuration. At the native agent boundary, actual
audio is converted to the configured provider format: the default 8 kHz μ-law
path uses the existing stateful telephony converter. `agent_audio/` records the
provider-facing input, and `agent_input_conversion.json` records source/provider
formats, durations, and hashes. Source sample offsets remain in 16 kHz PCM;
provider audio duration maps those offsets to its rate. The shared policy handles agent
interruptions, and its `stop_talking` action cancels the remaining queued ticks.
The original natural endpoint and actual stop remain separate evidence. Existing
last-tick padding and boundary fades still apply.

`events.json` contains a private `acoustic_clarification` section with the
annotation, exact alignment response, target and effective changed sample ranges,
gap union, random seed, audio hashes, delivered sample ranges, canceled audio,
and per-stage wall-clock times. `acoustic_audio/` retains the complete clean and
degraded audio, actual delivered user audio, and incoming assistant audio.
Clear follow-up user turns have separate delivery records. `synthesis.json`
preserves original TTS responses before resampling. Clean scripts are intentions,
not verified transcripts. They and the private annotation are removed from
agent-facing output tick text fields.

The timings distinguish the alignment request, local waveform degradation, and
preparation after TTS. Network time is not local corruption time, and simulation
time advances by the existing tick duration regardless of preparation wall time.
A target record marked `incomplete` can describe a truncated original utterance
whose damaged target was already delivered; conversation completion is separate.
The final evaluation must additionally establish whether delivered acoustic
damage mattered, whether the agent made an appropriate clarification request,
and whether a later clear repair was used correctly. Successful audio corruption
alone does not establish clarification success.

After conversation termination, the runner writes consumption evidence and then
performs acoustic assessment. User output from tick `n` is consumed in agent tick
`n+1`; an emitted final frame with no receiving tick is excluded. When available,
ASR uses the decoded provider input tap, with its first sample mapped to receiving
tick 1. Intended user scripts remain private intent, never verified transcripts.

The evaluator discovers semantic clarification requests from actual assistant
ASR. It judges request quality using only saved audio context before the earliest
possible request onset, then judges repair using the complete later interaction.
A midutterance context is separately cropped and transcribed; the rest of that
utterance and later repair cannot influence request adequacy. User nonresponse
leaves repair incomplete without invalidating a reasonable request. A supported
contextual inference without a question is recorded separately from completed
clarification; unsupported guessing can fail.

Question timing uses conservative segment bounds, independently of semantic
quality. During-user and after-user results use the same content standard. A
segment crossing the original natural endpoint stays uncertain; cancellation of
an unplayed tail does not move that endpoint or invalidate an already delivered
damaged target. Timing precision is one simulation tick, not ASR arrival time.

`assessment.json` retains the private construction/delivery references under
`acoustic_clarification`, with an `assessment` containing input validity,
conversation/evidence status, question, repair, timing, and explicit denominators
and missing/judge-failure counts. A timeout or incomplete evidence cannot pass.
`--no-semantic-judge` records mechanical assessment without final ASR calls.
`python -m experiments.tau_voice.duplex_bench assess --output RUN_DIR` reuses saved
ASR and cropped prefixes; it never synthesizes, aligns, transcribes or opens an
agent session. Add `--judge-model MODEL` to rerun only semantic judgment.
