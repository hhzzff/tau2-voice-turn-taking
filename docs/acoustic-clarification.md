# Acoustic clarification tasks

An acoustic task uses the existing live user conversation and complete TTS
pipeline. The same user model decision includes a neutral keyword and its explicit
textual occurrence. The private audio executor aligns the complete clean audio,
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

Hard DNS noise uses the same full-TTS alignment and tick path. Select it with
`--acoustic-clarification dns_noise --dns-noise-path /path/to/noise.wav`
on `prepare` or `run`; a directory is also accepted. Without an override, the
configured DNS corpus path is used. The simulator scans a directory once and
preloads one seeded asset, decoding, converting to mono and resampling before
utterance processing. It retains that in-memory asset across state resets;
dropchunk and ordinary tasks do not load noise. Missing, malformed or too-quiet
assets produce explicit failed input evidence rather than a fallback span.

DNS hard means a target SNR of -20 dB before clipping: noise RMS is approximately
10 times the clean target RMS. The source recipe selects an in-memory fragment
with RMS at least 0.01 using up to 12 seeded offsets, repeating short assets when
necessary. Failure to find a qualified fragment is explicit. Only the keyword
span is mixed; samples are clipped to [-1, 1] and quantized to PCM16. This does
not guarantee unintelligibility or imply that the clipped audio achieved -20 dB.
Private records include asset path/hash, fragment offset, repetition, scale,
preclip RMS values, target SNR and clipped-sample count. Asset preparation time is
separate from local degradation and TTS postprocessing. No per-utterance asset
scan, decode, resample or download occurs in the timed degradation operation.

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

## ModelHub gateway and reproducible validation

The explicitly selected `--byted-gateway` route uses `TERUMI_BT_AK` for every
service: `byted/gpt-5.6-luna` for the user and semantic judges,
`byted/gpt-realtime-2.1` for the actual audio agent, ElevenLabs
`eleven_multilingual_v2` for complete TTS, `scribe_v1` for perception/final ASR,
and ElevenLabs forced alignment. Original official provider defaults remain
available. No credential is passed through model kwargs or written to reports.
Gateway chat uses the Azure-compatible ModelHub endpoint; realtime uses the
internal `/openai/realtime` WebSocket with `Api-Key`, rather than guessing a URL
from the text endpoint. See central `tau2.config` constants for these routes.

```bash
uv run python -m experiments.tau_voice.duplex_bench prepare \
  --acoustic-clarification dropchunk --language English --seed 42 \
  --user-model byted/gpt-5.6-luna --output /tmp/acoustic-scenario.json
uv run python -m experiments.tau_voice.duplex_bench run \
  --scenario /tmp/acoustic-scenario.json --byted-gateway --env-file .env \
  --max-seconds 180 --output /tmp/acoustic-modelhub-run
```

For DNS, change the mode to `dns_noise` and supply `--dns-noise-path` on either
command. A single mounted WAV avoids scanning a large corpus. `--tts-model`
selects the actual ElevenLabs model ID; `--primary-voice` accepts its voice ID.
The gateway default foreground voice is the verified Sarah voice
`EXAVITQu4vr4xnSDxMaL`. Acoustic-only trials do not synthesize a background
speaker. For event sets using background speech, provide a distinct available
ElevenLabs `--background-voice`; OpenAI voice names are not ElevenLabs IDs.
No live request is made during preparation. An explicitly supplied scenario may
contain ordinary background goals and style preferences, without a spoken script
or prescribed agent action. Model trajectories are stochastic despite a fixed
local corruption seed; retain every trial rather than overwrite a directory.

The runtime model now annotates `{text, occurrence}`: an exact keyword substring
and zero-based **non-overlapping** occurrence number. The executor resolves that
explicit occurrence to character offsets before alignment. Existing exact
`{text, start_char, end_char}` annotations still work. Missing, conflicting or
out-of-range annotations fail explicitly; an incorrect character range is never
silently repaired using a text search. This avoids asking the model to count
hundreds of characters while retaining the intended occurrence in the same call.
The original annotation and resolved range are private audit fields.

To measure local processing independently of all API calls:

```bash
uv run python -m experiments.tau_voice.clarification_benchmark \
  --source-wav /path/to/complete-clean-tts.wav \
  --noise-path /path/to/dns-noise.wav --seed 42 \
  --warmup 50 --iterations 500 --output /tmp/acoustic-kernels.json
```

This runs both actual kernels over 1, 10 and 60 second waveforms with 0.1 and
0.4 second target spans. It preloads noise once, measures warm calls including
full-waveform copy, local RNG and operation metadata, and saves every duration,
per-case p50/p95/max, hardware, source hashes and conditions. Noise preparation
is a separate cold observation; Python imports are excluded. Alignment and
TTS postprocessing timings remain in each conversation's private input record.
No external project is imported at runtime, and no API key is read by this
benchmark. There is no numerical latency SLA or claimed speedup against the old
Torch implementation.

The conversation preview normalizes both tracks to their highest sample rate
before concatenation and padding. This permits 16 kHz user PCM beside 8 kHz native
agent output without stretching either timeline. Original source and provider
input recordings remain separately auditable.

The [2026-11-02 validation record](acoustic-clarification-validation.md) reports
all real attempts, excluded inputs, actual no-query outcomes, a pending semantic
candidate-selection defect and the local kernel distributions. It distinguishes
service success, model behavior and the remaining evaluator review.
