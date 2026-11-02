# Acoustic clarification validation: 2026-11-02

This snapshot separates real service/model evidence from controlled mechanism
checks. Persistent local artifacts are under
`/mnt/bn/audio-visual-llm-data6/terumi/playground/tau2-clarification-integration/data/simulations/acoustic-clarification-validation-20261102`. They include all four conversation attempts, their actual audio and
ASR, source/provider input formats, private alignment/degradation records,
post-conversation assessments, model calls and per-case kernel durations.
`validation-summary.json` and `waveform-audit.json` index the evidence. Generated
audio and credentials are not checked into Git.

## Conversation outcomes

| Trial | Input | Conversation | Actual observed behavior | Evaluation at this snapshot |
| --- | --- | --- | --- | --- |
| English dropchunk attempt01 | Invalid keyword character offsets; no alignment | max_steps, 90 seconds | Excluded from acoustic behavior population | Not evaluated |
| Chinese DNS attempt01 | Invalid keyword character offsets; no alignment | max_steps, 90 seconds | Excluded from acoustic behavior population | Not evaluated |
| English dropchunk attempt02 | Valid; birdwatching, 4.50–5.08 s | user_stop | Pocket bird guide suggested; no clarification request | Supported inference, pass |
| Chinese DNS attempt02 | Valid; 观鸟, 1.28–1.70 s | user_stop | Mini-tripod suggested; no clarification request | Candidate-selection defect; review fix pending |

Attempt01 also exposed the preview exporter's mismatch between 16 kHz user PCM
and decoded 8 kHz agent audio. Both failures remain preserved. Explicit same-call
occurrence indices and mixed-rate preview export were subsequently added with
regressions. Attempt02 used ordinary gift-advice background goals and concise user
style, with no fixed utterance, degradation hint or prescribed agent question.
The 180 second ceiling was not a trigger to force completion. No additional
conversation was generated to obtain a clarification request or a favorable score.

The valid population is **2 conversations**, both naturally complete; there were
**0 actual clarification requests and 0 actual repair sequences**. The other
2 attempts are explicit invalid-input exclusions. Controlled tests cover both
interrupting and after-user clarification, queue cancellation and clear repair;
these are separate from the real-model population.

At this snapshot the Chinese discovery judge chose an acknowledgment before the
later substantive recommendation, causing a misleading `no_response` assessment.
This is an evaluator correctness defect pending review, not proof that the agent
failed to answer. Its raw ASR and judgment are retained. Final semantic acceptance
must wait for that fix. Initial semantic calls also exposed the gateway JSON-mode
requirement: the prompt must explicitly contain “JSON.” The provider returned
code `-4003`, parameter `messages`. The instruction was fixed and the same frozen
ASR/prefix evidence was rejudged, without repeating TTS, alignment, ASR or an agent
session. Original assessments are `assessment-before-json-fix.json`.

Early progress updates inferred successful alignment from waveform files; this
was incorrect for attempt01. Its private records show zero alignment time and
`keyword_text_position_mismatch`. Attempt02 contains actual service responses and
resolved target ranges. No success claim rests on file existence alone.

## Audio and service checks

Independent waveform inspection found identical clean/degraded sample counts
(English 113,685; Chinese 140,434 at 16 kHz) and **zero changed samples outside the
target**. Dropchunk changed 7,831 samples using two permitted overlapping gaps;
DNS changed all 6,720 target samples. The actual DNS preclip RMS ratio is 10:1,
within floating point tolerance, and clipping is separately recorded. Neither
hard label guarantees unintelligibility or a need to ask.

Clean-TTS Scribe checks independently recognized the original words birdwatching
and 观鸟. These QC transcripts are separate files and never enter delivered-context
judgment. English damaged-audio ASR still recognized bird watching; Chinese
damaged-audio ASR recognized 逛街. ASR is fallible and these transcripts do not
establish human perceptual accuracy or exact alignment error.

Standalone service canaries cover English and Chinese TTS, Scribe and alignment.
The English tutorial audio was reused without new requests; the Chinese canary
made one short call to each endpoint. Provenance and unmodified responses are in
`service-canaries/`. Both full attempt02 runs used real generated complete TTS,
real alignment, actual native WebSocket audio exchange and final after-conversation
assessment, all via `TERUMI_BT_AK`. The user/judge response version recorded in
model logs is `gpt-5.6-luna-2026-07-09`; the requested native model is
`gpt-realtime-2.1`. Gateway token pricing is unknown, not reported as free.

## Local kernel measurements

CPU: Intel(R) Xeon(R) Platinum 8336C CPU @ 2.30GHz; Linux x86_64; NumPy 2.3.5.
The host exposes 128 logical CPUs and the process affinity
allows 122. Operations are sequential CPU NumPy calls.
For every row: 50 warmups, 500 measured calls, seed 42 plus iteration index.
Input is a repeated genuine English TTS waveform; the actual mounted DNS WAV is
preloaded once. Timing includes full waveform copy, RNG setup, corruption and
returned operation metadata; excludes alignment, TTS, filesystem artifacts and
noise preparation. Percentiles use NumPy linear interpolation. Raw durations,
source/asset hashes and exact conditions are in `kernel-benchmark.json`.

| Mode | Waveform s | Target s | p50 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| dropchunk | 1 | 0.1 | 0.0264 | 0.0334 | 0.0427 |
| dns_noise | 1 | 0.1 | 0.0890 | 0.1028 | 0.3437 |
| dropchunk | 1 | 0.4 | 0.0269 | 0.0346 | 0.0593 |
| dns_noise | 1 | 0.4 | 0.1047 | 0.1208 | 1.8119 |
| dropchunk | 10 | 0.1 | 0.0360 | 0.0464 | 0.0790 |
| dns_noise | 10 | 0.1 | 0.1080 | 0.1306 | 0.1862 |
| dropchunk | 10 | 0.4 | 0.0359 | 0.0437 | 0.0704 |
| dns_noise | 10 | 0.4 | 0.1260 | 0.1399 | 0.2332 |
| dropchunk | 60 | 0.1 | 0.1656 | 0.1763 | 0.2210 |
| dns_noise | 60 | 0.1 | 0.2428 | 0.2643 | 0.3304 |
| dropchunk | 60 | 0.4 | 0.1665 | 0.1783 | 0.2462 |
| dns_noise | 60 | 0.4 | 0.2573 | 0.2744 | 0.3438 |

One cold noise preparation observation was 0.499
seconds, excluding Python imports. This is not a cold latency distribution.
Actual attempt02 alignment calls took 0.611 seconds (English) and 1.968 seconds
(Chinese); local degradation took 0.366 ms and 0.651 ms respectively. TTS
postprocessing totals were 0.824 and 2.225 seconds. These are individual
observations, distinct from the warmed distributions and simulation time.
There is no agreed millisecond SLA, and no old-Torch speedup factor is claimed.

## Automated checks

The final targeted regression covers actual simulator ticks/native input, final
assessment, both hard modes, existing duplex events and audio effects. External
services are replaced only in default automated tests. The standalone provider
suite ran as required: **0 passed, 77 skipped, 0 errors** because its official-key
gates do not recognize TERUMI credentials. Those skips are not provider behavior
validation; the full native gateway trials supply that separate evidence.
