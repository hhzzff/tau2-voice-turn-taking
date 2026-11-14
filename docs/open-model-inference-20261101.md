# Open-model inference results — 2026-11-01

Both requested native inference runs are finished and their one-A100 jobs are
terminal (`done` / `finished`). This completes inference execution; SALMONN did
not complete the requested conversational behavior.

| Model | Native width | Recorded time | Intervals | Outcome |
| --- | --- | --- | --- | --- |
| SALMONN before-SFT, checkpoint-44000 | 80ms | 180s | 2250 | Greeting and user correction; no further reply after listening transition |
| MiniCPM-o-4.5 | 1000ms | 180s | 180 | All five user stages requested; 6 utterance generations and 1 user backchannel |

## MiniCPM

The shared user stages occurred at 0s (greeting), 8s (interruption), 28s
(backchannel), 33s (long user turn), and 87s (first dynamic follow-up).
Further follow-up requests occurred at 101s and 141s. Activity measured on the
played tracks totals 76.3s for the user, 108.5s for the model and 34.1s overlap.
These figures use 10ms PCM16 RMS activity frames at threshold 200; overlap does
not establish successful interruption, correction acceptance or a natural model
backchannel.

The exported 180 rows preserve previous-interval handoff hashes, frozen pacing,
179 native generation events and bounds on 140 incremental ASR events. The
stereo is 8kHz PCM16, 1,440,000 frames. These checks establish scheduling and
recording integrity, not semantic success.

## SALMONN

The native run executed 2249 steps plus the initialization interval. Its
post-run source-hash export failed on a symlinked checkout path. The exporter
has been repaired and its symlink regression and existing export regression
passed. Original run evidence was retained unchanged.

The stereo was recovered from native requests/responses. The inverse recovery
of every one of the 2249 user inputs passed byte-for-byte forward conversion;
model output used the same streaming PCM16-to-telephony conversion as the
adapter. The final silent user interval was checked in the controller log.
Original in-memory Tick objects, incremental ASR snapshots and computation
fields are unavailable and are not fabricated.

The user starts at 0s and requests the correction at 7.04s, after actual model
speech begins. The trace contains one assistant start and one native listening
transition at 11.12s. Subsequent stages were not triggered because there was no
new model reply. The recovered audio is a diagnostic result, not a successful
long-conversation demonstration.

## Artifacts and playback

- SALMONN: `data/simulations/open-model-ordered/20261101-run1/recovered-salmonn/`
- MiniCPM: `data/simulations/open-model-ordered/20261101-minicpm-resume1/dialogues/minicpm/`
- Audit and presentation: `data/simulations/open-model-ordered/20261101-comparison/`

The comparison includes exact recording SHA256s, input provenance, model phase
times, native evidence and sentence ASR offsets. Played audio was transcribed
in bounded segments after full-file upload timed out; original offsets were
retained. Every displayed sentence is bounded by the audio duration and overlaps
actual activity. ASR includes occasional phonetic/multilingual recognition
errors; it is not proof of uniform spoken-language compliance.

The existing private demonstration page is updated with the current recordings,
dual activity lanes, seeking, track switching, onset buttons, sentence captions
and a shared stage comparison. Missing SALMONN stages are explicitly marked.
Browser verification uses real Chromium desktop interaction and mobile touch
emulation, not a physical phone.

Published private demonstration: [play recordings](https://tau-voice-open-model-demo-20261024.tterumiimurett.chatgpt.site).
Deployment `appgdep_6ae7027f3d4c8191a3beb413d8d85599` succeeded from Site source
`d47c36b232d12812042cef1404b0e90ae5149b74`.
