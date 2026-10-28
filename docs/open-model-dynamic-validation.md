# Dynamic dialogue validation — 2026-10-28

## Scope and provenance

This run replaces short scripted examples with the original τ-voice user
turn-taking policy, free-form LLM-generated utterances, Doubao TTS and ASR, and
native open-model inference on Merlin. The travel background is predefined;
assistant replies and user utterances are not prerecorded or scripted.

- Controller: `efa802c50a8342282ef4a9e7055d393bc81e12ee`.
- Native worker: `4094a00c831a454e4358a01b77b2309463eb8c08`;
  worker source bytes were verified identical to the controller snapshot.
- Merlin job: `7ea7c02a3bf95de5`, one A100 80GB.
- Models: SALMONN before-SFT checkpoint-44000 (80ms), MiniCPM-o-4.5 (1000ms).
- Dialogue artifacts: `data/simulations/open-model-dynamic/20261028-run3/`.
- Controlled replay artifacts: `data/simulations/open-model-dynamic/20261028-run1/replay/`.
- Pacing: frozen simulation time, previous-tick participant audio exchange.
- User LLM: `codex/gpt-5.6-luna`; listener classifiers constrained to YES/NO,
  normal utterances unrestricted. Listener checks are sampled every four
  simulated seconds, so a short semantic opportunity can be missed.

## SALMONN investigation

The old worker omitted actually played self audio. The native evaluation
supports supplying that signal. The repaired worker mixes the previous played
80ms block into the corresponding incoming stream; it never supplies queued
future audio and does not force a stop when the user starts talking.

The controlled replay uses the same fixed 40-second user input, prompt and
fresh model state per condition. These are diagnostic replays, not dynamic
conversations or a population-level interruption benchmark.

| Condition | Native listening transition | Cancelled unplayed audio |
| --- | ---: | ---: |
| Old input, correction present, echo=0 | 37.76s | 0s |
| Repaired input, correction present, echo=1 | 15.76s | 0.76s |
| Repaired input, correction removed, echo=1 | 15.28s | 0.60s |

The cancellation path works, but the no-correction control switches even
earlier. These observations do **not** establish that the user's correction
caused the transition or that natural interruption has been fixed. They apply
to this before-SFT checkpoint, prompt and input only.

Earlier trajectory files omitted raw native events during serialization.
Empty arrays in those old exports cannot establish absence of native stops.
This run retains JSON-safe native events; the controlled replay saved native
worker responses directly.

## Verification

- Red/green regression for actually played SALMONN self feedback; native CPU
  contract covers two fresh sessions and 32 causal ticks.
- Strict listener classification regression: 28 tests passed; real Luna
  classifier canary returned exactly YES.
- Native-event serialization regression group: 31 tests passed.
- Earlier dynamic regression group: 57 passed, one skipped. Groups overlap and
  must not be added together.
- Provider suite: zero failures/errors, 77 skipped (live provider gates were
  disabled); skipped tests are not live provider verification.
- `make check-all` passed. Two skill-directed reviewers checked standards and
  spec compliance; their actionable findings were addressed before recording.

## Rejected recordings

`20261028-run1/dialogues/salmonn` is a deliberately stopped diagnostic recording:
the default customer-service role incorrectly asked for account information.
`20261028-run2/dialogues/salmonn` is a deliberately stopped diagnostic recording:
listener decisions could return prose and be silently interpreted as NO.
The final controller gives an explicit travel-concierge role and constrains
listener decisions. Both rejected runs remain available for audit and are not
presented as successful final examples.

## Final SALMONN recording

The 180-second recording contains 2250 native ticks and nine substantive user
utterance generation actions, not nine completed successful exchanges. Three
utterance playbacks completed, five yielded to the speaking model, and the last
was cut by the recording limit. The long preference/comparison utterance begins
at 95.68s and played for 27.20s before yielding. Actual energy-detected user audio
is 94.46s, model audio 44.14s and overlap 7.62s. These use 20ms PCM16 RMS frames
with threshold 128; they are not semantic backchannel scores.

The recording contains six native listening transitions, three with nonzero
cancelled pending output. A transition is not automatically a user-caused
interruption: ordinary utterance completion and model control behavior also
produce transitions. The user requested no backchannels in this recording.
The dialogue contains persistent follow-ups about unmet travel constraints and
premature conversation closure; it is not a demonstration that the travel task
was solved.

## Final MiniCPM recording

The 180-second recording contains 180 native ticks, four substantive utterance
generation actions and one user backchannel. Three substantive utterance
playbacks completed; the fourth was cut by the recording limit. User audio is
41.94s, model audio 110.64s and overlap 5.64s under the same energy detector.
Long assistant turns limited the number of exchanges; the eight-turn generation
goal was not achieved by this model within the recording limit.

At 47s the listener requested `mm-hmm` after the heard ASR window described
visiting the Forbidden City and its historical interest. Its two-second playback
completed, with one second of detected user/model overlap. This is an observed
user backchannel request and playback, not a scored claim of optimal timing.
Later user questions compare stairs, cable-car access, walking and costs.
The conversation switches between Chinese and English dynamically.

No semantic success score or confirmed model-backchannel count is asserted for
either recording. Model output during user speech can also be an ordinary
answer or unwanted overlap. Actual heard snapshots and separate audio channels
remain available for inspection.

## Artifact checks and release

Both recordings passed checks for completed export, 8kHz PCM16 stereo audio,
1,440,000 frames per recording, tick count and duration, previous-tick opposing
audio hashes, nonempty user/model audio, and ASR heard-time bounds. Final
exports contain 9007 SALMONN and 179 MiniCPM JSON-safe native diagnostic events.
The full machine-readable result is `20261028-run3/validation.json`; it also
records per-utterance playback extent and completion.

The existing mobile Site keeps its owner-private audience. Its new home page
contains the two long recordings with separate user/model playback, action
seek buttons, played user text and heard ASR snapshots. Three fixed-input
SALMONN diagnostic recordings are separate. Fifteen audio assets, copied hashes,
HTML source links and mobile viewport metadata were checked. No physical-phone
or interactive browser verification was available.

Merlin job `7ea7c02a3bf95de5` reached `done` / `finished` after both controllers
completed, releasing its GPU. The terminal response is retained as
`20261028-run3/merlin-final-state.json`.

## Chinese incremental-history follow-up

Recording in Chinese exposed a word-boundary limitation in the incremental ASR
helper: a window without spaces could remain uncommitted until speech ended.
The final code splits Chinese characters for stable-prefix alignment while
preserving the exact source spacing. A regression first failed on the old
implementation and then passed with the repair.

The recordings above use frozen controller `efa802c`; this follow-up was made
after recording and was not used for these audio files. During the recordings,
listener decisions and user utterance generation still received the complete
latest heard ASR window through the authoritative snapshot context, so the
recorded reactions were based on actual audio. Chinese incremental history
could lag, and the recordings must not be presented as validating this later
repair in a full real-model run.

## Published mobile page

[Open the dynamic dialogue player](https://tau-voice-open-model-demo-20261024.tterumiimurett.chatgpt.site).
The first publication attempt returned `Unauthorized`; retrying deployment of
the same saved version succeeded. Deployment
`appgdep_6ae202a50f508191b1f796c47d5b7dd2` published Site source commit
`38187b7390c170b259a5c16230bb84eac6e1c5f9`. The native terminal response verifies
publication; it does not establish physical-phone playback testing.

The final Chinese/English incremental-history regression passed both focused
cases (`2 passed`). A broader repeat reached the adapter boundary and was
stopped when its runtime subprocess stalled; it is not counted as a passed
suite. The earlier adapter/serialization regression above remains the evidence
for that unchanged adapter code.

Final `make check-all` passed after restarting a stalled check runner. The
formatter touched three unrelated baseline files; only those formatter edits
were restored, preserving the scoped change.
