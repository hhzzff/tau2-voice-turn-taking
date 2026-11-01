# Shared English protocol and player fixes — 2026-10-30

## Player diagnosis and verification

The original player received pointer events but attempted to seek when the
browser's seekable range was `[0,0]`. The target was cleared even though the
browser clamped it back to zero. This reproduced with both main recordings,
including MiniCPM, using real desktop mouse and emulated mobile touch input.

Changing only the diagnostic HTTP server to provide byte ranges made the old
player pass. The repaired player loads a complete audio Blob, retains pending
seek targets until the media can seek, and caches sources for channel switches.
It no longer depends on the host providing range requests. The repaired player
passed on the original server without byte-range support:

- SALMONN and MiniCPM, desktop mouse and mobile touch: 20% → 80% drag reached
  approximately 144s in each 180s recording.
- Switching user/model tracks preserved playback time.
- User/model first-activity markers were present; first-user-audio buttons
  sought and played the corresponding point.
- No browser page errors were observed.
- Versioned audio, waveform and transcript requests were checked, preventing
  stale resources from different recordings being mixed after an update.

These are real Chromium browser checks including touch emulation, not physical
phone testing. The first-activity marker uses 100ms envelope bins and is not a
word-alignment claim. Audio is not cropped or shifted to disguise a delayed
opening in the retained older examples.

The UI fix was published to the existing owner-private Site at source commit
`84046affb3fdba9d1342f7e00e36d3a069150735`, deployment
`appgdep_6ae488fedadc8191a0403ed44c1b925e`. The page explicitly marked the old
recordings as pending replacement by the shared-protocol recordings.

## Shared protocol implementation

Controller source is frozen at `e5e132a99ad6a712502a2efdae61e983a2c23286`.
See [protocol definition](open-model-ordered-conversation.md). The existing
FullDuplexOrchestrator step boundary, native model widths, generation/TTS,
ASR, output chunking and frozen clock are retained.

Two native-width orchestration regressions first failed because the ordered
scenario was absent, and passed after implementation. Review found a conflicting
free-dialogue continuer classifier and a pause-after-backchannel stall. The
stage-specific classifier now allows a recent user turn and one completed useful
clause. Verified heard evidence survives a brief acknowledgement, so the longer
user turn can begin if the model finishes speaking during that acknowledgement.
An orchestration regression with a model becoming permanently silent at that
point failed at both widths before the repair and passed afterwards.

The final relevant regression group passed **29 tests**. `make check-all` passed;
only its unrelated baseline formatting edits were restored. Standards review
found the timing defaults outside central config and an unused stage timer;
both were corrected. Spec review's two functional blockers were corrected and
rechecked.

Real Luna listener canaries (synthetic text fixtures, not model audio) returned
YES for a completed useful clause after a recent correction, and NO for an
unfinished phrase. These establish classifier operation only. Actual model
stage completion, language compliance, interruption handling and backchannel
quality require the real recording artifacts.

## Recording provenance

Artifacts are retained under
`data/simulations/open-model-ordered/20261030-run1/`. No credentials are included
in the native worker snapshot. Failed startup attempts and queue state are
retained; a queued task is not treated as a successful inference run.


At the last recording check, job `8a2f4d76dd9441d9`, trial `442885904`, was
still queued on one A100 80GB in `gpuspot1-lq`; no worker pod existed and no
current alarm was reported. Earlier attempts failed because mounted volumes
were inaccessible from the selected clusters. The clean current request has
only the four needed LQ volumes. The queue delay has no confirmed root cause.

Following the user request to keep queueing, the local chain now waits for
worker readiness before starting each bounded native connection. The queue
wait is bounded at 24 hours. Its driver writes STOP markers for both workers
on failure or completion;
the GPU entrypoint also has a 90-minute wall timeout. Recordings and final
publication remain pending until a native worker can start. This report does
not establish real English compliance or completion of the shared stages.


## 2026-11-01 recovery

The replacement `cloudnative-lq` A100 job `386e698abb49afad` reached running
and finished; the old `gpuspot1-lq` queued job was stopped. SALMONN executed all
2249 steps plus the initialization interval (180 seconds), but provenance
collection raised before export: a symlinked source path was combined with a
resolved checkout root. This prevented the sequential MiniCPM controller from
starting. The GPU job is terminal (`done` / `finished`).

The source hash function now constructs the controller path from the same
resolved root as the remaining entries. A symlinked-snapshot regression failed
before the fix; it and the existing audio export regression passed afterwards
(2 tests). The original snapshot and mailbox evidence are unchanged.

SALMONN playback was recovered under
`data/simulations/open-model-ordered/20261101-run1/recovered-salmonn/`.
All 2249 native user inputs passed a byte-for-byte forward-conversion check
after inverse recovery. The recovered stereo includes the initialization
interval; the final silent user interval is verified in the controller log.
Missing original in-memory tick objects, ASR snapshots and computation durations
are explicitly recorded. These are recovered diagnostic recordings, not a
successful original export.

The native trace contains one assistant start and one listening transition.
Only greeting and correction user utterances were generated; subsequent
backchannel stages were not reached. A completed 180-second inference is not
completion of the requested long conversation.

MiniCPM is being run independently in job `294cd3c18536bb1f`, using the original
frozen scenario and model code with a canonical controller import path. Its
artifacts are under `data/simulations/open-model-ordered/20261101-minicpm-resume1/`.
