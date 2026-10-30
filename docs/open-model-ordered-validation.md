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
