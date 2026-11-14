# Control signal probes

These scripts exercise real endpoints or real native weights and retain evidence.
They are environment-specific diagnostics, not a replacement for full τ-voice adapter tests.

## Reproduction

Run from the repository root. Each output root needs `stimuli/greeting.wav`,
`interruption.wav`, and `backchannel.wav`. Use only generic synthetic content:

```bash
.venv/bin/python scripts/control_signal_probes/synthetic_stimuli.py /tmp/control-probe/stimuli --language en
.venv/bin/python scripts/control_signal_probes/closed.py --help
.venv/bin/python scripts/control_signal_probes/native.py --help
.venv/bin/python scripts/control_signal_probes/analyze.py --help
.venv/bin/python scripts/control_signal_probes/build_evidence.py --root data/simulations/control-signals --output docs/control-signals-live-evidence.json
```

`closed.py` loads credentials in its process via pi-bench's credential loader;
never copy credentials into GPU bundles or logs. It uses pi-bench's actual endpoint
transport and queue. The stimuli generator makes paid external TTS requests;
closed probes send the synthetic inputs to the selected service.

Native probes must run in the respective model Python environment on a GPU.
The completed Merlin runs under `data/simulations/control-signals/20261112-*`
retain explicit configs, frozen sources, SHA256SUMS, bounded job scripts,
precheck/create results, per-case traces and audio. Reuse those configs rather
than relying on a different environment's model defaults. Raw artifacts are
local and ignored by git; the committed ledger records paths and hashes.
The ledger lists job SIDs and exact checkpoint/runtime provenance.

## Interpretation

Cases include normal speech, interruption after audible model speech,
backchannel, MiniCPM forced listening, and a SALMONN natural-first-turn plus
second-turn interruption. Later probes wait 3 seconds after actual model speech;
Freeze-Omni also separates user turns by 2.5 seconds of silence. Cases preserve
their own probe/config source versions; do not pool them as a success rate.

`status=observed` means audio was received. It does not prove a correction was
understood or an interruption succeeded. The evidence builder excludes failed
cases and rejects stale old playback after cancellation. MiniCPM break/force
and Freeze-Omni reference stop-TTS are external policy, not autonomous model
signals. SALMONN playback completion is controller generated, separately from
its unconfirmed native listening transition. GPT Live/Moshika continuous quiet
frames retain time and have no admitted discrete cancellation signal.

The fixed stimuli are synthetic, not a dynamic user simulator. Native probes
freeze simulated time while inference runs; no wall-clock realtime guarantee
is established. Doubao, Freeze-Omni and Moshi still need full τ adapter integration.
See `docs/control-signals-audit.md` for the exact eight tested model bindings.
