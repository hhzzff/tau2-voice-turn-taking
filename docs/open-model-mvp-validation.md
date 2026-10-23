# Open-model frozen-time MVP validation

Validated on 2026-10-23 on branch `openFDB-time-froze-MVP`.

## Frozen versions

- Controller: `37fc5b75d275e5d2feb6b8635fefa3b5bfde03ad`.
- GPU worker: `032d6bfd96aaf4a703b30316e3c777bf98365f88`; all worker source files are byte-identical to the controller snapshot.
- Merlin job: [9a801cb75ede16b7](https://ml.bytedance.net/development/instance/jobs/9a801cb75ede16b7), one A100 80GB, models run sequentially.
- User: GPT 5.6 Luna through the local Codex login; Doubao TTS and Doubao ASR. Credentials stayed on the controller. The job finished normally and released its GPU.

## Real conversation matrix

Each run uses an independent model session and the existing full-duplex orchestrator.
The requested limit is 40 simulated seconds; normal conversation termination can end a run earlier.
Trigger observations measure the documented scheduling condition, not semantic success.

| Model | Scenario | Simulated seconds | Ticks | Observed trigger | Total audio overlap seconds |
| --- | --- | ---: | ---: | --- | ---: |
| salmonn | qa | 23.68 | 296 | False | 0.00 |
| salmonn | correction | 40.00 | 500 | True | 3.58 |
| salmonn | user_backchannel | 40.00 | 500 | True | 6.06 |
| salmonn | model_backchannel | 40.00 | 500 | True | 3.60 |
| minicpm | qa | 40.00 | 40 | True | 0.00 |
| minicpm | correction | 40.00 | 40 | True | 3.08 |
| minicpm | user_backchannel | 40.00 | 40 | True | 1.26 |
| minicpm | model_backchannel | 40.00 | 40 | True | 0.00 |

All eight runs completed. Mechanical validation checks WAV channel/sample counts, tick lengths, previous-tick input hashes, model durations, and artifact completeness.
Semantic outcomes remain `not_assessed`: overlap alone does not prove a successful correction or a model backchannel.

SALMONN correction evidence: the user asked to revise three days to one day; the correction overlapped the model by 3.58 seconds, but no native playback cancellation was recorded and ASR still contained a three-day response. This is not evidence of successful correction handling.
MiniCPM correction evidence: the user corrected three days to one day, with 3.08 seconds of overlap. The final ASR window contains a one-day itinerary. This is transcript evidence, not a manual listening or semantic score.
SALMONN user backchannel evidence: the user said `mm-hmm`, with 0.90 seconds of overlap within that utterance; total run overlap is larger and includes other speech.
SALMONN QA produced an ASR-recognized answer, then the user selected normal termination before audible follow-up. Its follow-up trigger is correctly false. The terminal stop tick contains no user audio and is padded with silence; the artifact audit permits this only for a final explicit `###STOP###` marker.

MiniCPM model-backchannel opportunity was present, but total overlap was zero: this run does not demonstrate an overlapping model backchannel.

## Artifacts

Local shared-workspace root:

```text
data/simulations/open-model-mvp/20261023-matrix-3
```

- `index.html`: entry point for all eight playable demos.
- `demos/<model>/<scenario>/`: `conversation.wav`, `ticks.jsonl`, `summary.json`, `index.html`.
- `matrix-audit.json`: validation results, audio activity, trigger outcomes and WAV hashes.
- `source.tar`, `SHA256SUMS`, runtime manifests and `worker-reuse.json`: reproducibility evidence.
- The worker bundle and Merlin payload/precheck/job records live in sibling `20261023-matrix-2/`.
- The partial failed first run in `20261023-matrix-2/demos/salmonn/qa/` is retained. Doubao status `20000003` means no detected speech ([official codes](https://docs.volcengine.com/docs/DoubaoVoice/error-code-query?lang=zh)); the controller now records an empty recognition without falsely triggering QA. Genuine API failures still stop the run.

## Checks

- Relevant integrated scheduler/demo/adapter/native tests: 29 passed, 1 skipped before the final ASR repair.
- ASR repair and demo regression: 17 passed, including empty recognition followed by valid speech.
- Native SALMONN CPU contract: two fresh episodes, 32 causal ticks.
- `make check-all` passed for implementation and repairs.
- Provider suite had 77 live-gated skips; these are not real-provider validation. The separate Luna/TTS/ASR canaries and this matrix provide live evidence.

## Independent native canaries

| Model | Merlin job | Simulated duration | Tick computation wall time | Ticks with generated audio |
| --- | --- | ---: | ---: | ---: |
| SALMONN before-SFT | `ad1669ae889500ad` | 16 s (200 × 80 ms) | 38.6624 s | 86 |
| MiniCPM-o-4.5 | `d6b2473c247f768e` | 16 s (16 × 1000 ms) | 8.5025 s | 7 |

These fixed-input canaries establish native transport/audio execution; they are distinct from the live simulator matrix. Startup time is excluded from tick computation wall time.

See [launch and evidence documentation](open-model-demo.md) for the public demo command.
