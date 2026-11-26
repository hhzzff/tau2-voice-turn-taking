# Controlled local voice user

This experiment preserves the local staged scenario format and its executor.
The 19 scenes exercise turn taking, interruptions, information disclosure and
listening behavior. They do not implement native tau domain tasks or rewards.

## Ownership and dependency direction

- `scenarios.py`, `agenda.py`, `policy.py`, `scenario.py`, `scene_audio.py` and
  `llm.py` own the frozen scene definitions, stage progression, triggers and
  private-fact disclosure. Stage completion is distinct from semantic success.
- `runtime.py`, `local_asr.py`, `backends.py`, `streaming.py` and `feedback.py`
  coordinate played-audio recognition and the local B/TTS services.
- `participant.py` implements tau's `FullDuplexUser`. The experiment entry point
  explicitly registers it with `tau2.registry`; importing core tau does not
  import or register this experiment.
- `driver.py` owns the real-time send/playback/receive tasks. `target.py` adapts
  provider events and queues unplayed audio. Both are specific to this experiment.
- `recording.py` preserves timing and delivery evidence. Shared provider API
  calls and PCM conversion remain in `tau2.voice.audio_native`.

Core tau does not import this package. Native tau tasks and the existing dynamic
duplex experiment keep their own task interpretation and progression rules.

## Audio and lifecycle contract

1. `start()` initializes ASR before the live clock. `begin_live()` starts the
   scenario and ASR workers after the target session is acknowledged.
2. `begin_agent` / `agent_frame` / `end_agent` describe audio actually released at
   the playback clock. Server transcripts and unplayed audio are audit data only.
3. `poll_output()` returns one PCM16/16kHz frame (at most 20 ms of source audio).
   It does not imply delivery. A frame must be acknowledged by its exact sequence
   after successful transport submission before another can be polled.
4. `acknowledge_delivery()` advances spoken-text and private-fact accounting.
   It confirms client transport submission, not remote hearing.
5. The driver cancels and joins its clock/receiver tasks. The runner then ends
   played utterances, closes the runtime, disconnects the provider and closes
   the HTTP client. The runtime stops generation, drains already-played audio
   when possible and joins native ASR work before releasing handles. The recorder
   closes after partial tracks and the final summary have been exported.

The local runtime intentionally uses real elapsed time: model slowness can miss
an interruption window. It does not freeze time during B, ASR or TTS inference.

## Running and verifying

See [the experiment entry points](../README.md) for installation and commands,
and [the integration guide](../../../../docs/local-user-integration.md) for
recordings, provenance and acceptance criteria. Use Python 3.12 as pinned by the
repository. Model assets and services are external; runtime Python source is
entirely in this repository.

```bash
uv run pytest tests/test_local_user_cli.py tests/test_streaming/test_local_user_*.py
make check-all
```
