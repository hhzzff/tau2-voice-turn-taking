# Dynamic long conversation

The `long_conversation` scenario uses the original
`VoiceStreamingUserSimulator._next_turn_taking_action()` and its
`basic_turn_taking_policy`, LLM interruption decision, semantic backchannel
decision, generation, TTS, queueing and yielding. A background story specifies
travel constraints and goals; it does not specify the assistant's response or
force a correction at a fixed time. User wording and subsequent topics depend
on audio already played and transcribed.

The default story asks for a multi-turn accessible Beijing trip plan. The user
should negotiate details over at least eight substantive turns and describe
preferences at length in a suitable turn. This is a generation instruction,
not a guarantee that any model will sustain eight successful exchanges.

Listener decisions use the same configured user LLM. Corrections and
backchannels require actual audible model audio, measured in 20ms frames, with
at least one second of accumulated active audio since the user's last substantive
turn. The original policy then decides whether to act based on heard content.
No future model text or queued waveform is exposed to the simulator. Repeated
ASR prefixes are not repeatedly appended to the dialogue.

```bash
python -m tau2.scripts.open_model_demo --model salmonn \
  --scenario long_conversation --seconds 180 \
  --runtime-config /path/to/frozen-runtime.json \
  --mailbox /shared/run/salmonn-mailbox --output /shared/run/salmonn-dialogue
```

Use `minicpm` for the other native runtime. Frozen simulation time and native
80ms / 1000ms durations remain the same.

## SALMONN self-audio feedback investigation

The first worker supplied only the incoming user's audio to SALMONN's acoustic
stream. The native evaluator supports adding actually played self audio to that
stream. The repaired worker supplies the previous played 80ms output alongside
the previous user block, using `settings.echo_ratio` (default 1.0; use 0.0 for
controlled comparison). It never feeds future synthesized output into the model.
Both inputs correspond to the same previous scheduler interval.

The native CPU contract verifies the actual waveform supplied to the frontend,
including output quantization and session reset. Its synthetic model also
predicts a listening transition while output is pending and verifies that
unplayed output is cancelled. This establishes the implementation path, not the
real checkpoint's ability or willingness to interrupt. Real-weight controlled
replay is needed to assess the effect on behavior.
