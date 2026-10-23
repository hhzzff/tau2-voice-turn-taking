# Freeze simulation time during open-model inference

The SALMONN and MiniCPM τ-voice MVP waits for each simulation step to finish before advancing conversation time by the configured tick duration. Record actual computation time separately: a one-second tick that takes three seconds to compute still advances the conversation timeline by one second. This makes the demo exercise bidirectional audio scheduling independently of inference speed; its timeline does not establish wall-clock realtime performance.

The first demo covers each model interacting with the τ-voice user simulator, with both audio tracks and per-tick events available to inspect overlap and scheduling. Completing domain tasks with tool calls is outside this MVP's acceptance scope.
