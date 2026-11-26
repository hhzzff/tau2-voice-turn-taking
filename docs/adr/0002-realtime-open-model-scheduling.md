# Use 100 ms asynchronous scheduling for open models

**Status:** Accepted. Supersedes [ADR 0001](0001-frozen-simulation-time-for-open-model-mvp.md).

The registered SALMONN and MiniCPM agents and their demo use 100 ms scheduler
windows while inference consumes the model’s native audio blocks in the
background. The frozen mode is retired from those entry points; historical
comparison scripts can still reproduce the synchronous baseline.

User generation can temporarily stop new audio submission. On return, overdue
windows may advance without added waiting; the scheduler does not inject audio
to fill that wall-clock gap.
