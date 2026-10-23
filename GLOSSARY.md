# τ-voice Simulation

## Language

**Simulation tick**:
A fixed interval of conversation audio time represented by one scheduler step. Its duration is distinct from the time needed to compute that step.
_Avoid_: wall-clock interval

**Simulation time**:
The conversation timeline advanced by completed simulation ticks.

**Frozen simulation time**:
A clock policy in which simulation time waits for the current step's computation to finish, then advances by one tick duration regardless of computation time.

**Tick computation time**:
The elapsed wall-clock time required to complete one simulation step. It is recorded separately from conversation audio time.
