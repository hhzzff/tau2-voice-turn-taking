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

**Acoustic clarification task**:
A conversation task in which selected words in user speech are acoustically degraded and the agent can request clarification. The user simulator knows its intended meaning but is unaware of the degradation; the evaluator knows the targeted words and applied degradation.

**Dynamic user simulation**:
A conversation in which the user's next action and speech can respond to the agent output actually received during that run.

**Controlled audio replay**:
A diagnostic conversation input whose audio sequence is fixed in advance and does not change in response to the current agent's behavior.

**Agent interruption of user**:
An attempt by the agent to take the floor while the user is still speaking. Overlapping audio alone does not establish that the user yielded.

**User interruption of agent**:
An attempt by the user to take the floor while the agent is still speaking. It is distinct from the agent interrupting the user.

**Backchannel**:
A brief listener acknowledgement that does not seek to take the floor.

**User speech cancellation**:
Withdrawal of the user's remaining unheard speech after yielding the floor. Speech already heard by the agent remains part of the conversation.
