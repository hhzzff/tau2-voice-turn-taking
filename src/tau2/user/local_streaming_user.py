"""FullDuplexUser adapter for the local continuous, audio-only simulator."""

from tau2.data_model.audio import AudioEncoding, AudioFormat, audio_bytes_to_string
from tau2.data_model.message import UserMessage
from tau2.user.user_simulator_base import FullDuplexUser, UserState

LOCAL_PCM = AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=16000)


class LocalStreamingUser(FullDuplexUser):
    """Nonblocking chunk API; explicitly started by the continuous driver.

    The ordinary synchronous builder cannot silently use this participant:
    callers must start the runtime and acknowledge every submitted packet.
    """

    def __init__(self, runtime, instructions=None, tools=None):
        super().__init__(instructions=instructions, tools=tools)
        self.runtime = runtime
        self.packet = None

    def get_init_state(self, message_history=None):
        """A trial starts empty; private scene state lives in the runtime."""
        if message_history:
            raise ValueError("Controlled local trials require a fresh session")
        return UserState(system_messages=[], messages=[])

    def get_next_chunk(self, state, participant_chunk=None, tool_results=None):
        """Consume only played PCM and return a ready frame, including silence."""
        if tool_results:
            raise ValueError("Controlled scenes have no domain tools")
        if participant_chunk is not None:
            self.observe_played(participant_chunk)
        self.packet = self.runtime.poll_output()
        return UserMessage(
            role="user",
            content=None,
            audio_format=LOCAL_PCM,
            audio_content=audio_bytes_to_string(self.packet["pcm"]),
            chunk_id=self.packet["sequence"],
            is_final_chunk=False,
            contains_speech=self.packet["voice"],
            source="local_user",
        ), state

    def observe_played(self, participant_chunk):
        """Playback callback, also usable while an outgoing send is blocked."""
        if participant_chunk.audio_content:
            if participant_chunk.audio_format != LOCAL_PCM:
                raise ValueError("Played observations must be mono PCM16/16kHz")
            item_id = participant_chunk.utterance_ids[0]
            item = self.runtime.playback.items[item_id]
            self.runtime.agent_frame(
                item,
                participant_chunk.get_audio_bytes(),
                participant_chunk.raw_data["played_ns"],
            )

    def acknowledge_delivery(self, sequence, sent_ns):
        """Host callback after successful client transport submission."""
        self.runtime.acknowledge_delivery(sequence, sent_ns)

    def stop(self, participant_chunk=None, state=None, tool_results=None):
        """Request shutdown; the owning async driver awaits runtime.close()."""
        self.runtime.stop = True
