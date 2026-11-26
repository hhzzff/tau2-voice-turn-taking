"""Audio-only observation state, with no target connection or future transcript."""

from dataclasses import dataclass, field


@dataclass
class HeardItem:
    item_id: str
    response_id: str
    for_turn: int
    played: int = 0
    buffered_samples: int = 0
    started_ns: int | None = None
    ended: bool = False
    cancelled: bool = False
    complete: bool = False


@dataclass
class PlaybackObservation:
    """Only lifecycle metadata is shared; unplayed audio stays in the driver."""

    model: str = ""
    items: dict[str, HeardItem] = field(default_factory=dict)
    responses: dict = field(default_factory=dict)
    last_voice_ns: int = 0
    last_end_ns: int = 0
    continuous_duplex: bool = False
    causal_replies: bool = False

    @property
    def speaking(self):
        return self.audible_item is not None

    @property
    def audible_item(self):
        return next(
            (
                i
                for i in self.items.values()
                if i.started_ns is not None and not i.ended
            ),
            None,
        )
