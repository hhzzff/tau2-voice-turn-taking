"""Audio-only observation state, with no target connection or future transcript."""

from pydantic import BaseModel, ConfigDict, Field


class HeardItem(BaseModel):
    """Playback lifecycle and sample counters for one assistant audio item."""

    model_config = ConfigDict(strict=True, extra="forbid")
    item_id: str
    response_id: str
    for_turn: int
    played: int = 0
    buffered_samples: int = 0
    started_ns: int | None = None
    ended: bool = False
    cancelled: bool = False
    complete: bool = False


class PlaybackObservation(BaseModel):
    """Only lifecycle metadata is shared; unplayed audio stays in the driver."""

    model_config = ConfigDict(strict=True, extra="forbid")

    model: str = ""
    items: dict[str, HeardItem] = Field(default_factory=dict)
    responses: dict = Field(default_factory=dict)
    last_voice_ns: int = 0
    last_end_ns: int = 0
    continuous_duplex: bool = False
    causal_replies: bool = False

    @property
    def speaking(self):
        """Whether an item has started local playback and has not yet ended."""
        return self.audible_item is not None

    @property
    def audible_item(self):
        """Return the first started, unfinished item without exposing its buffered audio."""
        return next(
            (
                i
                for i in self.items.values()
                if i.started_ns is not None and not i.ended
            ),
            None,
        )
