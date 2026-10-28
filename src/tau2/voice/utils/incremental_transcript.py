"""Commit stable ASR additions without replaying revised rolling windows."""

import re
from difflib import SequenceMatcher

from pydantic import BaseModel, Field


class IncrementalTranscript(BaseModel):
    """Keep the prior window and its committed boundary in word coordinates."""

    previous: list[str] = Field(default_factory=list)
    committed: int = 0

    def update(self, text: str, *, final: bool = False) -> str:
        """Emit additions confirmed in two windows, or flush on speech end.

        Revisions before the committed boundary stay in the separately recorded
        full ASR snapshot. They are not replayed as fresh assistant speech.
        """
        # Chinese ASR commonly omits spaces; treating a whole window as one
        # word prevents any stable prefix from being committed while speaking.
        spans = list(re.finditer(r"[\u3400-\u9fff]|[^\s\u3400-\u9fff]+", text))
        words = [span.group() for span in spans]

        def normalize(values):
            return [re.sub(r"\W+", "", w).casefold() for w in values]

        blocks = SequenceMatcher(
            None, normalize(self.previous), normalize(words), autojunk=False
        ).get_matching_blocks()
        boundary = 0
        for block in blocks:
            if not block.size:
                continue
            if block.a <= self.committed <= block.a + block.size:
                boundary = block.b + self.committed - block.a
                break
            if block.a >= self.committed:
                boundary = block.b
                break
            boundary = block.b + block.size
        stable = max((b.b + b.size for b in blocks if b.size), default=0)
        end = len(words) if final else stable
        end = max(boundary, end)
        addition = (
            text[spans[boundary].start() : spans[end - 1].end()]
            if end > boundary
            else ""
        )
        self.previous = words
        self.committed = end
        return addition
