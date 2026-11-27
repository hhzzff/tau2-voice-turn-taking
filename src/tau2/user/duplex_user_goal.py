"""Private user goals grounded in the conversation actually available now."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GoalEvidence(BaseModel):
    """An available quotation with its speaker and conversation timestamp."""

    model_config = ConfigDict(extra="forbid")
    speaker: Literal["primary", "background", "assistant"]
    seconds: float = Field(ge=0)
    quote: str = Field(min_length=1)


class CurrentUserGoal(BaseModel):
    """The user's assessment before choosing the next action, not a test score."""

    model_config = ConfigDict(extra="forbid")
    objective: str = Field(min_length=1)
    pending_request: str = Field(
        default="",
        description="The request still awaiting resolution; empty when satisfied.",
    )
    status: Literal[
        "not_started",
        "awaiting_answer",
        "answer_in_progress",
        "needs_repair",
        "satisfied",
    ] = "not_started"
    unresolved_issues: list[str] = Field(default_factory=list, max_length=6)
    stance: Literal["neutral", "concerned", "frustrated"] = "neutral"
    evidence: list[GoalEvidence] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def consistent_assessment(self):
        """An unresolved problem cannot also be a satisfied goal."""
        if self.status == "satisfied" and (
            self.unresolved_issues or self.pending_request.strip()
        ):
            raise ValueError(
                "A satisfied goal cannot have unresolved issues or a pending request"
            )
        if self.status == "needs_repair" and not (
            self.pending_request.strip() and self.unresolved_issues and self.evidence
        ):
            raise ValueError(
                "A repair needs a pending request, unresolved issues and evidence"
            )
        return self

    def validate_evidence(self, conversation: list[dict]) -> None:
        """Reject fabricated/future quotes or a background quote labeled primary."""
        for evidence in self.evidence:
            quote = " ".join(evidence.quote.split())
            matches = []
            for row in conversation:
                speaker = (
                    "assistant"
                    if row["role"] == "assistant"
                    else row.get("source", "primary")
                )
                text = " ".join(
                    (row.get("text", "") + " " + row.get("continuation", "")).split()
                )
                if (
                    speaker == evidence.speaker
                    and abs(row.get("seconds", -1) - evidence.seconds) < 0.001
                    and quote in text
                ):
                    matches.append(row)
            if not matches:
                raise ValueError(
                    "Current goal evidence must quote an available utterance with its actual speaker and time"
                )

    def action_feedback(self, action: str) -> str:
        """Keep identified conversation problems ahead of test progression."""
        if self.status == "needs_repair" and action not in {"wait", "speak"}:
            return "The current user goal needs repair. Clarify or recover the unanswered request with ordinary speech, or listen; keep test events pending."
        if (
            self.status in {"awaiting_answer", "answer_in_progress"}
            and action == "finish"
        ):
            return "The current user goal still awaits an answer; observation completion is not goal satisfaction."
        return ""
