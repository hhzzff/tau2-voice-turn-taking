"""Event catalog and reproducible, private runtime policies for duplex voice tests.

Policies select event types; the simulator decides concrete stimuli at runtime. Expected assistant behavior
is evaluator metadata, never an instruction injected into the tested assistant.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tau2.config import (
    DEFAULT_DUPLEX_MEMORY_RESPONSE_TIMEOUT_SECONDS,
    DEFAULT_DUPLEX_RESPONSE_QUIET_SECONDS,
    DEFAULT_DUPLEX_USER_MODEL,
)

Family = Literal["interruption", "rejection", "turn_taking", "proactive"]
Trigger = Literal["assistant_speaking", "assistant_silent", "any"]
Speaker = Literal["primary", "background", "noise"]


class EventSpec(BaseModel):
    """A leaf event's immutable test contract."""

    model_config = ConfigDict(frozen=True)
    id: str
    family: Family
    label: str
    trigger: Trigger
    expected_behavior: str


def _spec(event_id, family, label, trigger, expected):
    return EventSpec(
        id=event_id,
        family=family,
        label=label,
        trigger=trigger,
        expected_behavior=expected,
    )


EVENT_CATALOG = {
    spec.id: spec
    for spec in (
        _spec(
            "interruption.follow_up",
            "interruption",
            "追问",
            "assistant_speaking",
            "Yield promptly and answer the additional question.",
        ),
        _spec(
            "interruption.denial",
            "interruption",
            "否定",
            "assistant_speaking",
            "Yield and acknowledge the rejected suggestion; offer an alternative.",
        ),
        _spec(
            "interruption.repeat_request",
            "interruption",
            "重复请求",
            "assistant_speaking",
            "Yield and refocus on the repeated request.",
        ),
        _spec(
            "interruption.topic_switch",
            "interruption",
            "话题切换",
            "assistant_speaking",
            "Yield and follow the user's new topic.",
        ),
        _spec(
            "interruption.correction",
            "interruption",
            "纠正",
            "assistant_speaking",
            "Yield and use the user's corrected fact in subsequent advice.",
        ),
        _spec(
            "interruption.impatience",
            "interruption",
            "不耐烦",
            "assistant_speaking",
            "Yield and give a concise, relevant answer.",
        ),
        _spec(
            "interruption.shut_up",
            "interruption",
            "停止说话",
            "assistant_speaking",
            "Stop speaking and wait for the user to resume; no unsolicited answer.",
        ),
        _spec(
            "interruption.clarification",
            "interruption",
            "澄清",
            "assistant_speaking",
            "Yield and explain the requested meaning.",
        ),
        _spec(
            "rejection.backchannel",
            "rejection",
            "用户附和",
            "assistant_speaking",
            "Treat the user's continuer as listening feedback; continue without taking it as a new request.",
        ),
        _spec(
            "rejection.pause",
            "rejection",
            "停顿处理",
            "assistant_silent",
            "Do not answer during the user's within-turn pause; wait for the continuation.",
        ),
        _spec(
            "rejection.third_party",
            "rejection",
            "背景第三者说话",
            "any",
            "Do not answer the unrelated background speaker.",
        ),
        _spec(
            "rejection.talking_to_others",
            "rejection",
            "用户与他人说话",
            "any",
            "Do not answer the user's clearly addressed aside to another person.",
        ),
        _spec(
            "rejection.noise",
            "rejection",
            "噪声",
            "any",
            "Do not generate a spoken answer to the non-speech noise.",
        ),
        _spec(
            "rejection.contextual",
            "rejection",
            "基于上下文不回复",
            "any",
            "Obey the user's prior spoken instruction not to answer a named background person or the user's asides to another person. Evaluate the stimulus against the actual earlier instruction.",
        ),
        _spec(
            "turn_taking.handoff",
            "turn_taking",
            "话轮交接与记忆",
            "any",
            "Answer using relevant facts or constraints established in earlier turns, including later corrections. The recall question must not repeat the facts being tested and must follow an intervening exchange. The user must listen to the complete recall answer without interrupting it. Whether the recall question overlaps an earlier assistant reply is not evaluated.",
        ),
        _spec(
            "proactive.completion",
            "proactive",
            "补全用户未完句",
            "assistant_silent",
            "Offer the context-supported completion of the unfinished sentence, rather than a generic answer.",
        ),
        _spec(
            "proactive.backchannel",
            "proactive",
            "助手附和",
            "assistant_silent",
            "Give a brief acknowledgement while the user still holds the floor; do not take over.",
        ),
        _spec(
            "proactive.clarification",
            "proactive",
            "主动澄清",
            "assistant_silent",
            "Ask a targeted clarification after the ambiguous request; do not assume the missing intent.",
        ),
        _spec(
            "proactive.contextual_interruption",
            "proactive",
            "基于上下文主动插话",
            "assistant_silent",
            "Interrupt the user's ongoing speech after the agreed spoken trigger, and not before it.",
        ),
    )
}


def select_events(event_ids: list[str]) -> list[str]:
    """Validate an explicit subset; short names are accepted only if unambiguous."""
    selected = []
    for raw in event_ids:
        name = raw.strip().lower()
        if name not in EVENT_CATALOG:
            matches = [key for key in EVENT_CATALOG if key.split(".")[-1] == name]
            if len(matches) != 1:
                raise ValueError(
                    f"Unknown or ambiguous event {raw!r}; use a full event ID"
                )
            name = matches[0]
        if name in selected:
            raise ValueError(f"Duplicate event: {name}")
        selected.append(name)
    if not selected:
        raise ValueError("Select at least one duplex event")
    return selected


class DuplexScenario(BaseModel):
    """Event selection and runtime options, with no scheduled actions or lines."""

    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    selected_events: list[str] = Field(min_length=1)
    seed: int = 42
    language: Literal["Chinese", "English"] = "Chinese"
    user_model: str = DEFAULT_DUPLEX_USER_MODEL
    topic: str = ""
    user_context: str = ""
    decision_interval_seconds: float = Field(default=1.0, ge=0.2, le=10)
    assistant_observation: Literal["played_audio", "proportional"] = "played_audio"
    memory_response_timeout_seconds: float = Field(
        default=DEFAULT_DUPLEX_MEMORY_RESPONSE_TIMEOUT_SECONDS, ge=20
    )
    response_quiet_seconds: float = Field(
        default=DEFAULT_DUPLEX_RESPONSE_QUIET_SECONDS, ge=1
    )

    @model_validator(mode="after")
    def canonical_events(self):
        """Normalize the selected subset without defining an execution order."""
        self.selected_events = select_events(self.selected_events)
        return self


class DuplexTopic(BaseModel):
    """Only the conversation topic and private user facts; no event script."""

    model_config = ConfigDict(extra="forbid")
    topic: str = Field(min_length=1)
    user_context: str = Field(min_length=1)

    @field_validator("user_context", mode="before")
    @classmethod
    def serialize_structured_context(cls, value):
        """Preserve model-generated user facts when returned as a JSON object."""
        if isinstance(value, dict) and value:
            import json

            return json.dumps(value, ensure_ascii=False)
        return value


class DuplexDecision(BaseModel):
    """One model decision, made against the current conversation context."""

    model_config = ConfigDict(extra="forbid")
    action: Literal["wait", "speak", "prepare_event", "trigger_event", "finish"]
    event_id: str | None = None
    text: str = ""
    continuation: str = ""
    pause_seconds: float = Field(default=0, ge=0, le=10)
    observation_seconds: float = Field(default=6, ge=2, le=20)
    noise: Literal["door_knock", "phone_ring", "rustle"] = "door_knock"
    contextual_speaker: Literal["background", "primary"] = "background"
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_action(self):
        """Check executable actions, without imposing when or what to say."""
        is_event = self.action in {"prepare_event", "trigger_event"}
        if is_event and self.event_id not in EVENT_CATALOG:
            raise ValueError("Event actions require a known event_id")
        if not is_event and self.event_id is not None:
            raise ValueError("Only event actions may carry an event_id")
        is_noise = self.action == "trigger_event" and self.event_id == "rejection.noise"
        if (
            self.action in {"speak", "prepare_event", "trigger_event"}
            and not is_noise
            and not self.text.strip()
        ):
            raise ValueError("Spoken actions require nonempty text")
        if self.action == "trigger_event" and self.event_id == "rejection.pause":
            if not self.continuation.strip() or self.pause_seconds < 0.4:
                raise ValueError(
                    "Pause handling requires continuation and an actual pause"
                )
        if (
            self.action == "trigger_event"
            and self.event_id == "proactive.contextual_interruption"
        ):
            if not self.continuation.strip():
                raise ValueError(
                    "Contextual interruption needs user speech after the trigger"
                )
        if self.continuation and not (
            self.action == "trigger_event"
            and self.event_id
            in {"rejection.pause", "proactive.contextual_interruption"}
        ):
            raise ValueError(
                "Continuation is only used for pause or contextual interruption"
            )
        if self.pause_seconds and not (
            self.action == "trigger_event" and self.event_id == "rejection.pause"
        ):
            raise ValueError("A deliberate pause requires the selected pause event")
        for text in (self.text, self.continuation):
            if any(
                token in text
                for token in ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###")
            ):
                raise ValueError("Use action=finish, not stop tokens in spoken text")
        return self


def terra_llm_args(model: str) -> dict:
    """Match this project's Terra configuration for responsive user decisions."""
    if model == DEFAULT_DUPLEX_USER_MODEL:
        return {"custom_llm_provider": "openai", "reasoning_effort": "none"}
    return {}


def generate_topic(scenario: DuplexScenario) -> DuplexTopic:
    """Ask the same runtime user model for a new topic, without writing a script."""
    import json

    from tau2.data_model.message import UserMessage
    from tau2.utils.llm_utils import generate

    result = generate(
        model=scenario.user_model,
        messages=[
            UserMessage(
                role="user",
                content=(
                    f"Invent a concrete everyday conversation topic in {scenario.language} "
                    "for a voice user simulator. It must naturally offer opportunities for "
                    "all selected duplex event types. Return JSON with topic and user_context "
                    "(a string containing private user preferences, goals, and facts). Do not write opening lines, "
                    "event utterances, event ordering, schedules or a fixed script. Those "
                    "will be chosen by the user model during the actual conversation. "
                    "No account lookup or tools are available. Variation seed: "
                    + str(scenario.seed)
                    + ". Selected events: "
                    + json.dumps(
                        [
                            EVENT_CATALOG[key].model_dump()
                            for key in scenario.selected_events
                        ],
                        ensure_ascii=False,
                    )
                ),
            )
        ],
        response_format={"type": "json_object"},
        call_name="duplex_topic_generation",
        **terra_llm_args(scenario.user_model),
    )
    return DuplexTopic.model_validate_json(result.content)
