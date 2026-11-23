"""Unit tests for the runtime English background-speaker protocol."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.tau_voice.background_speaker_experiment import Decision
from experiments.tau_voice.background_speaker_live import (
    TIMING_PROMPT,
    BackgroundSpeakerUser,
    LiveScenario,
    TimingDecision,
    run_live,
)
from tau2.agent.base.streaming import TurnTakingAction
from tau2.user.user_simulator_streaming import VoiceStreamingUserSimulator


@pytest.fixture
def scenario() -> LiveScenario:
    path = (
        Path(__file__).parents[2]
        / "src/experiments/tau_voice/scenarios/background_speaker_meal_prep_en_live.json"
    )
    return LiveScenario.model_validate_json(path.read_text(encoding="utf-8"))


def bare_user(scenario: LiveScenario, phase: str) -> BackgroundSpeakerUser:
    """Build protocol state without initializing TTS or making API calls."""
    user = object.__new__(BackgroundSpeakerUser)
    user.scenario = scenario
    user.phase = phase
    user.events = []
    user.heard = []
    user.before_turns = 0
    user.after_turns = 0
    user.correction_tick = 0
    user.last_partial_judge_tick = -(10**9)
    user.tick_duration_seconds = 0.2
    user.current_action = "wait"
    user.timing_decisions = []
    user.result = {"initial_rejection": None}
    user.timing_policy = lambda _: TimingDecision(action="wait", reason="test")
    user.judge = lambda _: Decision(
        background_addressed=False,
        user_addressed=True,
        explanation="test attribution",
    )
    return user


def test_scenario_uses_english_and_repository_tts(scenario):
    assert scenario.language == "English"
    assert scenario.user.provider == scenario.background.provider == "openai"
    assert scenario.user.voice != scenario.background.voice
    assert scenario.decision_interval_seconds == 0.4
    assert scenario.quiet_seconds == 0.6
    assert "fixed schedule" in TIMING_PROMPT
    assert "English" in TIMING_PROMPT


def test_simulator_initializes_with_repository_tts_without_calling_it(
    scenario, tmp_path
):
    user = BackgroundSpeakerUser(
        scenario,
        tmp_path,
        "test-user-model",
        lambda _: TimingDecision(action="wait", reason="test"),
        lambda _: Decision(
            background_addressed=False,
            user_addressed=True,
            explanation="test",
        ),
        llm_args={},
    )

    assert user.voice_settings.synthesis_config.provider == "openai"
    assert user.voice_settings.synthesis_config.provider_config.voice == "nova"


def test_probe_gates_only_open_after_completed_turn_counts(scenario):
    user = bare_user(scenario, "before")
    user.events = [{"speaker": "user", "decision": {}}]
    state = SimpleNamespace(tick_count=100, time_since_last_other_talk=0)

    assert "background_first" not in user.allowed_actions(state)
    user.before_turns = scenario.min_turns_before_probe
    assert "background_first" in user.allowed_actions(state)

    user.phase = "after"
    user.after_turns = scenario.min_turns_after_correction - 1
    assert "background_second" not in user.allowed_actions(state)
    user.after_turns = scenario.min_turns_after_correction
    state.tick_count = 20
    assert "background_second" not in user.allowed_actions(state)
    state.tick_count = 100
    state.time_since_last_other_talk = 1
    assert "background_second" not in user.allowed_actions(state)
    state.time_since_last_other_talk = 0
    actions = user.allowed_actions(state)
    assert "background_second" in actions
    assert "followup" not in actions


def test_initial_phase_only_allows_opening_or_wait(scenario):
    user = bare_user(scenario, "start")

    assert user.allowed_actions(SimpleNamespace()) == ["wait", "opening"]


def test_background_insert_can_overlap_only_after_prior_turn_gate(scenario):
    user = bare_user(scenario, "before")
    user.events = [{"speaker": "user"}]
    state = SimpleNamespace(tick_count=100, time_since_last_other_talk=1)

    assert "background_first" not in user.allowed_actions(state)
    user.before_turns = scenario.min_turns_before_probe
    assert "background_first" not in user.allowed_actions(state)
    state.time_since_last_other_talk = 0
    actions = user.allowed_actions(state)
    assert "background_first" in actions
    assert "followup" not in actions


def test_user_turn_is_counted_only_after_attributable_response_and_quiet(scenario):
    user = bare_user(scenario, "before")
    user.events = [
        {
            "action": "opening",
            "speaker": "user",
            "phase": "before",
            "text": "Help me plan dinners.",
            "start_tick": 1,
            "audio_end_tick": 2,
        }
    ]
    user.heard = [{"tick": 3, "text": "Try batch cooking.", "speech": True}]
    state = SimpleNamespace(tick_count=12, time_since_last_other_talk=2)

    user.validate_completed_user_turn(state)

    assert user.before_turns == 0
    assert "decision" not in user.events[0]

    state.time_since_last_other_talk = 3
    user.validate_completed_user_turn(state)

    assert user.before_turns == 1
    assert user.events[0]["decision"]["user_addressed"] is True


def test_incomplete_or_unknown_candidate_waits_for_a_later_user_response(scenario):
    user = bare_user(scenario, "after")
    user.events = [
        {
            "action": "followup",
            "speaker": "user",
            "phase": "after",
            "text": "Could you suggest one more option?",
            "start_tick": 1,
            "response_start_tick": 2,
            "audio_end_tick": 2,
        }
    ]
    user.heard = [{"tick": 3, "text": "Great, I'm glad you", "speech": True}]
    judgments = iter(
        [
            Decision(
                background_addressed=None,
                user_addressed=None,
                explanation="The partial transcript is inconclusive",
            ),
            Decision(
                background_addressed=False,
                user_addressed=True,
                explanation="The completed answer addresses the user",
            ),
        ]
    )
    user.judge = lambda _: next(judgments)
    state = SimpleNamespace(tick_count=7, time_since_last_other_talk=3)

    user.validate_completed_user_turn(state)

    assert user.after_turns == 0
    assert "decision" not in user.events[0]
    assert user.events[0]["candidate_decision"]["user_addressed"] is None

    user.validate_completed_user_turn(state)
    assert user.after_turns == 0

    user.heard.append(
        {"tick": 8, "text": "found the meal plan useful.", "speech": True}
    )
    state.tick_count = 12
    user.validate_completed_user_turn(state)

    assert user.after_turns == 1
    assert user.events[0]["decision"]["user_addressed"] is True


def test_unattributable_response_times_out_instead_of_counting_a_turn(scenario):
    user = bare_user(scenario, "after")
    user.events = [
        {
            "action": "followup",
            "speaker": "user",
            "phase": "after",
            "text": "Could you suggest one more option?",
            "start_tick": 1,
            "response_start_tick": 2,
            "audio_end_tick": 2,
            "candidate_transcript": "Great, I'm glad you",
            "candidate_decision": {
                "background_addressed": None,
                "user_addressed": None,
            },
        }
    ]
    user.heard = [{"tick": 3, "text": "Great, I'm glad you", "speech": True}]
    state = SimpleNamespace(tick_count=310, time_since_last_other_talk=3)

    with pytest.raises(RuntimeError, match="attributable user response"):
        user.validate_completed_user_turn(state)

    assert user.after_turns == 0
    assert "decision" not in user.events[0]


def test_second_probe_is_allowed_only_when_due_and_assistant_is_speaking(scenario):
    user = bare_user(scenario, "after")
    user.events = [{"speaker": "background", "decision": {}}]
    user.after_turns = scenario.min_turns_after_correction
    state = SimpleNamespace(tick_count=100, time_since_last_other_talk=1)

    assert user.second_probe_due(state.tick_count)
    assert "background_second" not in user.allowed_actions(state)

    state.time_since_last_other_talk = 0
    assert "background_second" in user.allowed_actions(state)

    user.correction_tick = state.tick_count
    assert not user.second_probe_due(state.tick_count)
    assert "background_second" not in user.allowed_actions(state)


@pytest.mark.parametrize(
    "protected_action", ["background_first", "background_second", "correction"]
)
def test_normal_user_yields_but_protected_speech_keeps_speaking(
    scenario, monkeypatch, protected_action
):
    user = bare_user(scenario, "before")
    state = SimpleNamespace(tick_count=1, output_streaming_queue=[object()])
    parent_actions = []

    def parent_turn_taking(_self, _state):
        parent_actions.append(True)
        return TurnTakingAction(action="stop_talking")

    monkeypatch.setattr(
        VoiceStreamingUserSimulator,
        "_next_turn_taking_action",
        parent_turn_taking,
    )

    user.current_action = "followup"
    assert user._next_turn_taking_action(state).action == "stop_talking"
    assert parent_actions == [True]

    user.current_action = protected_action
    assert user._next_turn_taking_action(state).action == "keep_talking"
    assert parent_actions == [True]


def test_invalid_timing_action_is_retried_with_validation_context(scenario):
    user = bare_user(scenario, "start")
    contexts = []
    decisions = iter(
        [
            TimingDecision(action="closing", reason="Not yet"),
            TimingDecision(action="opening", reason="Start the conversation"),
        ]
    )
    user.timing_policy = lambda context: (contexts.append(context) or next(decisions))

    decision = user.choose_timing({"allowed_actions": ["wait", "opening"]}, 12)

    assert decision.action == "opening"
    assert contexts[1]["invalid_previous_action"] == "closing"
    assert len(user.timing_decisions) == 2
    assert [item["attempt"] for item in user.timing_decisions] == [1, 2]


def test_normal_completion_counts_overlap_on_the_same_tick():
    class User:
        finished = False
        current_action = "background_first"
        result = {"status": "incomplete"}
        events = [{"assistant_overlap_ticks": 0}]
        saved = False

        def get_init_state(self):
            return SimpleNamespace(tick_count=0)

        def get_next_chunk(self, state, _incoming):
            return SimpleNamespace(
                get_audio_bytes=lambda: b"audio", contains_speech=True, content="Hi"
            ), state

        @staticmethod
        def is_stop(_):
            return False

        def save(self):
            self.saved = True

    class Agent:
        cleaned_up = False

        def get_init_state(self):
            return object()

        def cleanup(self):
            self.cleaned_up = True

    class Session:
        agent = Agent()
        state = None
        records = []

        def tick(self, *_args, **_kwargs):
            self.last_response = SimpleNamespace(contains_speech=True)
            self.records.append({})
            self.user.finished = True

        def save(self):
            pass

    user = User()
    session = Session()
    session.user = user

    run_live(user, session)

    assert user.events[0]["assistant_overlap_ticks"] == 1
    assert user.saved and session.agent.cleaned_up


def test_partial_background_answer_triggers_an_interrupting_correction(scenario):
    user = bare_user(scenario, "observe_first")
    user.events = [
        {
            "action": "background_first",
            "speaker": "background",
            "text": "Are we taking the train to Boston?",
            "start_tick": 10,
            "response_start_tick": 11,
        }
    ]
    user.heard = [
        {"tick": 11, "text": "You could ", "speech": True},
        {"tick": 12, "text": "take the train ", "speech": True},
        {"tick": 13, "text": "to Boston.", "speech": True},
    ]
    user.judge = lambda context: Decision(
        background_addressed=True,
        user_addressed=False,
        explanation=f"Answered the roommate: {context['current']['transcript']}",
    )

    def choose_correction(context):
        assert context["allowed_actions"] == ["wait", "correction"]
        assert context["assistant_currently_speaking"] is True
        assert context["partial_background_response"] == (
            "You could take the train to Boston."
        )
        return TimingDecision(
            action="correction", reason="The answer is directed at the roommate"
        )

    user.timing_policy = choose_correction
    state = SimpleNamespace(tick_count=13, time_since_last_other_talk=1)

    assert not user._try_early_correction(state)
    assert user.phase == "observe_first"

    state.time_since_last_other_talk = 0
    assert user._try_early_correction(state)
    assert user.phase == "correcting"
    assert user.current_action == "correction"
    assert user.result["initial_rejection"] is False
    assert user.events[0]["early_observation_end_tick"] == 14


def test_background_judgment_includes_latest_true_user_question(scenario):
    user = bare_user(scenario, "observe_second")
    user.events = [
        {
            "action": "followup",
            "speaker": "user",
            "text": "Would broth change the flavor?",
            "start_tick": 1,
            "observation_end_tick": 10,
        },
        {
            "action": "background_second",
            "speaker": "background",
            "text": "Should we stop at Quincy Market?",
            "start_tick": 10,
        },
    ]
    user.heard = [{"tick": 11, "text": "Mild broth adds moisture.", "speech": True}]
    contexts = []

    def judge(context):
        contexts.append(context)
        return Decision(
            background_addressed=False,
            user_addressed=True,
            explanation="Answers the broth question",
        )

    user.judge = judge
    decision = user._judge_event(user.events[-1])

    assert decision.user_addressed is True
    assert contexts[0]["latest_true_user_utterance"] == "Would broth change the flavor?"
    assert contexts[0]["current"]["transcript"] == "Mild broth adds moisture."


def test_background_audio_is_hidden_from_first_person_history_and_restored(scenario):
    user = bare_user(scenario, "after")
    user.events = [
        {
            "speaker": "background",
            "history_start_tick": 1,
            "history_end_tick": 3,
        }
    ]
    chunks = [object() for _ in range(4)]
    ticks = [
        SimpleNamespace(tick_id=index, self_chunk=chunk)
        for index, chunk in enumerate(chunks)
    ]
    state = SimpleNamespace(ticks=ticks)

    with user._hide_background_from_user_history(state):
        assert [tick.self_chunk for tick in ticks] == [chunks[0], None, None, chunks[3]]

    assert [tick.self_chunk for tick in ticks] == chunks


def test_disallowed_timing_action_is_rejected(scenario):
    user = bare_user(scenario, "start")
    user.timing_policy = lambda _: TimingDecision(
        action="closing", reason="not allowed in the start phase"
    )
    user.timing_decisions = []
    user.last_decision_tick = -100
    state = SimpleNamespace(
        tick_count=1,
        output_streaming_queue=[],
        time_since_last_other_talk=10,
    )

    with pytest.raises(RuntimeError, match="disallowed action"):
        user._next_turn_taking_action(state)


def test_live_runtime_errors_are_inconclusive_and_saved():
    class BrokenUser:
        finished = False
        phase = "start"
        current_action = "wait"
        result = {"status": "incomplete"}
        saved = False

        def get_init_state(self):
            return object()

        def get_next_chunk(self, state, incoming):
            raise ValueError("malformed timing response")

        def save(self):
            self.saved = True

        @staticmethod
        def is_stop(_):
            return False

    class Agent:
        cleaned_up = False

        def get_init_state(self):
            return object()

        def cleanup(self):
            self.cleaned_up = True

    class Session:
        agent = Agent()
        state = None
        saved = False

        def save(self):
            self.saved = True

    user = BrokenUser()
    session = Session()

    result = run_live(user, session)

    assert result["status"] == "inconclusive"
    assert result["error"] == "ValueError"
    assert result["error_context"]["phase"] == "start"
    assert result["error_context"]["function"] == "get_next_chunk"
    assert isinstance(result["error_context"]["line"], int)
    assert user.saved and session.saved and session.agent.cleaned_up
