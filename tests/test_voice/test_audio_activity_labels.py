"""Audio-aligned speech label coverage."""

import numpy as np

from tau2.data_model.audio import (
    PCM_SAMPLE_RATE,
    AudioEncoding,
    AudioFormat,
    audio_bytes_to_string,
)
from tau2.data_model.message import AssistantMessage, Tick, UserMessage
from tau2.voice.synthesis.conversation_builder import (
    _collect_speech_segments,
    generate_audacity_labels,
)


def assistant_chunk(samples, *, text, contains_speech):
    return AssistantMessage(
        role="assistant",
        content=text,
        contains_speech=contains_speech,
        audio_content=audio_bytes_to_string(samples.astype("<i2").tobytes()),
        audio_format=AudioFormat(
            encoding=AudioEncoding.PCM_S16LE, sample_rate=PCM_SAMPLE_RATE
        ),
    )


def test_speech_segments_follow_audio_energy_and_trim_silence():
    audible = np.full(3200, 1000, dtype="<i2")
    silent = np.zeros(3200, dtype="<i2")
    chunks = [
        assistant_chunk(silent, text="early transcript", contains_speech=True),
        assistant_chunk(audible, text="Hello ", contains_speech=False),
        assistant_chunk(audible, text="there", contains_speech=True),
        *[assistant_chunk(silent, text="", contains_speech=False) for _ in range(4)],
        assistant_chunk(audible, text="again", contains_speech=True),
        *[assistant_chunk(silent, text="", contains_speech=False) for _ in range(6)],
        assistant_chunk(audible, text="Later", contains_speech=True),
    ]
    ticks = [
        Tick(tick_id=index, timestamp="2026-01-01T00:00:00", agent_chunk=chunk)
        for index, chunk in enumerate(chunks)
    ]

    segments = _collect_speech_segments(ticks, "assistant")

    assert [(s.start_tick, s.end_tick, s.text) for s in segments] == [
        (1, 7, "Hello thereagain"),
        (14, 14, "Later"),
    ]


def test_quiet_tick_words_survive_within_played_utterance():
    audible = np.full(3200, 1000, dtype="<i2")
    quiet = np.zeros(3200, dtype="<i2")
    chunks = []
    for text, samples in [
        ("I ", quiet),
        ("will ", audible),
        ("directly ", quiet),
        ("answer", audible),
        (" you.", quiet),
    ]:
        chunk = assistant_chunk(samples, text=text, contains_speech=True)
        chunk.utterance_ids = ["played"]
        # The unplayed remainder must never appear in labels.
        chunk.audio_script_gold = "I will directly answer you. Unplayed remainder."
        chunks.append(chunk)
    ticks = [
        Tick(tick_id=i, timestamp="2026-01-01T00:00:00", agent_chunk=c)
        for i, c in enumerate(chunks)
    ]
    segments = _collect_speech_segments(ticks, "assistant")
    assert [(s.start_tick, s.end_tick, s.text) for s in segments] == [
        (1, 3, "I will directly answer you."),
    ]


def test_quiet_words_without_ids_survive_short_internal_pauses():
    ticks = [
        Tick(
            tick_id=i,
            timestamp="2026-01-01T00:00:00",
            agent_chunk=assistant_chunk(
                np.full(3200, volume, dtype="<i2"),
                text=text,
                contains_speech=True,
            ),
        )
        for i, (text, volume) in enumerate(
            [("Use ", 1000), ("two ", 0), ("eggs.", 1000)]
        )
    ]
    assert _collect_speech_segments(ticks, "assistant")[0].text == "Use two eggs."


def test_labels_separate_background_and_adjacent_utterances(tmp_path):
    ticks = []
    for i, (text, source, uid) in enumerate(
        [
            ("Dinner", "primary", "u1"),
            ("File?", "background", "b1"),
            ("Two eggs?", "primary", "u2"),
            ("Thank you.", "primary", "u3"),
            ("_noise_", "noise", "n1"),
        ]
    ):
        ticks.append(
            Tick(
                tick_id=i,
                timestamp="2026-01-01T00:00:00",
                user_chunk=UserMessage(
                    role="user",
                    content=text,
                    source=source,
                    utterance_ids=[uid],
                    contains_speech=True,
                ),
            )
        )
    ticks.append(
        Tick(
            tick_id=5,
            timestamp="2026-01-01T00:00:00",
            user_chunk=UserMessage(
                role="user", content="###STOP###", contains_speech=True
            ),
        )
    )
    paths = generate_audacity_labels(ticks, 200, tmp_path)
    assert paths["user"].read_text(encoding="utf-8").splitlines() == [
        "0.000\t0.200\tDinner",
        "0.400\t0.600\tTwo eggs?",
        "0.600\t0.800\tThank you.",
    ]
    assert paths["third_party"].read_text(encoding="utf-8") == "0.200\t0.400\tFile?"


def test_silent_utterance_does_not_leak_into_next_response():
    ticks = []
    for i, (uid, volume, text) in enumerate(
        [("unplayed", 0, "Never heard."), ("played", 1000, "Hello.")]
    ):
        chunk = assistant_chunk(
            np.full(3200, volume, dtype="<i2"), text=text, contains_speech=True
        )
        chunk.utterance_ids = [uid]
        ticks.append(
            Tick(tick_id=i, timestamp="2026-01-01T00:00:00", agent_chunk=chunk)
        )
    assert [s.text for s in _collect_speech_segments(ticks, "assistant")] == ["Hello."]
