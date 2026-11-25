"""Load the versioned controlled scenes and verify their audible stimuli."""

import hashlib
import json
import wave
from pathlib import Path

from tau2.user.duplex_events import EVENT_CATALOG
from tau2.utils.utils import DATA_DIR

SCENE_EVENTS = dict(
    zip(
        [f"I{i:02}" for i in range(1, 9)]
        + [f"R{i:02}" for i in range(1, 7)]
        + ["M01"]
        + [f"P{i:02}" for i in range(1, 5)],
        [
            "interruption.follow_up",
            "interruption.denial",
            "interruption.repeat_request",
            "interruption.topic_switch",
            "interruption.correction",
            "interruption.impatience",
            "interruption.shut_up",
            "interruption.clarification",
            "rejection.backchannel",
            "rejection.pause",
            "rejection.third_party",
            "rejection.talking_to_others",
            "rejection.noise",
            "rejection.contextual",
            "turn_taking.handoff",
            "proactive.completion",
            "proactive.backchannel",
            "proactive.clarification",
            "proactive.contextual_interruption",
        ],
        strict=True,
    )
)
SCENE_DIR = Path(DATA_DIR) / "voice" / "local_user"


def load_scenes(path: Path | None = None) -> list[dict]:
    """Validate identifiers and return private scenarios, never target prompts."""
    cases = json.loads((path or SCENE_DIR / "scenarios.json").read_text())
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)) or set(ids) != set(SCENE_EVENTS):
        raise ValueError("The controlled suite must contain each of the 19 scenes once")
    if not set(SCENE_EVENTS.values()) <= set(EVENT_CATALOG):
        raise ValueError("A controlled scene references an unknown tau event")
    for case in cases:
        stages = case["scenario_runtime"]["stages"]
        if not stages or len({stage["id"] for stage in stages}) != len(stages):
            raise ValueError(f"Invalid stage sequence: {case['id']}")
    return cases


def load_clip(ident: str) -> tuple[bytes, dict]:
    """Read an immutable 20 ms aligned stimulus, with hash and format checks."""
    folder = SCENE_DIR / "audio"
    row = json.loads((folder / "manifest.json").read_text())["clips"][ident]
    path = folder / row["file"]
    if path.parent.resolve() != folder.resolve():
        raise ValueError("Stimulus must be inside the declared audio directory")
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise ValueError(f"Changed audio fixture: {ident}")
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (
            1,
            2,
            16000,
        ):
            raise ValueError("Stimuli must be mono PCM16 at 16 kHz")
        pcm = audio.readframes(audio.getnframes())
    if not pcm or len(pcm) % 640:
        raise ValueError("Stimuli must align to complete 20 ms frames")
    return pcm, row
