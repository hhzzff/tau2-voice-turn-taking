"""Goal-neutral responses for existing audio/event execution fixtures.

These fixtures test mechanics, not semantic goal assessment. Goal-specific tests
construct explicit assessments and exercise the real runtime validation.
"""

import json


def runtime_json(content: str) -> str:
    """Add an explicit satisfied goal to a scripted decision response."""
    try:
        value = json.loads(content)
    except (TypeError, ValueError):
        return content
    if isinstance(value, dict) and "action" in value:
        value.setdefault(
            "current_goal",
            {
                "objective": "Exercise the configured audio and event behavior",
                "status": "satisfied",
                "pending_request": "",
                "unresolved_issues": [],
                "evidence": [],
                "stance": "neutral",
            },
        )
    return json.dumps(value, ensure_ascii=False)
