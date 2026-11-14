"""Report provenance and coverage denominators must survive partial evidence."""

import json

import pytest

from experiments.tau_voice.duplex_bench import build_parser
from experiments.tau_voice.evaluation_report import audit_runs
from tau2.user.duplex_events import DuplexScenario


def save_run(directory, topic):
    directory.mkdir()
    scenario = DuplexScenario(selected_events=["follow_up"], topic=topic)
    (directory / "scenario.json").write_text(
        scenario.model_dump_json(), encoding="utf-8"
    )
    return directory


def test_report_preserves_missing_assessments_and_exact_topic_denominator(tmp_path):
    first = save_run(tmp_path / "first", "  Meal  planning ")
    second = save_run(tmp_path / "second", "meal planning")
    third = save_run(tmp_path / "prepared_only", "")
    summary = {"selected": 1, "observed": 0, "not_evaluated": 1}
    (first / "assessment.json").write_text(json.dumps({"summary": summary}))
    audit = audit_runs([first, second, third])
    assert audit["included_run_count"] == 3
    assert audit["runs_with_nonempty_topic"] == 2
    assert audit["distinct_normalized_topic_strings"] == 1
    assert audit["runs"][0]["assessment_summary"] == summary
    assert audit["runs"][1]["assessment_summary"] is None
    assert audit["runs"][2]["assessment_summary"] is None
    assert "user_context" not in audit["runs"][0]


def test_report_refuses_duplicate_run_inclusion(tmp_path):
    directory = save_run(tmp_path / "run", "Dinner")
    with pytest.raises(ValueError, match="Duplicate run directory"):
        audit_runs([directory, directory / "."])


def test_main_report_is_offline_and_escapes_saved_topic(tmp_path, monkeypatch):
    def forbidden_call(*args, **kwargs):
        raise AssertionError("Report generation must reuse saved evidence")

    monkeypatch.setattr("tau2.utils.llm_utils.generate", forbidden_call)
    directory = save_run(tmp_path / "run", '<script>alert("topic")</script>')
    original = (directory / "scenario.json").read_bytes()
    output = tmp_path / "report.html"
    args = build_parser().parse_args(
        ["report", "--output", str(output), "--run-dir", str(directory)]
    )
    assert args.func(args) == 0
    page = output.read_text(encoding="utf-8")
    assert '<script>alert("topic")</script>' not in page
    assert "&lt;script&gt;alert(&quot;topic&quot;)&lt;/script&gt;" in page
    assert "{{SOURCE_" not in page
    evidence = json.loads(output.with_suffix(".evidence.json").read_text())
    assert len(evidence["event_catalog"]) == 19
    assert evidence["run_audit"]["included_run_count"] == 1
    assert (directory / "scenario.json").read_bytes() == original
