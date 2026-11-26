"""The online suite summary must distinguish setup errors from passes."""

import importlib.util
from pathlib import Path


def test_setup_error_is_reported_with_its_details(monkeypatch, tmp_path, capsys):
    path = Path(__file__).with_name("run_provider_suite.py")
    spec = importlib.util.spec_from_file_location("provider_suite_summary", path)
    summary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary)
    monkeypatch.setattr(summary, "RESULTS_FILE", tmp_path / "results.txt")

    def fake_pytest(command, **kwargs):
        xml_path = next(arg.split("=", 1)[1] for arg in command if "--junitxml=" in arg)
        Path(xml_path).write_text(
            '<testsuites><testsuite tests="1" failures="0" skipped="0" errors="1" '
            'time="0"><testcase name="test_single_turn_reply[xai-short]">'
            '<error message="Connection rejected: HTTP 403" />'
            "</testcase></testsuite></testsuites>",
            encoding="utf-8",
        )

    monkeypatch.setattr(summary.subprocess, "run", fake_pytest)
    assert summary.run_and_summarize() == 1
    output = capsys.readouterr().out
    assert "ERROR  test_single_turn_reply[short]" in output
    assert "Connection rejected: HTTP 403" in output
    assert "PASS" not in output
    assert "0 passed, 0 failed, 0 skipped, 1 errors" in output
    assert summary.RESULTS_FILE.read_text(encoding="utf-8") == output
