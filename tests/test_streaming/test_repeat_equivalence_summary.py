"""Repeated-run statistics retain failed and incomparable observations."""

import importlib.util
import json
from pathlib import Path


def test_failed_cases_are_not_counted_as_zero_error(tmp_path):
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/control_signal_probes/summarize_repeats.py"
    )
    spec = importlib.util.spec_from_file_location("summarize_repeats", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for mode, onset in [("direct", 1000), ("realtime", 1200)]:
        directory = tmp_path / "moshi" / f"0-{mode}"
        directory.mkdir(parents=True)
        (directory / "result.json").write_text(
            json.dumps(
                {"first_voice_ms": onset, "speech_regions_ms": [[onset, onset + 500]]}
            )
        )
        (directory / "native.jsonl").write_text(
            json.dumps({"audit": {"input_sha256": "i", "output_sha256": "o"}}) + "\n"
        )
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "moshi-outcomes.jsonl").write_text(
        json.dumps(
            {"pair": 1, "mode": "realtime", "returncode": 1, "has_result": False}
        )
        + "\n"
    )
    report = module.summarize(tmp_path, "moshi", count=2)
    comparison = report["comparisons"]["direct_vs_realtime"]
    assert comparison["completed_pairs"] == 1
    assert comparison["onset_signed"]["mean_ms"] == 200
    assert comparison["onset_absolute"]["absolute_over_100ms_fraction"] == 1
    assert comparison["observed_pass_count"] == 0
    assert len(report["failed_attempts"]) == 1
    assert report["completed_by_arm"]["native_realtime"] == 0
