"""Compare actual native and production-scheduler artifacts, retaining failures."""

import argparse
import json
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def summarize(root, scheduled_mode="scheduled"):
    """Produce per-run and per-pair evidence without smoothing or time alignment."""
    runs, pairs, failures = {}, {}, {}
    for path in sorted(root.glob("*/*/result.json")):
        key = str(path.parent.relative_to(root))
        runs[key] = json.loads(path.read_text())
    for path in sorted(root.glob("*/*/error.json")):
        failures[str(path.parent.relative_to(root))] = json.loads(path.read_text())
    for model in ["salmonn", "moshi", "minicpm", "freeze_omni"]:
        for left, right in [
            ("0-direct", f"1-{scheduled_mode}"),
            ("3-direct", f"2-{scheduled_mode}"),
            ("0-direct", "3-direct"),
        ]:
            ka, kb = f"{model}/{left}", f"{model}/{right}"
            if ka not in runs or kb not in runs:
                continue
            a, b = runs[ka], runs[kb]
            ra, rb = [read_rows(root / k / "native.jsonl") for k in [ka, kb]]
            row = {
                "step_counts": [len(ra), len(rb)],
                "same_user_track": a["input_sha256"] == b["input_sha256"],
            }
            for field in ["input_sha256", "output_sha256"]:
                row[field + "_equal_steps"] = sum(
                    x["audit"][field] == y["audit"][field] for x, y in zip(ra, rb)
                )
            row["text_equal_steps"] = sum(
                x.get("text") == y.get("text") for x, y in zip(ra, rb)
            )
            row["state_equal_steps"] = sum(
                all(
                    x.get(k) == y.get(k)
                    for k in [
                        "is_listen",
                        "end_of_turn",
                        "speaking",
                        "vad_status",
                        "text_token",
                    ]
                )
                for x, y in zip(ra, rb)
            )
            for metric in ["first_voice_ms", "post_interruption_voice_ms"]:
                diff = (
                    None
                    if a[metric] is None or b[metric] is None
                    else b[metric] - a[metric]
                )
                row[metric + "_difference"] = diff
            row["first_voice_within_100ms"] = (
                row["first_voice_ms_difference"] is not None
                and abs(row["first_voice_ms_difference"]) <= 100
            )
            ar, br = a["speech_regions_ms"], b["speech_regions_ms"]
            row["speech_region_counts"] = [len(ar), len(br)]
            row["max_speech_boundary_difference_ms"] = (
                max(abs(x - y) for p, q in zip(ar, br) for x, y in zip(p, q))
                if ar and len(ar) == len(br)
                else None
            )
            boundary_error = row["max_speech_boundary_difference_ms"]
            row["observed_replay_contract_pass"] = (
                len(ra) == len(rb)
                and row["same_user_track"]
                and row["text_equal_steps"] == len(ra)
                and row["input_sha256_equal_steps"] == len(ra)
                and row["output_sha256_equal_steps"] == len(ra)
                and row["state_equal_steps"] == len(ra)
                and boundary_error is not None
                and boundary_error <= 100
            )
            row["full_equivalence_pass"] = (
                False  # Never infer full closure from first onset alone.
            )
            pairs[f"{ka} -> {kb}"] = row
    failure_pairs = {}
    for model in ["salmonn", "moshi", "minicpm", "freeze_omni"]:
        left, right = f"{model}/0-direct", f"{model}/1-{scheduled_mode}"
        if left in failures and right in failures:
            a, b = [read_rows(root / k / "native.jsonl") for k in [left, right]]
            failure_pairs[model] = {
                "completed_steps": [len(a), len(b)],
                "same_error": failures[left]["error"] == failures[right]["error"],
                "input_equal_prefix_steps": sum(
                    x["audit"]["input_sha256"] == y["audit"]["input_sha256"]
                    for x, y in zip(a, b)
                ),
                "output_equal_prefix_steps": sum(
                    x["audit"]["output_sha256"] == y["audit"]["output_sha256"]
                    for x, y in zip(a, b)
                ),
                "full_equivalence_pass": False,
            }
    output = {
        "runs": runs,
        "pairs": pairs,
        "failures": failures,
        "failure_pairs": failure_pairs,
        "notes": [
            "scheduled audio is actual FullDuplexOrchestrator tick output",
            "post-interruption segment onset is not automatically a new semantic answer",
            "SALMONN/Moshi/Freeze share the audited native runtime/controller across arms; native runtime fidelity is not independently validated",
        ],
    }
    (root / "analysis.json").write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {"completed_runs": len(runs), "pairs": pairs, "failures": failures},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--scheduled-mode",
        choices=["scheduled", "realtime", "native_realtime"],
        default="scheduled",
    )
    args = parser.parse_args()
    summarize(args.root, args.scheduled_mode)
