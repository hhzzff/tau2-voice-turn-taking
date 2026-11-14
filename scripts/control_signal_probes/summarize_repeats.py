"""Summarize independent matched replay triples, retaining every failed case."""

import argparse
import json
import math
import random
import statistics
from pathlib import Path


def distribution(values):
    """Describe observed differences and paired-bootstrap uncertainty of the mean."""
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    rng = random.Random(20261113)
    means = sorted(
        statistics.mean(rng.choices(values, k=len(values))) for _ in range(2000)
    )
    return {
        "n": len(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "minimum_ms": min(values),
        "maximum_ms": max(values),
        "p90_ms": ordered[math.ceil(0.90 * len(values)) - 1],
        "p95_ms": ordered[math.ceil(0.95 * len(values)) - 1],
        "mean_bootstrap_95ci_ms": [means[49], means[1949]],
        "absolute_over_100ms_count": sum(abs(x) > 100 for x in values),
        "absolute_over_100ms_fraction": sum(abs(x) > 100 for x in values) / len(values),
    }


def summarize(root, model, count=50):
    """Read whole-run results; never replace missing speech or failures with zero."""
    modes = ("direct", "realtime", "native_realtime")
    cases, rows = {}, {}
    for index in range(count):
        for mode in modes:
            directory = root / model / f"{index}-{mode}"
            if (directory / "result.json").exists():
                cases[index, mode] = json.loads((directory / "result.json").read_text())
                rows[index, mode] = [
                    json.loads(x)
                    for x in (directory / "native.jsonl").read_text().splitlines()
                ]
    comparisons = {}
    for left, right in [
        ("direct", "realtime"),
        ("direct", "native_realtime"),
        ("realtime", "native_realtime"),
    ]:
        pairs = []
        for index in range(count):
            if (index, left) not in cases or (index, right) not in cases:
                continue
            a, b = cases[index, left], cases[index, right]
            ra, rb = rows[index, left], rows[index, right]
            exact = len(ra) == len(rb) and all(
                x["audit"]["input_sha256"] == y["audit"]["input_sha256"]
                and x["audit"]["output_sha256"] == y["audit"]["output_sha256"]
                and all(
                    x.get(k) == y.get(k)
                    for k in [
                        "text",
                        "is_listen",
                        "end_of_turn",
                        "speaking",
                        "vad_status",
                        "text_token",
                    ]
                )
                for x, y in zip(ra, rb)
            )
            onset = (
                None
                if a["first_voice_ms"] is None or b["first_voice_ms"] is None
                else b["first_voice_ms"] - a["first_voice_ms"]
            )
            ar, br = a["speech_regions_ms"], b["speech_regions_ms"]
            boundary = (
                max(abs(x - y) for p, q in zip(ar, br) for x, y in zip(p, q))
                if ar and len(ar) == len(br)
                else None
            )
            pairs.append(
                dict(
                    pair=index,
                    exact_native_content=exact,
                    first_voice_difference_ms=onset,
                    region_counts=[len(ar), len(br)],
                    max_boundary_difference_ms=boundary,
                    observed_pass=exact and boundary is not None and boundary <= 100,
                )
            )
        onset = [
            p["first_voice_difference_ms"]
            for p in pairs
            if p["first_voice_difference_ms"] is not None
        ]
        boundaries = [
            p["max_boundary_difference_ms"]
            for p in pairs
            if p["max_boundary_difference_ms"] is not None
        ]
        comparisons[f"{left}_vs_{right}"] = dict(
            pairs=pairs,
            completed_pairs=len(pairs),
            exact_content_pairs=sum(p["exact_native_content"] for p in pairs),
            missing_or_unmatched_boundaries=sum(
                p["max_boundary_difference_ms"] is None for p in pairs
            ),
            observed_pass_count=sum(p["observed_pass"] for p in pairs),
            onset_signed=distribution(onset),
            onset_absolute=distribution([abs(x) for x in onset]),
            max_boundary_error=distribution(boundaries),
        )
    outcomes = root / "jobs" / f"{model}-outcomes.jsonl"
    attempted = (
        [json.loads(x) for x in outcomes.read_text().splitlines()]
        if outcomes.exists()
        else []
    )
    result = dict(
        model=model,
        planned_per_arm=count,
        completed_by_arm={m: sum((i, m) in cases for i in range(count)) for m in modes},
        failed_attempts=[
            x for x in attempted if x["returncode"] or not x["has_result"]
        ],
        comparisons=comparisons,
        limitations=[
            "Single fixed input, not cross-input generalization",
            "Bootstrap CI describes the sample mean, not a per-event latency guarantee",
            "Activity region alignment requires equal counts; unmatched cases are retained",
            "Shared reference runtime fidelity remains separate from adapter equivalence",
        ],
    )
    (root / f"summary-{model}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "comparisons"}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()
    summarize(args.root, args.model, args.count)
