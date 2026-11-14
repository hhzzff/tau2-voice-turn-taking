"""Run 50 independent, order-balanced paired triples on one GPU per model."""

import argparse
import itertools
import json
import random
import subprocess
import sys
import time
from pathlib import Path

MODES = ("direct", "realtime", "native_realtime")


def schedule(count=50):
    """Balance all six execution orders; keep each matched triple together."""
    orders = list(itertools.permutations(MODES))
    sequence = [orders[index % len(orders)] for index in range(count)]
    random.Random(20261113).shuffle(sequence)
    return [{"pair": index, "order": order} for index, order in enumerate(sequence)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("count must be positive")
    root = args.root.resolve()
    plan = schedule(args.count)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    jobs = root / "jobs"
    jobs.mkdir(exist_ok=True)
    plan_path = jobs / f"{args.model}-plan.json"
    if plan_path.exists():
        raise FileExistsError("This batch already started; preserve its evidence")
    plan_path.write_text(json.dumps(plan, indent=2))
    failed = 0
    for block in plan:
        for mode in block["order"]:
            pair = block["pair"]
            command = [
                "timeout",
                "--signal=TERM",
                "--kill-after=30",
                "900",
                sys.executable,
                str(Path(__file__).with_name("open_equivalence.py")),
                "--root",
                str(root),
                "--model",
                args.model,
                "--mode",
                mode,
                "--run",
                str(pair),
            ]
            started = time.time()
            with (jobs / f"{args.model}-{pair}-{mode}.log").open("x") as out:
                result = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT)
            path = root / args.model / f"{pair}-{mode}" / "result.json"
            valid = result.returncode == 0 and path.exists()
            failed += not valid
            row = dict(
                model=args.model,
                pair=pair,
                mode=mode,
                returncode=result.returncode,
                has_result=path.exists(),
                started_epoch=started,
                ended_epoch=time.time(),
            )
            with (jobs / f"{args.model}-outcomes.jsonl").open("a") as out:
                out.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("summarize_repeats.py")),
            str(root),
            "--model",
            args.model,
        ],
        check=True,
    )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
