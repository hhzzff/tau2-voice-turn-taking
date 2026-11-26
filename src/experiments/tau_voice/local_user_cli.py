"""Dependency-light registration for the optional local-user experiment."""

import argparse
import asyncio
from pathlib import Path

from tau2.config import DEFAULT_LOCAL_USER_TARGET_MODEL


def _run(args: argparse.Namespace) -> int:
    from experiments.tau_voice.local_user_bench import run_suite

    return asyncio.run(run_suite(args))


def add_parser(commands: argparse._SubParsersAction) -> None:
    """Register options without importing optional ASR or audio libraries."""
    parser = commands.add_parser(
        "local", help="Run controlled local-user scenes on continuous real clocks"
    )
    parser.add_argument("--scenes", help="Comma-separated IDs; default all 19")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asr-model-path", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_LOCAL_USER_TARGET_MODEL)
    parser.add_argument(
        "--max-seconds",
        type=float,
        help="Override each selected scene's real deadline; recorded in case.json",
    )
    parser.add_argument(
        "--target-max-output-tokens",
        type=int,
        help="Override target output cap; recorded in case.json and session acknowledgement",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.set_defaults(func=_run)
