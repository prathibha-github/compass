#!/usr/bin/env python3
"""Validate benchmark evaluations for required quality diagnostics."""

import argparse
import logging
from pathlib import Path

from compass.benchmark import log_errors_and_exit, run_or_exit
from compass.benchmark.validation import validate_benchmark_report

logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate benchmark evaluation output for required quality diagnostics."
    )
    parser.add_argument(
        "evaluations_path",
        type=Path,
        help="Path to benchmark evaluations JSONL file",
    )
    return parser


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()
    errors = run_or_exit(
        lambda: validate_benchmark_report(args.evaluations_path),
        logger,
        exit_code=1,
    )
    if errors:
        log_errors_and_exit(
            logger,
            [str(error) for error in errors],
            exit_code=1,
        )
    print(f"OK: benchmark report validation passed for {args.evaluations_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
