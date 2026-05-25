#!/usr/bin/env python3
"""Validate benchmark evaluations for required quality diagnostics."""

import argparse
import sys
from pathlib import Path

from compass.benchmark.validation import validate_benchmark_report


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
    errors = validate_benchmark_report(args.evaluations_path)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"OK: benchmark report validation passed for {args.evaluations_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
