#!/usr/bin/env python3
"""Validate changed benchmark evaluation reports in a pull request."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable, Sequence

from compass.benchmark.validation import (
    BenchmarkValidationIssue,
    validate_benchmark_report,
)


def is_benchmark_report_path(path: str) -> bool:
    candidate = Path(path)
    name = candidate.name
    if not name.endswith(".jsonl"):
        return False
    if ".bak" in name:
        return False
    return "evaluations" in candidate.stem


def changed_benchmark_report_paths(changed_files: Iterable[str]) -> list[Path]:
    report_paths = []
    for raw_path in changed_files:
        if not raw_path or not is_benchmark_report_path(raw_path):
            continue
        path = Path(raw_path)
        if path.exists():
            report_paths.append(path)
    return report_paths


def validate_changed_reports(
    changed_files: Iterable[str],
    *,
    validator: Callable[[Path], Sequence[BenchmarkValidationIssue | str]] = (
        validate_benchmark_report
    ),
) -> list[str]:
    errors: list[str] = []
    for report_path in changed_benchmark_report_paths(changed_files):
        report_errors = validator(report_path)
        errors.extend(
            f"{report_path}: {error}"
            for error in report_errors
        )
    return errors


def _changed_files_from_git(base_ref: str) -> list[str]:
    proc = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate changed benchmark evaluation reports in a PR."
    )
    parser.add_argument(
        "changed_files",
        nargs="*",
        help="Optional explicit changed file list. If omitted, --base-ref is required.",
    )
    parser.add_argument(
        "--base-ref",
        help="Git base ref to diff against, for example origin/main.",
    )
    args = parser.parse_args(argv)

    if args.changed_files:
        changed_files = list(args.changed_files)
    elif args.base_ref:
        changed_files = _changed_files_from_git(args.base_ref)
    else:
        parser.error("pass changed_files or --base-ref")

    report_paths = changed_benchmark_report_paths(changed_files)
    if not report_paths:
        print("No changed benchmark reports to validate.")
        return 0

    errors = validate_changed_reports(changed_files)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    for report_path in report_paths:
        print(f"OK: benchmark report validation passed for {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
