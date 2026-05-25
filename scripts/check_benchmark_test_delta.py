#!/usr/bin/env python3
"""Require benchmark test changes when benchmark logic changes in a PR."""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Iterable, Sequence

BENCHMARK_LOGIC_PREFIXES = (
    "compass/benchmark/",
)
BENCHMARK_LOGIC_FILES = frozenset(
    {
        "examples/constitutional_compliance_benchmark.py",
        "scripts/validate_benchmark_report.py",
    }
)
BENCHMARK_TEST_PREFIXES = (
    "tests/test_benchmark_",
)
BENCHMARK_TEST_FILES = frozenset(
    {
        "tests/test_constitutional_benchmark_core.py",
        "tests/test_examples.py",
    }
)


def _matches_path(
    path: str,
    *,
    exact_files: frozenset[str],
    prefixes: Sequence[str],
) -> bool:
    return path in exact_files or any(path.startswith(prefix) for prefix in prefixes)


def benchmark_logic_changes(changed_files: Iterable[str]) -> list[str]:
    return [
        path
        for path in changed_files
        if _matches_path(
            path,
            exact_files=BENCHMARK_LOGIC_FILES,
            prefixes=BENCHMARK_LOGIC_PREFIXES,
        )
    ]


def benchmark_test_changes(changed_files: Iterable[str]) -> list[str]:
    return [
        path
        for path in changed_files
        if _matches_path(
            path,
            exact_files=BENCHMARK_TEST_FILES,
            prefixes=BENCHMARK_TEST_PREFIXES,
        )
    ]


def validate_benchmark_test_delta(changed_files: Iterable[str]) -> list[str]:
    changed = [path for path in changed_files if path]
    logic_changes = benchmark_logic_changes(changed)
    if not logic_changes:
        return []
    test_changes = benchmark_test_changes(changed)
    if test_changes:
        return []
    return [
        "Benchmark logic changed without a benchmark test update.",
        "Benchmark logic files:",
        *[f"  - {path}" for path in logic_changes],
        "Add or update at least one benchmark test file in tests/.",
    ]


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
        description="Require benchmark tests when benchmark logic changes."
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

    errors = validate_benchmark_test_delta(changed_files)
    if not errors:
        print("Benchmark test delta gate passed.")
        return 0

    for line in errors:
        print(line, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
