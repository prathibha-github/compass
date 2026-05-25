"""Benchmark report validation helpers."""

from pathlib import Path
from typing import Iterable, List

from compass.benchmark.io import load_evaluation_records
from compass.benchmark.reporting import analyze_results

REQUIRED_EVALUATION_QUALITY_FIELDS = (
    "generation_visible_chars",
    "generation_visible_word_count",
    "generation_hit_token_cap",
    "generation_is_fragment",
    "generation_quality_flagged",
    "generation_finish_reason",
    "generation_token_cap_inferred_legacy",
)

REQUIRED_STATS_QUALITY_FIELDS = (
    "quality_flagged_pct",
    "token_cap_pct",
    "fragment_pct",
    "legacy_cap_inferred_pct",
    "quality_filtered_total",
    "quality_filtered_hit_rate",
)


def validate_evaluation_records_for_quality(records: Iterable[dict]) -> List[str]:
    """Return validation errors for benchmark evaluation rows."""
    errors: List[str] = []
    any_rows = False
    for idx, record in enumerate(records, 1):
        any_rows = True
        missing = [
            field for field in REQUIRED_EVALUATION_QUALITY_FIELDS if field not in record
        ]
        if missing:
            errors.append(
                f"evaluation row {idx} missing quality fields: {', '.join(missing)}"
            )
    if not any_rows:
        errors.append("no evaluation rows found")
    return errors


def validate_stats_for_quality(stats: dict) -> List[str]:
    """Return validation errors for aggregated benchmark stats."""
    errors: List[str] = []
    if not stats:
        return ["no benchmark stats generated"]

    for key, row in stats.items():
        missing = [field for field in REQUIRED_STATS_QUALITY_FIELDS if field not in row]
        if missing:
            errors.append(
                f"stats row {key!r} missing quality fields: {', '.join(missing)}"
            )
    return errors


def validate_benchmark_report(evaluations_path: Path) -> List[str]:
    """Validate that a benchmark report includes required quality diagnostics."""
    rows = load_evaluation_records(evaluations_path)
    errors = validate_evaluation_records_for_quality(rows)
    if errors:
        return errors
    stats = analyze_results(evaluations_path, evaluations_path.parent)
    errors.extend(validate_stats_for_quality(stats))
    return errors
