"""Benchmark report validation helpers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping

from compass.benchmark.io import load_evaluation_records
from compass.benchmark.reporting import BenchmarkSummaryRow, analyze_results

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


@dataclass(frozen=True)
class BenchmarkValidationIssue:
    code: str
    message: str
    location: str

    def __str__(self) -> str:
        if not self.location:
            return self.message
        return f"{self.location}: {self.message}"


def validate_evaluation_records_for_quality(
    records: Iterable[dict],
) -> List[BenchmarkValidationIssue]:
    """Return validation errors for benchmark evaluation rows."""
    errors: List[BenchmarkValidationIssue] = []
    any_rows = False
    for idx, record in enumerate(records, 1):
        any_rows = True
        missing = [
            field for field in REQUIRED_EVALUATION_QUALITY_FIELDS if field not in record
        ]
        if missing:
            errors.append(
                BenchmarkValidationIssue(
                    code="missing_evaluation_quality_fields",
                    location=f"evaluation row {idx}",
                    message=f"missing quality fields: {', '.join(missing)}",
                )
            )
    if not any_rows:
        errors.append(
            BenchmarkValidationIssue(
                code="no_evaluation_rows",
                location="evaluation report",
                message="no evaluation rows found",
            )
        )
    return errors


def validate_stats_for_quality(
    stats: Mapping[str, BenchmarkSummaryRow],
) -> List[BenchmarkValidationIssue]:
    """Return validation errors for aggregated benchmark stats."""
    errors: List[BenchmarkValidationIssue] = []
    if not stats:
        return [
            BenchmarkValidationIssue(
                code="no_benchmark_stats",
                location="summary stats",
                message="no benchmark stats generated",
            )
        ]

    for key, row in stats.items():
        missing = [
            field
            for field in REQUIRED_STATS_QUALITY_FIELDS
            if not hasattr(row, field)
        ]
        if missing:
            errors.append(
                BenchmarkValidationIssue(
                    code="missing_summary_quality_fields",
                    location=f"stats row {key!r}",
                    message=f"missing quality fields: {', '.join(missing)}",
                )
            )
    return errors


def validate_benchmark_report(
    evaluations_path: Path,
) -> List[BenchmarkValidationIssue]:
    """Validate that a benchmark report includes required quality diagnostics."""
    rows = load_evaluation_records(evaluations_path, strict=True)
    errors = validate_evaluation_records_for_quality(rows)
    if errors:
        return errors
    stats = analyze_results(evaluations_path, evaluations_path.parent)
    errors.extend(validate_stats_for_quality(stats))
    return errors
