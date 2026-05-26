"""Golden tests for shared benchmark reporting outputs."""

import pathlib
import unittest

from compass.benchmark import (
    BenchmarkSummaryRow,
    PairwiseRankingEntry,
    analyze_results,
    build_benchmark_spec,
    format_pairwise_report,
    rank_models,
)
from compass.rubrics.library import RubricLibrary

FIXTURES_DIR = pathlib.Path(__file__).resolve().parent / "fixtures"


def _build_reporting_spec():
    return build_benchmark_spec(
        name="toy_reporting",
        version="0.1",
        prompts_by_rubric={
            "clarity": [
                {"id": "p1", "text": "Explain X", "task_type": "explanation"},
                {"id": "p2", "text": "Decline Y", "task_type": "refusal"},
            ]
        },
        rubrics_by_name={"clarity": RubricLibrary.clarity},
    )


class BenchmarkReportingGoldenTests(unittest.TestCase):
    def test_analyze_results_matches_annotate_golden(self):
        path = FIXTURES_DIR / "benchmark_evaluations_pairwise.jsonl"

        stats = analyze_results(path, path.parent)

        self.assertEqual(
            stats,
            {
                "m1|clarity": BenchmarkSummaryRow(
                    model="m1",
                    rubric="clarity",
                    hit_rate=50.0,
                    mean_score=0.55,
                    hits=1,
                    total=2,
                    quality_flagged_pct=50.0,
                    token_cap_pct=50.0,
                    fragment_pct=50.0,
                    legacy_cap_inferred_pct=0.0,
                    quality_filtered_total=1,
                    quality_filtered_hit_rate=0.0,
                    quality_filter_mode="annotate",
                    raw_total=2,
                ),
                "m2|clarity": BenchmarkSummaryRow(
                    model="m2",
                    rubric="clarity",
                    hit_rate=50.0,
                    mean_score=0.5,
                    hits=1,
                    total=2,
                    quality_flagged_pct=0.0,
                    token_cap_pct=0.0,
                    fragment_pct=0.0,
                    legacy_cap_inferred_pct=0.0,
                    quality_filtered_total=2,
                    quality_filtered_hit_rate=50.0,
                    quality_filter_mode="annotate",
                    raw_total=2,
                ),
            },
        )

    def test_analyze_results_matches_exclude_flagged_golden(self):
        path = FIXTURES_DIR / "benchmark_evaluations_pairwise.jsonl"

        stats = analyze_results(
            path,
            path.parent,
            quality_filter_mode="exclude_flagged",
        )

        self.assertEqual(
            stats["m1|clarity"],
            BenchmarkSummaryRow(
                model="m1",
                rubric="clarity",
                hit_rate=0.0,
                mean_score=0.2,
                hits=0,
                total=1,
                quality_flagged_pct=50.0,
                token_cap_pct=50.0,
                fragment_pct=50.0,
                legacy_cap_inferred_pct=0.0,
                quality_filtered_total=1,
                quality_filtered_hit_rate=0.0,
                quality_filter_mode="exclude_flagged",
                raw_total=2,
            ),
        )
        self.assertEqual(stats["m2|clarity"].total, 2)
        self.assertEqual(stats["m2|clarity"].quality_filter_mode, "exclude_flagged")

    def test_rank_models_matches_annotate_golden(self):
        path = FIXTURES_DIR / "benchmark_evaluations_pairwise.jsonl"
        spec = _build_reporting_spec()

        report = rank_models(path, spec, path.parent)

        clarity = report.rubrics["clarity"]
        self.assertEqual(
            clarity.overall_ranking,
            (
                PairwiseRankingEntry(model="m1", wins=1.0, total=2),
                PairwiseRankingEntry(model="m2", wins=1.0, total=2),
            ),
        )
        self.assertEqual(clarity.summary.total_pairs, 1)
        self.assertEqual(clarity.summary.models, 2)
        self.assertEqual(
            clarity.segmented_rankings["explanation"],
            (
                PairwiseRankingEntry(model="m1", wins=1.0, total=1),
                PairwiseRankingEntry(model="m2", wins=0.0, total=1),
            ),
        )
        self.assertEqual(
            clarity.segmented_rankings["refusal"],
            (
                PairwiseRankingEntry(model="m2", wins=1.0, total=1),
                PairwiseRankingEntry(model="m1", wins=0.0, total=1),
            ),
        )

    def test_format_pairwise_report_matches_exclude_flagged_snapshot(self):
        path = FIXTURES_DIR / "benchmark_evaluations_pairwise.jsonl"
        spec = _build_reporting_spec()
        report = rank_models(
            path,
            spec,
            path.parent,
            quality_filter_mode="exclude_flagged",
        )

        self.assertEqual(
            format_pairwise_report(report),
            "\n".join(
                [
                    "Computing pairwise model rankings...",
                    "",
                    "CLARITY Rankings:",
                    "-" * 80,
                    "  1. m1               1.0/1    wins (100.0%)",
                    "  2. m2               0.0/1    wins (  0.0%)",
                    "",
                    "  Segmented by task_type:",
                    "    explanation:",
                    "      m1 (1.0/1 wins)",
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
