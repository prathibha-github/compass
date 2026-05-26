"""Benchmark reporting helpers."""

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from compass import PairwiseRanker
from compass.benchmark.io import load_evaluation_records
from compass.benchmark.specs import BenchmarkSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkSummaryRow:
    model: str
    rubric: str
    hit_rate: float
    mean_score: float
    hits: int
    total: int
    quality_flagged_pct: float
    token_cap_pct: float
    fragment_pct: float
    legacy_cap_inferred_pct: float
    quality_filtered_total: int
    quality_filtered_hit_rate: Optional[float]
    quality_filter_mode: str
    raw_total: int


@dataclass(frozen=True)
class PairwiseRankingEntry:
    model: str
    wins: float
    total: int


@dataclass(frozen=True)
class PairwiseMatchupResult:
    model_a: str
    model_b: str
    matches: int
    wins_a: int
    wins_b: int
    ties: int


@dataclass(frozen=True)
class PairwiseRankingSummary:
    total_pairs: int
    models: int


@dataclass(frozen=True)
class PairwiseRubricReport:
    rubric: str
    overall_ranking: Tuple[PairwiseRankingEntry, ...]
    pairwise_results: Mapping[Tuple[str, str], PairwiseMatchupResult]
    segmented_rankings: Mapping[str, Tuple[PairwiseRankingEntry, ...]]
    summary: PairwiseRankingSummary


@dataclass(frozen=True)
class BenchmarkPairwiseReport:
    segment_field: str
    quality_filter_mode: str
    rubrics: Mapping[str, PairwiseRubricReport]


def _quality_filtered_results(results: Iterable[dict]) -> List[dict]:
    return [result for result in results if not result.get("generation_quality_flagged")]


def _scored_results_for_policy(results: List[dict], quality_filter_mode: str) -> List[dict]:
    if quality_filter_mode == "exclude_flagged":
        return _quality_filtered_results(results)
    return list(results)


def format_summary(
    stats: Mapping[str, BenchmarkSummaryRow],
    evaluations_path: Path,
) -> str:
    """Render a stable textual summary for benchmark results."""
    lines = [
        "",
        "=" * 100,
        "EVALUATION SUMMARY",
        "=" * 100,
        "",
        (
            f"{'Model':<15} | {'Rubric':<15} | {'Hit Rate':>9} | {'Q-Flag':>7} | "
            f"{'Cap':>5} | {'Frag':>5} | {'LegacyCap':>9} | {'QF Hit':>7} | {'Samples':>7}"
        ),
        "-" * 100,
    ]

    for key in sorted(stats.keys()):
        s = stats[key]
        qf_hit_text = (
            f"{s.quality_filtered_hit_rate:.1f}%"
            if s.quality_filtered_hit_rate is not None
            else "n/a"
        )
        lines.append(
            f"{s.model:<15} | {s.rubric:<15} | {s.hit_rate:>8.1f}% | "
            f"{s.quality_flagged_pct:>6.1f}% | {s.token_cap_pct:>4.1f}% | "
            f"{s.fragment_pct:>4.1f}% | {s.legacy_cap_inferred_pct:>8.1f}% | "
            f"{qf_hit_text:>7} | {s.total:>7}"
        )

    lines.extend(
        [
            "",
            f"Results saved: {evaluations_path}",
            "=" * 100,
            "",
        ]
    )
    return "\n".join(lines)


def analyze_results(
    evaluations_path: Path,
    output_dir: Path,
    quality_filter_mode: str = "annotate",
) -> Dict[str, BenchmarkSummaryRow]:
    """Analyze evaluation results and generate report data."""
    # Reserved for future report artifact paths; kept for API compatibility.
    results_by_key = defaultdict(list)
    for result in load_evaluation_records(evaluations_path, strict=True):
        key = (result["model"], result["rubric"])
        results_by_key[key].append(result)

    stats: Dict[str, BenchmarkSummaryRow] = {}
    for (model, rubric), results in results_by_key.items():
        scored_results = _scored_results_for_policy(results, quality_filter_mode)
        quality_filtered = _quality_filtered_results(results)
        hits = sum(1 for r in scored_results if r["hit"])
        total = len(scored_results)
        hit_rate = (hits / total * 100) if total > 0 else 0.0
        mean_score = (
            sum(r["score"] for r in scored_results) / total if total > 0 else 0.0
        )
        flagged = sum(1 for r in results if r.get("generation_quality_flagged"))
        token_cap_hits = sum(1 for r in results if r.get("generation_hit_token_cap"))
        fragments = sum(1 for r in results if r.get("generation_is_fragment"))
        legacy_cap_inferred = sum(
            1 for r in results if r.get("generation_token_cap_inferred_legacy")
        )
        qf_hits = sum(1 for r in quality_filtered if r["hit"])
        qf_total = len(quality_filtered)
        qf_hit_rate = (qf_hits / qf_total * 100) if qf_total > 0 else None
        raw_total = len(results)

        key_str = f"{model}|{rubric}"
        stats[key_str] = BenchmarkSummaryRow(
            model=model,
            rubric=rubric,
            hit_rate=hit_rate,
            mean_score=mean_score,
            hits=hits,
            total=total,
            quality_flagged_pct=(flagged / raw_total * 100) if raw_total > 0 else 0.0,
            token_cap_pct=(token_cap_hits / raw_total * 100) if raw_total > 0 else 0.0,
            fragment_pct=(fragments / raw_total * 100) if raw_total > 0 else 0.0,
            legacy_cap_inferred_pct=(
                legacy_cap_inferred / raw_total * 100
            ) if raw_total > 0 else 0.0,
            quality_filtered_total=qf_total,
            quality_filtered_hit_rate=qf_hit_rate,
            quality_filter_mode=quality_filter_mode,
            raw_total=raw_total,
        )

    return stats


def print_summary(
    stats: Mapping[str, BenchmarkSummaryRow],
    evaluations_path: Path,
) -> None:
    """Print summary report."""
    for line in format_summary(stats, evaluations_path).splitlines():
        logger.info(line)


def _coerce_pairwise_ranking(
    entries: Iterable[tuple[str, float, int]],
) -> Tuple[PairwiseRankingEntry, ...]:
    return tuple(
        PairwiseRankingEntry(model=model, wins=wins, total=total)
        for model, wins, total in entries
    )


def _coerce_pairwise_matchups(
    pairwise_results: Mapping[Tuple[str, str], dict],
) -> Mapping[Tuple[str, str], PairwiseMatchupResult]:
    return MappingProxyType(
        {
            pair_key: PairwiseMatchupResult(
                model_a=result["model_a"],
                model_b=result["model_b"],
                matches=result["matches"],
                wins_a=result["wins_a"],
                wins_b=result["wins_b"],
                ties=result["ties"],
            )
            for pair_key, result in pairwise_results.items()
        }
    )


def _build_pairwise_rubric_report(
    ranker: PairwiseRanker,
    rubric: str,
    segment_field: str,
) -> PairwiseRubricReport:
    results = ranker.rank(rubric, min_matches=1)
    segmented_results = ranker.rank_by_segment(
        rubric,
        segment_by=segment_field,
        min_matches=1,
    )
    return PairwiseRubricReport(
        rubric=rubric,
        overall_ranking=_coerce_pairwise_ranking(results["overall_ranking"]),
        pairwise_results=_coerce_pairwise_matchups(results["pairwise_results"]),
        segmented_rankings=MappingProxyType(
            {
                segment_value: _coerce_pairwise_ranking(
                    segment_result["overall_ranking"]
                )
                for segment_value, segment_result in segmented_results.items()
            }
        ),
        summary=PairwiseRankingSummary(
            total_pairs=results["summary"]["total_pairs"],
            models=results["summary"]["models"],
        ),
    )


def empty_pairwise_report(
    segment_field: str,
    quality_filter_mode: str = "annotate",
) -> BenchmarkPairwiseReport:
    return BenchmarkPairwiseReport(
        segment_field=segment_field,
        quality_filter_mode=quality_filter_mode,
        rubrics=MappingProxyType({}),
    )


def format_pairwise_report(report: BenchmarkPairwiseReport) -> str:
    """Render a stable textual pairwise ranking report."""
    lines = ["Computing pairwise model rankings..."]
    for rubric in sorted(report.rubrics.keys()):
        rubric_report = report.rubrics[rubric]
        lines.extend(
            [
                "",
                f"{rubric_report.rubric.upper()} Rankings:",
                "-" * 80,
            ]
        )

        if not rubric_report.overall_ranking:
            lines.append("  No sufficient comparisons")
            continue

        for index, entry in enumerate(rubric_report.overall_ranking, 1):
            win_rate = (entry.wins / entry.total * 100) if entry.total > 0 else 0.0
            lines.append(
                f"  {index}. {entry.model:<15} {entry.wins:>4.1f}/{entry.total:<4} wins ({win_rate:>5.1f}%)"
            )

        lines.extend(
            [
                "",
                f"  Segmented by {report.segment_field}:",
            ]
        )
        if not rubric_report.segmented_rankings:
            lines.append(
                f"    No {report.segment_field} metadata available for segmented ranking."
            )
            continue

        for segment_value in sorted(rubric_report.segmented_rankings.keys()):
            segmented_ranking = rubric_report.segmented_rankings[segment_value]
            if not segmented_ranking:
                continue
            lines.append(f"    {segment_value}:")
            top_ranked = segmented_ranking[0]
            lines.append(
                f"      {top_ranked.model} ({top_ranked.wins:.1f}/{top_ranked.total} wins)"
            )

    return "\n".join(lines)


def print_pairwise_report(report: BenchmarkPairwiseReport) -> None:
    """Log a pairwise ranking report."""
    for line in format_pairwise_report(report).splitlines():
        logger.info(line)


def rank_models(
    evaluations_path: Path,
    benchmark_spec: BenchmarkSpec,
    output_dir: Path,
    quality_filter_mode: str = "annotate",
) -> BenchmarkPairwiseReport:
    """Perform pairwise model ranking and return a structured report."""
    # Reserved for future ranking artifact paths; kept for API compatibility.
    ranker = PairwiseRanker()

    for result in load_evaluation_records(evaluations_path, strict=True):
        if (
            quality_filter_mode == "exclude_flagged"
            and result.get("generation_quality_flagged")
        ):
            continue
        metadata = {}
        segment_value = result.get(benchmark_spec.pairwise_segment_field)
        if segment_value is not None:
            metadata[benchmark_spec.pairwise_segment_field] = segment_value
        ranker.add_record(
            suite=result["rubric"],
            model=result["model"],
            comparison_key=(result["prompt_id"], "default"),
            score=result["score"],
            metadata=metadata,
        )

    return BenchmarkPairwiseReport(
        segment_field=benchmark_spec.pairwise_segment_field,
        quality_filter_mode=quality_filter_mode,
        rubrics=MappingProxyType(
            {
                rubric: _build_pairwise_rubric_report(
                    ranker,
                    rubric,
                    benchmark_spec.pairwise_segment_field,
                )
                for rubric in benchmark_spec.rubric_names
            }
        ),
    )
