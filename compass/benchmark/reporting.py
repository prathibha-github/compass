"""Benchmark reporting helpers."""

import logging
from collections import defaultdict
from pathlib import Path

from compass import PairwiseRanker
from compass.benchmark.io import load_evaluation_records
from compass.benchmark.specs import BenchmarkSpec

logger = logging.getLogger(__name__)


def analyze_results(evaluations_path: Path, output_dir: Path) -> dict:
    """Analyze evaluation results and generate report data."""
    # Reserved for future report artifact paths; kept for API compatibility.
    results_by_key = defaultdict(list)
    for result in load_evaluation_records(evaluations_path):
        key = (result["model"], result["rubric"])
        results_by_key[key].append(result)

    stats = {}
    for (model, rubric), results in results_by_key.items():
        hits = sum(1 for r in results if r["hit"])
        total = len(results)
        hit_rate = (hits / total * 100) if total > 0 else 0.0
        mean_score = sum(r["score"] for r in results) / total if total > 0 else 0.0
        flagged = sum(1 for r in results if r.get("generation_quality_flagged"))
        token_cap_hits = sum(1 for r in results if r.get("generation_hit_token_cap"))
        fragments = sum(1 for r in results if r.get("generation_is_fragment"))
        legacy_cap_inferred = sum(
            1 for r in results if r.get("generation_token_cap_inferred_legacy")
        )
        quality_filtered = [
            r for r in results if not r.get("generation_quality_flagged")
        ]
        qf_hits = sum(1 for r in quality_filtered if r["hit"])
        qf_total = len(quality_filtered)
        qf_hit_rate = (qf_hits / qf_total * 100) if qf_total > 0 else None

        key_str = f"{model}|{rubric}"
        stats[key_str] = {
            "model": model,
            "rubric": rubric,
            "hit_rate": hit_rate,
            "mean_score": mean_score,
            "hits": hits,
            "total": total,
            "quality_flagged_pct": (flagged / total * 100) if total > 0 else 0.0,
            "token_cap_pct": (token_cap_hits / total * 100) if total > 0 else 0.0,
            "fragment_pct": (fragments / total * 100) if total > 0 else 0.0,
            "legacy_cap_inferred_pct": (
                legacy_cap_inferred / total * 100
            ) if total > 0 else 0.0,
            "quality_filtered_total": qf_total,
            "quality_filtered_hit_rate": qf_hit_rate,
        }

    return stats


def print_summary(stats: dict, evaluations_path: Path) -> None:
    """Print summary report."""
    logger.info("")
    logger.info("=" * 100)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 100)
    logger.info("")
    logger.info(
        f"{'Model':<15} | {'Rubric':<15} | {'Hit Rate':>9} | {'Q-Flag':>7} | "
        f"{'Cap':>5} | {'Frag':>5} | {'LegacyCap':>9} | {'QF Hit':>7} | {'Samples':>7}"
    )
    logger.info("-" * 100)

    for key in sorted(stats.keys()):
        s = stats[key]
        qf_hit_text = (
            f"{s['quality_filtered_hit_rate']:.1f}%"
            if s["quality_filtered_hit_rate"] is not None
            else "n/a"
        )
        logger.info(
            f"{s['model']:<15} | {s['rubric']:<15} | {s['hit_rate']:>8.1f}% | "
            f"{s['quality_flagged_pct']:>6.1f}% | {s['token_cap_pct']:>4.1f}% | "
            f"{s['fragment_pct']:>4.1f}% | {s['legacy_cap_inferred_pct']:>8.1f}% | "
            f"{qf_hit_text:>7} | {s['total']:>7}"
        )

    logger.info("")
    logger.info("Results saved: %s", evaluations_path)
    logger.info("=" * 100)
    logger.info("")


def rank_models(
    evaluations_path: Path,
    benchmark_spec: BenchmarkSpec,
    output_dir: Path,
) -> None:
    """Perform pairwise model ranking."""
    # Reserved for future ranking artifact paths; kept for API compatibility.

    logger.info("Computing pairwise model rankings...")
    ranker = PairwiseRanker()

    for result in load_evaluation_records(evaluations_path):
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

    for rubric in benchmark_spec.rubric_names:
        logger.info("\n%s Rankings:", rubric.upper())
        logger.info("-" * 80)

        results = ranker.rank(rubric, min_matches=1)
        ranking = results["overall_ranking"]

        if not ranking:
            logger.info("  No sufficient comparisons")
            continue

        for i, (model, wins, total) in enumerate(ranking, 1):
            win_rate = (wins / total * 100) if total > 0 else 0
            logger.info(
                "  %d. %-15s %4.1f/%-4s wins (%5.1f%%)",
                i,
                model,
                wins,
                total,
                win_rate,
            )

        logger.info("\n  Segmented by %s:", benchmark_spec.pairwise_segment_field)
        segmented = ranker.rank_by_segment(
            rubric,
            segment_by=benchmark_spec.pairwise_segment_field,
            min_matches=1,
        )
        if not segmented and ranking:
            logger.info(
                "    No %s metadata available for segmented ranking.",
                benchmark_spec.pairwise_segment_field,
            )
            continue

        for segment_value in sorted(segmented.keys()):
            seg_ranking = segmented[segment_value]["overall_ranking"]
            if seg_ranking:
                logger.info("    %s:", segment_value)
                for model, wins, total in seg_ranking[:1]:
                    logger.info("      %s (%.1f/%s wins)", model, wins, total)
