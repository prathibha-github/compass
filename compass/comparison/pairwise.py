"""Pairwise model comparison with segmentation support."""
import logging
import math
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PairwiseRanker:
    """Head-to-head model comparison with segmentation.

    Compares models on a shared set of evaluations, calculating win/loss/tie
    rates and rankings. Supports segmentation by task_type, condition, or other
    metadata fields for detailed analysis.

    Scoring:
    - Lower score wins (fewer tics = better)
    - Win: score < opponent's score
    - Tie: score == opponent's score (worth 0.5 point each)
    - Loss: score > opponent's score

    Usage:
        ranker = PairwiseRanker()
        for record in checkpoint_records:
            ranker.add_record(
                model="gpt-4o",
                comparison_key=("prompt_1", "neutral"),
                score=0.5,
                metadata={"task_type": "coding"}
            )
        rankings = ranker.rank("task_focus")
        segmented = ranker.rank_by_segment("task_focus", segment_by="task_type")
    """

    def __init__(self):
        """Initialize ranker."""
        # Structure: {suite: {comparison_key: [(model, score, metadata), ...]}}
        self.records = defaultdict(lambda: defaultdict(list))

    def add_record(
        self,
        suite: str,
        model: str,
        comparison_key: Tuple[Any, ...],
        score: float,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Add an evaluation record for pairwise comparison.

        Args:
            suite: Suite name (e.g., "task_focus")
            model: Model name (e.g., "gpt-4o")
            comparison_key: Unique sample key, usually (prompt_id, condition, sample_idx)
            score: Score for this evaluation (lower is better)
            metadata: Optional metadata (e.g., task_type, segment)
        """
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("Pairwise scores must be finite")
        if any(row["model"] == model for row in self.records[suite][comparison_key]):
            raise ValueError("Duplicate model/comparison key; include sample identity in comparison_key")
        self.records[suite][comparison_key].append({
            "model": model,
            "score": score,
            "metadata": metadata or {},
        })

    def rank(
        self,
        suite: str,
        min_matches: int = 2,
    ) -> Dict[str, Any]:
        """
        Calculate head-to-head rankings for all models in a suite.

        Args:
            suite: Suite name
            min_matches: Minimum shared sample keys across the model pair

        Returns:
            Dict with:
            - overall_ranking: List of (model, score) sorted by wins
            - pairwise_results: Dict of {(model_a, model_b): comparison_dict}
            - summary: Global statistics
        """
        if suite not in self.records:
            return {
                "overall_ranking": [],
                "pairwise_results": {},
                "summary": {"total_pairs": 0, "models": 0},
            }

        if type(min_matches) is not int or min_matches < 1:
            raise ValueError("min_matches must be a positive integer")
        records = self.records[suite]
        by_model = defaultdict(dict)
        for key, rows in records.items():
            for row in rows:
                by_model[row["model"]][key] = row["score"]
        models = sorted(by_model)
        pairwise_results = {}
        model_wins = defaultdict(float)
        model_total = defaultdict(int)
        for a, b in combinations(models, 2):
            shared = by_model[a].keys() & by_model[b].keys()
            if len(shared) < min_matches:
                continue
            wins_a = sum(by_model[a][key] < by_model[b][key] for key in shared)
            ties = sum(by_model[a][key] == by_model[b][key] for key in shared)
            wins_b = len(shared) - wins_a - ties
            pairwise_results[(a, b)] = {
                "model_a": a, "model_b": b, "matches": len(shared),
                "wins_a": wins_a, "wins_b": wins_b, "ties": ties,
                "unmatched_a": len(by_model[a]) - len(shared),
                "unmatched_b": len(by_model[b]) - len(shared),
            }
            model_wins[a] += wins_a + 0.5 * ties
            model_wins[b] += wins_b + 0.5 * ties
            model_total[a] += len(shared)
            model_total[b] += len(shared)
        ranking = [(model, model_wins[model], model_total[model]) for model in models if model_total[model]]
        ranking.sort(key=lambda row: (-row[1], -row[2], row[0]))
        return {
            "overall_ranking": ranking, "pairwise_results": pairwise_results,
            "summary": {"total_pairs": len(pairwise_results), "models": len(models)},
        }

    def rank_by_segment(
        self,
        suite: str,
        segment_by: str = "task_type",
        min_matches: int = 2,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate rankings segmented by metadata field (e.g., by task type).

        Args:
            suite: Suite name
            segment_by: Metadata field to segment on (e.g., "task_type")
            min_matches: Minimum shared records per segment

        Returns:
            Dict of {segment_value: ranking_dict} where ranking_dict
            has same structure as rank()
        """
        if suite not in self.records:
            return {}

        records = self.records[suite]

        for rows in records.values():
            values = {row.get("metadata", {}).get(segment_by) for row in rows}
            values.discard(None)
            if len(values) > 1:
                raise ValueError(f"Conflicting {segment_by} metadata for a shared comparison key")
        # Collect all segments
        segments = set()
        for comparison_records in records.values():
            for record in comparison_records:
                segment_val = record.get("metadata", {}).get(segment_by)
                if segment_val:
                    segments.add(segment_val)

        # Create temporary ranker for each segment
        segmented_results = {}
        for segment in segments:
            segment_ranker = PairwiseRanker()

            for comparison_key, comparison_records in records.items():
                for record in comparison_records:
                    segment_val = record.get("metadata", {}).get(segment_by)
                    if segment_val == segment:
                        segment_ranker.add_record(
                            suite,
                            record["model"],
                            comparison_key,
                            record["score"],
                            record.get("metadata"),
                        )

            segmented_results[segment] = segment_ranker.rank(suite, min_matches)

        return segmented_results
