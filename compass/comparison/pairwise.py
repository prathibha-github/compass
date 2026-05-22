"""Pairwise model comparison with segmentation support."""
import logging
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
            comparison_key: Tuple of (prompt_id, condition) for matching
            score: Score for this evaluation (lower is better)
            metadata: Optional metadata (e.g., task_type, segment)
        """
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
            min_matches: Minimum shared records to rank a pair

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

        records = self.records[suite]

        # Collect all models
        all_models = set()
        for comparison_records in records.values():
            for record in comparison_records:
                all_models.add(record["model"])

        all_models = sorted(all_models)

        # Calculate pairwise comparisons
        pairwise_results = {}
        model_wins = defaultdict(int)
        model_total = defaultdict(int)

        for comparison_key, comparison_records in records.items():
            # Group by model
            by_model = defaultdict(list)
            for record in comparison_records:
                by_model[record["model"]].append(record["score"])

            # Compare all model pairs
            for model_a, model_b in combinations(all_models, 2):
                if model_a not in by_model or model_b not in by_model:
                    continue

                scores_a = by_model[model_a]
                scores_b = by_model[model_b]

                # Match on count (zip pairs them)
                matches = min(len(scores_a), len(scores_b))
                if matches < min_matches:
                    continue

                # Count wins and ties
                comparisons = list(zip(scores_a[:matches], scores_b[:matches]))
                wins_a = sum(1 for sa, sb in comparisons if sa < sb)
                ties = sum(1 for sa, sb in comparisons if sa == sb)
                wins_b = matches - wins_a - ties

                pair_key = tuple(sorted([model_a, model_b]))
                if pair_key not in pairwise_results:
                    pairwise_results[pair_key] = {
                        "model_a": model_a,
                        "model_b": model_b,
                        "matches": 0,
                        "wins_a": 0,
                        "wins_b": 0,
                        "ties": 0,
                    }

                pairwise_results[pair_key]["matches"] += matches
                if model_a == pair_key[0]:
                    pairwise_results[pair_key]["wins_a"] += wins_a
                    pairwise_results[pair_key]["wins_b"] += wins_b
                else:
                    pairwise_results[pair_key]["wins_a"] += wins_b
                    pairwise_results[pair_key]["wins_b"] += wins_a

                model_wins[model_a] += wins_a + 0.5 * ties
                model_total[model_a] += matches

                model_wins[model_b] += wins_b + 0.5 * ties
                model_total[model_b] += matches

        # Generate overall ranking
        overall_ranking = [
            (model, model_wins[model], model_total[model])
            for model in all_models
            if model_total[model] > 0
        ]
        overall_ranking.sort(key=lambda x: (-x[1], -x[2]))  # Sort by wins desc, total desc

        return {
            "overall_ranking": overall_ranking,
            "pairwise_results": pairwise_results,
            "summary": {
                "total_pairs": len(pairwise_results),
                "models": len(all_models),
            },
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
