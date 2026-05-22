"""Multi-model comparison for judge evaluations."""
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from compass.judges import EvaluationResult


@dataclass
class ComparisonResult:
    """Result of comparing multiple judges on the same text."""

    rubric_name: str
    text: str
    judges: Dict[str, EvaluationResult] = field(default_factory=dict)

    def agreement_score(self) -> float:
        """Measure agreement across judges (1.0 = perfect agreement, 0.0 = maximum disagreement).

        Uses variance of scores. Lower variance = higher agreement.
        """
        if len(self.judges) < 2:
            return 1.0

        scores = [r.score for r in self.judges.values()]
        variance = float(np.var(scores))

        # Normalize variance: theoretical max is 0.25 (when half judges score 0.0, half score 1.0).
        # This normalization maps [0, 0.25] → [1, 0], so 1.0 = perfect agreement, 0.0 = max disagreement.
        normalized = variance / 0.25
        return max(0.0, 1.0 - normalized)

    def hit_agreement(self) -> float:
        """Fraction of judges that agree on the hit classification.

        Returns 1.0 if all judges agree, 0.0 if all disagree.
        """
        if len(self.judges) < 2:
            return 1.0

        hits = [r.hit for r in self.judges.values()]
        true_count = sum(hits)
        false_count = len(hits) - true_count

        # If all agree on hit or all agree on miss, return 1.0
        if true_count == 0 or false_count == 0:
            return 1.0

        # If split, return how close to consensus we are
        agreement_pct = max(true_count, false_count) / len(hits)
        return agreement_pct

    def score_range(self) -> tuple:
        """Return (min_score, max_score) across all judges."""
        if not self.judges:
            return (0.0, 0.0)
        scores = [r.score for r in self.judges.values()]
        return (min(scores), max(scores))

    def summary(self) -> str:
        """Human-readable comparison summary."""
        lines = [f"Rubric: {self.rubric_name}"]
        lines.append(f"Text: {self.text[:80]}..." if len(self.text) > 80 else f"Text: {self.text}")
        lines.append("")

        for model, result in self.judges.items():
            hit_indicator = "✓" if result.hit else "✗"
            lines.append(f"  {model:<25} {result.score:.2f}  {hit_indicator}  {result.rationale[:50]}")

        lines.append("")
        lines.append(f"  Agreement score: {self.agreement_score():.2f}")
        lines.append(f"  Hit agreement:   {self.hit_agreement():.2f}")
        min_score, max_score = self.score_range()
        lines.append(f"  Score range:     {min_score:.2f} - {max_score:.2f}")

        return "\n".join(lines)


class MultiModelComparator:
    """Compare judge responses across multiple models."""

    def __init__(self, judges: Dict[str, "LLMJudge"]):
        """Initialize with a dict of model_name -> judge.

        Args:
            judges: Dict mapping model names to LLMJudge instances.
                    All judges should use the same rubric for meaningful comparison.
        """
        self.judges = judges

    def compare(self, text: str) -> ComparisonResult:
        """Evaluate text with all judges and return comparison.

        Args:
            text: The text to evaluate with all judges.

        Returns:
            ComparisonResult with results from all judges.
        """
        if not self.judges:
            raise ValueError("No judges configured")

        results = {}
        for model_name, judge in self.judges.items():
            results[model_name] = judge.evaluate(text)

        # Get rubric name from first result
        first_result = next(iter(results.values()))
        rubric_name = first_result.name

        return ComparisonResult(
            rubric_name=rubric_name,
            text=text,
            judges=results,
        )

    def compare_batch(self, texts: List[str]) -> List[ComparisonResult]:
        """Evaluate multiple texts with all judges.

        Args:
            texts: List of texts to evaluate.

        Returns:
            List of ComparisonResult, one per text.
        """
        return [self.compare(text) for text in texts]

    def agreement_stats(self, comparisons: List[ComparisonResult]) -> dict:
        """Compute agreement statistics across multiple comparisons.

        Args:
            comparisons: List of ComparisonResult from compare_batch.

        Returns:
            Dict with agreement metrics.
        """
        if not comparisons:
            return {}

        agreement_scores = [c.agreement_score() for c in comparisons]
        hit_agreements = [c.hit_agreement() for c in comparisons]

        return {
            "mean_agreement": float(np.mean(agreement_scores)),
            "std_agreement": float(np.std(agreement_scores)),
            "min_agreement": float(np.min(agreement_scores)),
            "max_agreement": float(np.max(agreement_scores)),
            "mean_hit_agreement": float(np.mean(hit_agreements)),
            "std_hit_agreement": float(np.std(hit_agreements)),
            "n_comparisons": len(comparisons),
        }
