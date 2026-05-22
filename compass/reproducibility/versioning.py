"""Reproducibility and versioning for evaluation results."""
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional

from compass.judges import EvaluationResult


@dataclass(frozen=True)
class EvaluationMetadata:
    """Metadata for reproducible evaluations.

    Captures the complete context of an evaluation run, enabling
    reproduction of exact results across time and environments.
    """

    compass_version: str
    rubric_hash: str
    judge_model: str
    seed: int
    timestamp: str
    python_version: str
    prompt_version: str = "1.0"

    @classmethod
    def from_result(cls, result: EvaluationResult, compass_version: str) -> "EvaluationMetadata":
        """Create metadata from an evaluation result."""
        return cls(
            compass_version=compass_version,
            rubric_hash=result.rubric_hash,
            judge_model=result.judge_model,
            seed=42,  # Would come from JudgeConfig in practice
            timestamp=result.timestamp,
            python_version=sys.version.split()[0],
            prompt_version=result.prompt_version,
        )

    def to_dict(self) -> dict:
        """Serialize metadata."""
        return asdict(self)


def cost_summary(results: List[EvaluationResult]) -> Dict[str, float]:
    """Aggregate costs across evaluation results.

    Args:
        results: List of evaluation results.

    Returns:
        Dict with total_cost_usd, token_counts (input/output).
    """
    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    for result in results:
        total_cost += result.cost_usd
        if result.tokens_used:
            total_input_tokens += result.tokens_used.get("input", 0)
            total_output_tokens += result.tokens_used.get("output", 0)

    return {
        "total_cost_usd": round(total_cost, 4),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "results_count": len(results),
    }


def reproducibility_report(
    results: List[EvaluationResult], metadata: Optional[EvaluationMetadata] = None
) -> str:
    """Generate a human-readable reproducibility report.

    Args:
        results: List of evaluation results.
        metadata: Optional evaluation metadata.

    Returns:
        Formatted report showing how to reproduce results.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("REPRODUCIBILITY REPORT")
    lines.append("=" * 80)
    lines.append("")

    if metadata:
        lines.append("Evaluation Context:")
        lines.append(f"  Compass version:  {metadata.compass_version}")
        lines.append(f"  Python version:   {metadata.python_version}")
        lines.append(f"  Timestamp:        {metadata.timestamp}")
        lines.append("")

    if results:
        sample = results[0]
        lines.append("Evaluation Configuration:")
        lines.append(f"  Rubric hash:      {sample.rubric_hash}")
        lines.append(f"  Judge model:      {sample.judge_model}")
        lines.append(f"  Prompt version:   {sample.prompt_version}")
        lines.append("")

    costs = cost_summary(results)
    lines.append("Resource Usage:")
    lines.append(f"  Evaluations:      {costs['results_count']}")
    lines.append(f"  Input tokens:     {costs['total_input_tokens']:,}")
    lines.append(f"  Output tokens:    {costs['total_output_tokens']:,}")
    lines.append(f"  Total tokens:     {costs['total_tokens']:,}")
    lines.append(f"  Total cost:       ${costs['total_cost_usd']:.4f}")
    lines.append("")

    if results:
        cache_hits = sum(1 for r in results if r.cache_hit)
        lines.append("Cache Performance:")
        lines.append(f"  Cache hits:       {cache_hits}/{len(results)} ({100*cache_hits//len(results)}%)")
        lines.append("")

    lines.append("To Reproduce:")
    if results:
        lines.append(f"  1. Use compass version {metadata.compass_version if metadata else '?'}")
        lines.append(f"  2. Load rubric with hash: {sample.rubric_hash}")
        lines.append(f"  3. Initialize judge: {sample.judge_model}")
        lines.append(f"  4. Set seed to 42 (deterministic)")
        lines.append(f"  5. Run evaluation on same texts")

    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)


def cost_per_judge(results: List[EvaluationResult]) -> Dict[str, Dict[str, float]]:
    """Break down costs by judge model.

    Args:
        results: List of evaluation results.

    Returns:
        Dict mapping judge_model -> {total_cost_usd, count, avg_cost_per_eval}.
    """
    by_judge: Dict[str, List[EvaluationResult]] = {}
    for result in results:
        if result.judge_model not in by_judge:
            by_judge[result.judge_model] = []
        by_judge[result.judge_model].append(result)

    summary = {}
    for judge_model, judge_results in by_judge.items():
        total_cost = sum(r.cost_usd for r in judge_results)
        count = len(judge_results)
        summary[judge_model] = {
            "total_cost_usd": round(total_cost, 4),
            "count": count,
            "avg_cost_per_eval": round(total_cost / count, 4) if count > 0 else 0.0,
        }

    return summary
