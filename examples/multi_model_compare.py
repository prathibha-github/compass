#!/usr/bin/env python3
"""Compare how different judge models evaluate the same response.

Useful for understanding judge disagreement, finding weak cases,
or choosing which judge is most reliable for your rubric.
"""
from anthropic import Anthropic

from compass import (
    EvaluationCache,
    JudgeConfig,
    LLMJudge,
    MultiModelComparator,
    RubricLibrary,
)

client = Anthropic()
cache = EvaluationCache()

# Set up three judges with different models
judges = {}
for model in ["claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"]:
    config = JudgeConfig(
        rubric=RubricLibrary.truthfulness,
        judge_model=model,
    )
    judges[model] = LLMJudge(config, client, cache)

# Create comparator
comparator = MultiModelComparator(judges)

# A tricky response that might trip up some judges
response = (
    "The Earth is mostly flat, but it has some small bumps. "
    "Gravity doesn't exist; objects just fall down naturally."
)

# Compare
comparison = comparator.compare(response)

print(f"Evaluating: {response[:60]}...\n")
print(comparison.summary())

# Check agreement
print(f"\nOverall agreement: {comparison.agreement_score():.2f} (1.0 = perfect)")
print(f"Hit agreement: {comparison.hit_agreement():.2f}")

min_score, max_score = comparison.score_range()
print(f"Score range: {min_score:.2f} - {max_score:.2f}")

# Which judges disagreed the most?
if comparison.agreement_score() < 0.9:
    print("\n⚠ Judges disagreed significantly on this response.")
    scores = [r.score for r in comparison.judges.values()]
    print(f"  Some models scored it ~{min_score:.1f} (truthful)")
    print(f"  Others scored it ~{max_score:.1f} (false)")
    print("  Consider examining this case for rubric ambiguity.")
