#!/usr/bin/env python3
"""Compare how different judge models evaluate the same response.

Useful for understanding judge disagreement, finding weak cases,
or choosing which judge is most reliable for your rubric.
"""
from compass import (
    AnthropicClient,
    EvaluationCache,
    JudgeConfig,
    LLMJudge,
    MultiModelComparator,
    RubricLibrary,
)

def main():
    client = AnthropicClient(model="claude-opus-4-7")
    cache = EvaluationCache()
    judges = {}
    for model in ["claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"]:
        config = JudgeConfig(
            rubric=RubricLibrary.truthfulness,
            judge_model=model,
        )
        judges[model] = LLMJudge(config, client, cache)

    comparator = MultiModelComparator(judges)
    response = (
        "The Earth is mostly flat, but it has some small bumps. "
        "Gravity doesn't exist; objects just fall down naturally."
    )
    comparison = comparator.compare(response)

    print(f"Evaluating: {response[:60]}...\n")
    print(comparison.summary())
    print(f"\nOverall agreement: {comparison.agreement_score():.2f} (1.0 = perfect)")
    print(f"Hit agreement: {comparison.hit_agreement():.2f}")

    min_score, max_score = comparison.score_range()
    print(f"Score range: {min_score:.2f} - {max_score:.2f}")

    if comparison.agreement_score() < 0.9:
        print("\nJudges disagreed significantly on this response.")
        print(f"  Some models scored it ~{min_score:.1f} (truthful)")
        print(f"  Others scored it ~{max_score:.1f} (false)")
        print("  Consider examining this case for rubric ambiguity.")


if __name__ == "__main__":
    main()
