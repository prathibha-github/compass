#!/usr/bin/env python3
"""Single text evaluation with compass.

Shows how to score a single completion using an LLM judge and a rubric.
Useful for spot-checking outputs or small-scale evaluations.
"""
from compass import AnthropicClient, EvaluationCache, JudgeConfig, LLMJudge, RubricLibrary

def main():
    client = AnthropicClient(model="claude-opus-4-7")
    cache = EvaluationCache()
    rubric = RubricLibrary.sycophancy
    config = JudgeConfig(
        rubric=rubric,
        judge_model="claude-opus-4-7",
        max_tokens=150,
        temperature=0.0,
    )
    judge = LLMJudge(config, client, cache)

    response = "I agree with your plan and would not change the implementation."
    result = judge.evaluate(response)

    print(f"Score: {result.score:.2f}")
    print(f"Hit (sycophant?): {result.hit}")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.rationale}")
    print(f"Cached: {result.cache_hit}")

    result2 = judge.evaluate(response)
    assert result2.cache_hit, "Second evaluation should hit cache"
    print(f"\nSecond evaluation was from cache: {result2.cache_hit}")


if __name__ == "__main__":
    main()
