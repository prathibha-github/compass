#!/usr/bin/env python3
"""Evaluate multiple texts in a batch.

Processes many completions efficiently with caching. Useful for benchmarking
models or evaluating datasets. Caching makes re-runs nearly instant.
"""
from compass import (
    AnthropicClient,
    EvaluationCache,
    JudgeConfig,
    LLMJudge,
    RubricLibrary,
    cost_summary,
)

client = AnthropicClient(model="claude-opus-4-7")
cache = EvaluationCache()

# Set up judge for detecting therapy-speak patterns
rubric = RubricLibrary.therapy_speak
config = JudgeConfig(
    rubric=rubric,
    judge_model="claude-opus-4-7",
)
judge = LLMJudge(config, client, cache)

# Some example completions to evaluate
completions = [
    "I'm here to help you process your feelings about this.",
    "The function is broken because the return statement is missing.",
    "It's important to validate your experience and take time for self-care.",
    "Try adding console.log statements to debug the issue.",
]

print("Evaluating batch...")
results = []
for i, completion in enumerate(completions, 1):
    result = judge.evaluate(completion)
    results.append(result)
    status = "CACHED" if result.cache_hit else "FRESH"
    print(f"{i}. Score={result.score:.2f}, Hit={result.hit} [{status}]")

# Get cost breakdown
summary = cost_summary(results)
print(f"\nCost Summary:")
print(f"  Total: ${summary['total_cost_usd']:.4f}")
print(f"  Tokens: {summary['total_tokens']:,}")

# Now run again - everything should be cached
print("\n\nRe-running same batch...")
results2 = []
for i, completion in enumerate(completions, 1):
    result = judge.evaluate(completion)
    results2.append(result)
    print(f"{i}. Hit={result.hit} [CACHED]")

# Verify same results
for r1, r2 in zip(results, results2):
    assert r1.score == r2.score
    assert r1.hit == r2.hit
print("\nCached rerun matched the original results.")
