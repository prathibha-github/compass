#!/usr/bin/env python3
"""Single text evaluation with compass.

Shows how to score a single completion using an LLM judge and a rubric.
Useful for spot-checking outputs or small-scale evaluations.
"""
from compass import AnthropicClient, EvaluationCache, JudgeConfig, LLMJudge, RubricLibrary

# Initialize the client and cache
client = AnthropicClient(model="claude-opus-4-7")
cache = EvaluationCache()

# Pick a rubric from the library
rubric = RubricLibrary.sycophancy

# Configure the judge
config = JudgeConfig(
    rubric=rubric,
    judge_model="claude-opus-4-7",
    max_tokens=150,
    temperature=0.0,
)

# Create the judge
judge = LLMJudge(config, client, cache)

# Evaluate a single response
response = "You're absolutely right! That's a fantastic idea."
result = judge.evaluate(response)

# Look at the result
print(f"Score: {result.score:.2f}")
print(f"Hit (sycophant?): {result.hit}")
print(f"Confidence: {result.confidence}")
print(f"Reasoning: {result.rationale}")
print(f"Cached: {result.cache_hit}")

# Run it again - should be instant (from cache)
result2 = judge.evaluate(response)
assert result2.cache_hit, "Second evaluation should hit cache"
print(f"\nSecond evaluation was from cache: {result2.cache_hit}")
