#!/usr/bin/env python3
"""Show the difference between a fresh evaluation and a cache hit."""
import time

from compass import AnthropicClient, EvaluationCache, JudgeConfig, LLMJudge, RubricLibrary

client = AnthropicClient(model="claude-opus-4-7")

# Create two caches to show the difference
fresh_cache = EvaluationCache(cache_dir=".compass_cache_fresh")
persistent_cache = EvaluationCache(cache_dir=".compass_cache_persistent")

config = JudgeConfig(
    rubric=RubricLibrary.sycophancy,
    judge_model="claude-opus-4-7",
)

response = "I agree with the proposal and would not change the response."

print("=" * 60)
print("CACHE PERFORMANCE DEMO")
print("=" * 60)

# Fresh cache - everything is a miss
print("\n1. FRESH CACHE (no prior evaluations)")
judge_fresh = LLMJudge(config, client, fresh_cache)

start = time.time()
result1 = judge_fresh.evaluate(response)
elapsed_fresh = time.time() - start

print(f"   Time: {elapsed_fresh:.2f}s")
print(f"   From cache: {result1.cache_hit}")
print(f"   Score: {result1.score:.2f}")

# Persistent cache - first run is a miss
print("\n2. PERSISTENT CACHE (first evaluation)")
judge_persistent = LLMJudge(config, client, persistent_cache)

start = time.time()
result2 = judge_persistent.evaluate(response)
elapsed_first = time.time() - start

print(f"   Time: {elapsed_first:.2f}s")
print(f"   From cache: {result2.cache_hit}")

# Persistent cache - second run is a hit
print("\n3. PERSISTENT CACHE (second evaluation, same text)")

start = time.time()
result3 = judge_persistent.evaluate(response)
elapsed_cached = time.time() - start

print(f"   Time: {elapsed_cached:.3f}s")
print(f"   From cache: {result3.cache_hit}")
print(f"   Speedup: {elapsed_first / elapsed_cached:.0f}x faster")

# Verify results are identical
assert result2.score == result3.score
assert result2.hit == result3.hit
print(f"\n✓ Cached result matches original")

print("\n" + "=" * 60)
print("Repeated evaluations reuse the cached result and avoid another API call.")
print("=" * 60)
