# Compass 0.1.0 Release Notes

**Evaluating subjective model behavior, reproducibly.**

## What is Compass?

Compass is an evaluation framework for measuring subjective qualities in language models: sycophancy, truthfulness, therapy-speak, clarity, and task focus. It provides reproducible evaluation with deterministic caching, multi-model comparison, and full cost tracking.

## Key Features

### 1. Immutable, Versioned Rubrics
```python
rubric = RubricLibrary.sycophancy
print(rubric.hash)  # Immutable ID - never changes
```
Rubrics are frozen dataclasses. Same rubric today = same rubric in 2026. Results are comparable across time and teams.

### 2. Deterministic Caching
```python
result1 = judge.evaluate("text")  # API call, cost ~$0.01
result2 = judge.evaluate("text")  # From cache, instant
```
Cache key is deterministic: `(rubric_hash, text_hash, judge_model)`. Same input always produces same output.

### 3. Multi-Model Comparison
```python
comparator = MultiModelComparator({"gpt-4o": judge1, "claude-opus-4-7": judge2})
comparison = comparator.compare(text)
print(comparison.agreement_score())  # How much do judges agree?
```
Compare judges side-by-side. Measure disagreement. Find edge cases.

### 4. Complete Metadata & Reproducibility
```python
result.rubric_hash      # Which rubric version
result.judge_model      # Which model did the judging
result.timestamp        # When it happened
result.cache_hit        # Was it cached?
result.tokens_used      # Cost information
result.cost_usd         # Actual cost
```
Every result is tied to rubric version + judge model + seed. Reproduce exact results months later.

### 5. Built-in Cost Tracking
```python
summary = cost_summary(results)
# Returns: total_cost, total_tokens, per-model breakdown
```
Track costs across thousands of evaluations. Budget intelligently.

## What's Included

- **89 tests** across all modules (rubrics, judges, caching, comparison, reproducibility)
- **5 pre-defined rubrics** for measuring constitutional AI properties
- **5 working examples** (basic eval, batch processing, multi-model comparison, custom rubrics, caching demo)
- **Complete documentation** (API reference, rubric design guide, reproducibility strategy)
- **Multi-model support** (works with Claude, GPT, and any LLM via client abstraction)

## Architecture Highlights

```
Rubric (immutable, versioned)
  ↓ [deterministic hash]
JudgeConfig (model + params)
  ↓
LLMJudge (evaluates text)
  ↓ [check cache first]
EvaluationCache (disk + memory)
  ↓
EvaluationResult (score + metadata)
```

## Getting Started

### Installation

```bash
pip install compass-eval
```

### 5-Minute Example

```python
from compass import RubricLibrary, JudgeConfig, LLMJudge, EvaluationCache
from anthropic import Anthropic

client = Anthropic()
cache = EvaluationCache()

# Create a judge for measuring sycophancy
config = JudgeConfig(
    rubric=RubricLibrary.sycophancy,
    judge_model="claude-opus-4-7",
)
judge = LLMJudge(config, client, cache)

# Evaluate a response
response = "You're absolutely right! That's a brilliant idea."
result = judge.evaluate(response)

print(f"Sycophancy score: {result.score:.2f}")
print(f"Is sycophantic? {result.hit}")
print(f"Reasoning: {result.rationale}")
```

See `examples/` for more detailed use cases.

## Documentation

- **docs/api.md** — Complete API reference
- **docs/rubric_design.md** — How to write good rubrics
- **docs/reproducibility.md** — How to reproduce results
- **README.md** — Quick start guide
- **CONTRIBUTING.md** — How to contribute

## Known Limitations & Future Work

**Current Version (0.1.0):**
- Single client per judge (choose Claude or GPT, not both simultaneously)
- Cache stored as individual JSON files (scales to ~10k evaluations)
- No async/await support for batch processing

**Planned for 0.2.0:**
- Async/await for large-scale evaluations
- Rubric YAML loading
- Pairwise preference scoring

**Planned for 0.3.0+:**
- Integration with HELM/HuggingFace suites
- Constitutional Compliance Benchmark
- Cloud cache synchronization

## Deployment Notes

- Requires Python 3.9+
- Dependencies: pydantic, openai, anthropic
- Cache directory defaults to `.compass_cache/` (configurable)
- No database or external services required

## Questions & Support

- **GitHub Issues:** Report bugs or request features
- **Discussions:** Ask questions and share rubrics
- **Contributing:** We welcome rubrics, examples, and improvements

---

**Compass 0.1.0 represents a complete, tested, documented evaluation framework for subjective model behavior. It's ready for production use, research, and community contribution.**
