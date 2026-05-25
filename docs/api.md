# Compass API Reference

Complete reference for the Compass evaluation framework.

## Core Classes

### Rubric

Immutable definition of an evaluation criterion.

```python
from compass import Rubric

rubric = Rubric(
    name="sycophancy",
    version="1.0",
    category="constitutional",
    text="Score 1.0 when the response validates or flatters...",
    hit_threshold=0.5,
)

# Properties
rubric.hash          # Immutable hash - changes if text changes
rubric.name          # "sycophancy"
rubric.version       # "1.0"
rubric.category      # "constitutional"
rubric.text          # The actual evaluation criteria
rubric.hit_threshold # 0.5 (score >= 0.5 means hit=True)
```

### RubricLibrary

Pre-defined rubrics for common evaluation tasks.

```python
from compass import RubricLibrary

# Access by attribute
rubric = RubricLibrary.sycophancy
rubric = RubricLibrary.truthfulness
rubric = RubricLibrary.therapy_speak
rubric = RubricLibrary.clarity
rubric = RubricLibrary.task_focus

# List all available rubrics
all_rubrics = RubricLibrary.all_rubrics()

# Get by name
rubric = RubricLibrary.get("sycophancy")
```

Available rubrics:
- `sycophancy` — Flattery and validation without pushback
- `truthfulness` — Factual accuracy and honesty
- `therapy_speak` — Therapeutic/emotional language overuse
- `clarity` — Clear and understandable explanations
- `task_focus` — Staying on task vs. going off-topic

### JudgeConfig

Configuration for running a judge evaluation.

```python
from compass import JudgeConfig, RubricLibrary

config = JudgeConfig(
    rubric=RubricLibrary.sycophancy,
    judge_model="claude-opus-4-7",
    max_tokens=180,           # Default 180
    temperature=0.0,          # Default 0.0 (deterministic)
    seed=42,                  # Default 42
)

# Properties
config.config_hash      # Stable ID for this config
config.rubric          # The rubric being used
config.judge_model     # "claude-opus-4-7"
```

### EvaluationResult

Result from a single judge evaluation.

```python
from compass import LLMJudge, JudgeConfig

# After evaluating...
result = judge.evaluate("Some text")

# Core results
result.name            # Rubric name ("sycophancy")
result.score           # 0.0-1.0 (float)
result.hit             # True/False
result.confidence      # Judge's confidence (optional)
result.rationale       # Explanation of the score

# Reproducibility metadata
result.rubric_hash     # Which rubric version was used
result.judge_model     # "claude-opus-4-7"
result.prompt_version  # "1.0"
result.timestamp       # ISO 8601 timestamp
result.cache_hit       # Was this from cache?

# Resource tracking
result.tokens_used     # {"input": 123, "output": 45}
result.cost_usd        # 0.00189

# Methods
result.to_dict()       # Serialize to dict
```

### LLMJudge

Judge that uses an LLM to evaluate text.

```python
from compass import LLMJudge, JudgeConfig, EvaluationCache
from anthropic import Anthropic

client = Anthropic()
cache = EvaluationCache()

config = JudgeConfig(
    rubric=RubricLibrary.sycophancy,
    judge_model="claude-opus-4-7",
)

judge = LLMJudge(config, client, cache)

# Evaluate
result = judge.evaluate("Some response text")
```

### EvaluationCache

Persistent cache for judge results.

```python
from compass import EvaluationCache

# Default location (.compass_cache/)
cache = EvaluationCache()

# Custom location
cache = EvaluationCache(cache_dir="/tmp/my_cache")

# Methods
cached = cache.get("cache_key")        # Get cached result
cache.put("cache_key", result)         # Store result
stats = cache.stats()                  # Cache statistics

# Cache is safe to:
# - Copy to another machine
# - Back up and restore
# - Delete (results will be recomputed)
```

### ComparisonResult

Result from comparing the same text across multiple judges.

```python
from compass import MultiModelComparator

comparison = comparator.compare("Some text")

# Access individual judge results
comparison.judges     # Dict[judge_model, EvaluationResult]
comparison.text       # The text that was evaluated
comparison.rubric_name # Name of the rubric

# Metrics
comparison.agreement_score()   # 1.0 = perfect, 0.0 = max disagreement
comparison.hit_agreement()     # Fraction agreeing on hit/miss
comparison.score_range()       # (min_score, max_score)

# Display
print(comparison.summary())    # Human-readable summary
```

### MultiModelComparator

Compare the same text across multiple judge models.

```python
from compass import MultiModelComparator, LLMJudge, JudgeConfig

judges = {}
for model in ["claude-opus-4-7", "claude-sonnet-4-6"]:
    judges[model] = LLMJudge(
        JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model=model,
        ),
        client,
    )

comparator = MultiModelComparator(judges)

# Evaluate
comparison = comparator.compare("Text to compare")

# Batch evaluate
results = comparator.compare_batch(["text1", "text2", "text3"])

# Statistics
stats = comparator.agreement_stats(results)
# Returns: {mean_agreement, std_agreement, min_agreement, max_agreement, ...}
```

## Utility Functions

### cost_summary

Aggregate costs across evaluation results.

```python
from compass import cost_summary

summary = cost_summary([result1, result2, result3])

# Returns:
# {
#   "total_cost_usd": 0.0456,
#   "total_input_tokens": 1250,
#   "total_output_tokens": 456,
#   "total_tokens": 1706,
#   "results_count": 3
# }
```

### cost_per_judge

Break down costs by judge model.

```python
from compass import cost_per_judge

breakdown = cost_per_judge([result1, result2, result3])

# Returns:
# {
#   "claude-opus-4-7": {
#     "total_cost_usd": 0.0234,
#     "count": 2,
#     "avg_cost_per_eval": 0.0117
#   },
#   "gpt-4o": {
#     "total_cost_usd": 0.0222,
#     "count": 1,
#     "avg_cost_per_eval": 0.0222
#   }
# }
```

### reproducibility_report

Generate a human-readable reproducibility report.

```python
from compass import reproducibility_report, EvaluationMetadata

report = reproducibility_report(results)
print(report)

# Or with metadata
metadata = EvaluationMetadata.from_result(results[0], "0.1.0")
report = reproducibility_report(results, metadata)
```

### EvaluationMetadata

Capture complete evaluation context.

```python
from compass import EvaluationMetadata

metadata = EvaluationMetadata(
    compass_version="0.1.0",
    rubric_hash="abc123def456",
    judge_model="claude-opus-4-7",
    seed=42,
    timestamp="2026-05-21T10:00:00",
    python_version="3.9.7",
)

# Or create from a result
metadata = EvaluationMetadata.from_result(result, compass_version="0.1.0")

# Methods
d = metadata.to_dict()  # Serialize to dict
```

## Imports

```python
# Main classes
from compass import (
    Rubric,
    RubricLibrary,
    JudgeConfig,
    LLMJudge,
    EvaluationResult,
    EvaluationCache,
    ComparisonResult,
    MultiModelComparator,
)

# Utilities
from compass import (
    EvaluationMetadata,
    reproducibility_report,
    cost_summary,
    cost_per_judge,
)

# Version
from compass import __version__
```

## Common Patterns

### Single evaluation

```python
judge = LLMJudge(config, client, cache)
result = judge.evaluate(text)
```

### Batch evaluation with cost tracking

```python
results = []
for text in texts:
    result = judge.evaluate(text)
    results.append(result)

summary = cost_summary(results)
print(f"Total cost: ${summary['total_cost_usd']}")
```

### Compare multiple judges

```python
comparator = MultiModelComparator({"model1": judge1, "model2": judge2})
comparison = comparator.compare(text)
print(comparison.summary())
```

### Reproducibility check

```python
metadata = EvaluationMetadata.from_result(result, __version__)
report = reproducibility_report([result], metadata)
print(report)
```
