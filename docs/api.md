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

## Benchmark API

The shared benchmark core lives under `compass.benchmark`.

### BenchmarkSpec and Run Presets

Use `BenchmarkSpec` plus benchmark-owned presets to describe a benchmark
family without embedding policy defaults in a CLI wrapper.

```python
from compass.benchmark import (
    BenchmarkPolicyDefaults,
    BenchmarkPrompt,
    BenchmarkRunPreset,
    BenchmarkSpec,
)
from compass.rubrics.library import RubricLibrary

spec = BenchmarkSpec(
    name="toy_benchmark",
    version="1.0",
    prompts_by_rubric={
        "clarity": (
            BenchmarkPrompt(
                id="p1",
                text="Explain caching.",
                task_type="conceptual_explanation",
            ),
        ),
    },
    rubrics_by_name={"clarity": RubricLibrary.clarity},
    run_presets={
        "default": BenchmarkRunPreset(
            models=("llama3.1",),
            samples=2,
            judge_model="llama3.1",
            output_dir="results/toy_benchmark",
            policy=BenchmarkPolicyDefaults(
                token_budgets={"default": 150},
                analysis_lanes=("summary", "pairwise"),
            ),
        ),
    },
)

run_config = spec.make_run_config()
```

Key types:
- `BenchmarkPrompt`: prompt id, text, and stable `task_type`
- `BenchmarkPolicyDefaults`: benchmark-owned token budgets, fairness policy, quality filter, analysis lanes
- `BenchmarkRunPreset`: named default run settings
- `BenchmarkRunConfig`: resolved config after preset selection and overrides
- `BenchmarkSpec`: immutable benchmark definition

### BenchmarkRunner Contract

Benchmark registration enforces a `BenchmarkRunner` contract with these
methods:

- `validate_run_config(run_config) -> BenchmarkRunConfig`
- `generate(run_config) -> Path`
- `evaluate(generations_path, run_config) -> Path`
- `analyze(evaluations_path, run_config) -> dict`
- `rank(evaluations_path, run_config) -> None`
- `validate_report(evaluations_path, run_config) -> Sequence[str]`

Use `SharedBenchmarkRunner` when the default shared core is enough:

```python
from compass.benchmark import SharedBenchmarkRunner

runner = SharedBenchmarkRunner(spec)
```

### Registry Helpers

Register and look up benchmark families through the registry:

```python
from compass.benchmark import (
    get_benchmark_runner,
    get_benchmark_spec,
    list_benchmark_specs,
    register_benchmark_spec,
)

register_benchmark_spec(spec)
runner = get_benchmark_runner(spec.name)
spec = get_benchmark_spec("constitutional_compliance")
available = list_benchmark_specs()
```

### Core Benchmark Operations

The shared runner builds on reusable generation, evaluation, reporting, and
validation helpers:

```python
from pathlib import Path

from compass.benchmark import (
    analyze_results,
    evaluate_completions,
    generate_completions,
    print_summary,
    validate_benchmark_report,
)

generations_path = generate_completions(
    models=["llama3.1"],
    benchmark_spec=spec,
    samples=2,
    output_dir=Path("results/toy_benchmark"),
)
evaluations_path = evaluate_completions(
    generations_path=generations_path,
    benchmark_spec=spec,
    judge_model="llama3.1",
    output_dir=Path("results/toy_benchmark"),
)
stats = analyze_results(evaluations_path, Path("results/toy_benchmark"))
print_summary(stats, evaluations_path)
errors = validate_benchmark_report(evaluations_path)
```

### Benchmark Schemas

Persisted benchmark outputs use benchmark-owned schema fields:

- `benchmark_name`
- `benchmark_version`
- `benchmark_schema_version`
- `benchmark_record_type`

Helpers:
- `migrate_generation_record(...)`
- `migrate_evaluation_record(...)`
- `generation_identity(...)`
- `evaluation_identity(...)`

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

# Benchmark core
from compass.benchmark import (
    BenchmarkPolicyDefaults,
    BenchmarkPrompt,
    BenchmarkRunPreset,
    BenchmarkSpec,
    SharedBenchmarkRunner,
    analyze_results,
    evaluate_completions,
    generate_completions,
    get_benchmark_runner,
    get_benchmark_spec,
    register_benchmark_spec,
    validate_benchmark_report,
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
