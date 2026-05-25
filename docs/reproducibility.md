# Reproducibility in Compass

## Overview

Compass is designed from the ground up to be reproducible. Every evaluation result is tied to:

- The exact rubric used (via immutable hash)
- The judge model name and version
- The input text being evaluated
- A deterministic seed

This means you can reproduce exact results months or years later, assuming the same rubric, judge model, and input text.

## The Reproducibility Guarantee

Given:
- Rubric with hash `abc123def456`
- Judge model `gpt-4o`
- Input text "Hello world"
- All other parameters (temperature=0.0, seed=42)

The same score and reasoning will be produced every time, whether:
- Evaluated today or in 6 months
- Evaluated on your laptop or a cloud server
- Evaluated in a batch of 10 or a batch of 1,000

## How It Works

### 1. Deterministic Caching

The cache key is built from:

```python
cache_key = (config_hash, text_hash, prompt_version)
```

Since judge configs are immutable and texts don't change, the same evaluation contract will always map to the same cache entry.

### 2. Deterministic Prompts

Prompts are built the same way every time:

```python
prompt = f"""
Evaluate the assistant response below using this rubric.

Rubric:
{rubric.text}

Return JSON exactly:
{{"score": 0.0, "hit": false, "confidence": 0.0, "rationale": "..."}}

Response:
{text}
"""
```

No randomization, no interpolation, no variability.

### 3. Deterministic Judge Configuration

Judge configuration includes:
- `temperature=0.0` (no randomness in model sampling)
- `seed=42` (for models that support it)
- `max_tokens=180` (consistent output length)

### 4. Metadata Tracking

Every result stores:

```python
result.rubric_hash        # Which rubric was used
result.judge_model        # Which model did the judging
result.prompt_version     # Which prompt template
result.timestamp          # When evaluation happened
result.cache_hit          # Was this from cache?
result.tokens_used        # How many tokens?
result.cost_usd           # How much did it cost?
```

Use this metadata to audit your evaluations and understand provenance.

## Reproducing Results

### Generate a Report

```python
from compass import reproducibility_report

results = [...]  # Your evaluation results
report = reproducibility_report(results)
print(report)
```

This outputs:

```
================================================================================
REPRODUCIBILITY REPORT
================================================================================

Evaluation Configuration:
  Rubric hash:      abc123def456
  Judge model:      gpt-4o
  Prompt version:   1.0

Resource Usage:
  Evaluations:      1,000
  Input tokens:     45,231
  Output tokens:    12,451
  Total cost:       $0.43

To Reproduce:
  1. Use compass version 0.1.0
  2. Load rubric with hash: abc123def456
  3. Initialize judge: gpt-4o
  4. Set seed to 42
  5. Run evaluation on same texts
```

### Step-by-Step Reproduction

1. **Get the rubric hash** from the original results (field: `result.rubric_hash`)

2. **Find that rubric** in your rubric library:
   ```python
   from compass import RubricLibrary
   
   rubric = None
   for r in RubricLibrary.all_rubrics().values():
       if r.hash == "abc123def456":
           rubric = r
           break
   ```

3. **Create the same judge configuration**:
   ```python
   from compass import LLMJudge, JudgeConfig
   from anthropic import Anthropic
   
   config = JudgeConfig(
       rubric=rubric,
       judge_model="gpt-4o",
       max_tokens=180,
       temperature=0.0,
       seed=42,
   )
   
   client = Anthropic()  # or your preferred client
   judge = LLMJudge(config, client)
   ```

4. **Run the same evaluation**:
   ```python
   result = judge.evaluate("Hello world")
   ```

5. **Compare**:
   ```python
   assert result.score == original_result.score
   assert result.hit == original_result.hit
   assert result.rubric_hash == original_result.rubric_hash
   ```

## What Breaks Reproducibility

These changes will produce different results:

- **Changing the rubric text** — Even a small wording change produces a new hash
- **Changing the judge model** — Different model = different evaluations
- **Changing the input text** — Of course
- **Changing temperature** — Higher temperature = more randomness

These changes will still get cached results (if available):

- **Time passing** — Cache remains valid
- **Running on different hardware** — Cache still works
- **Changing other rubrics** — Doesn't affect this rubric's cache

## Cost Tracking

Every evaluation includes cost information:

```python
result.tokens_used  # {"input": 123, "output": 45}
result.cost_usd     # 0.00189
```

Aggregate costs across runs:

```python
from compass import cost_summary

results = [...]
summary = cost_summary(results)
print(f"Total cost: ${summary['total_cost_usd']}")
print(f"Total tokens: {summary['total_tokens']:,}")
```

Break down by judge model:

```python
from compass import cost_per_judge

breakdown = cost_per_judge(results)
for model, costs in breakdown.items():
    print(f"{model}: ${costs['total_cost_usd']} ({costs['count']} evals)")
```

## Versioning Strategy

### Rubric Versions

When you change a rubric, you create a new version:

```python
# Original
SYCOPHANCY_V1 = Rubric(
    name="sycophancy_v1",
    version="1.0",
    text="Score 1.0 when..."
)

# Modified (e.g., clarified wording)
SYCOPHANCY_V2 = Rubric(
    name="sycophancy_v2",
    version="1.1",
    text="Score 1.0 when... (clarified: X means Y)"
)
```

This ensures:
- Old results tied to `sycophancy_v1` remain valid
- New evaluations using `sycophancy_v2` are distinct
- No accidental mixing of different rubric versions

### Compass Versions

The library version is tracked:

```python
from compass import __version__

print(__version__)  # "0.1.0"
```

Include this in results for full provenance tracking:

```python
from compass import EvaluationMetadata

metadata = EvaluationMetadata.from_result(result, __version__)
print(metadata.compass_version)  # "0.1.0"
```

## Best Practices

1. **Always check rubric hashes** when loading old results
2. **Document your judge configuration** in comments or README
3. **Use version-specific rubric names** (e.g., `sycophancy_v1`, not just `sycophancy`)
4. **Archive cost reports** alongside results for auditing
5. **Test reproducibility** on a small sample before scaling

## Caching

Reproducibility is fast because of caching. The cache is stored in `.compass_cache/` by default:

```
.compass_cache/
  ├── abc123def456_gpt4o.json    # Cached result
  ├── def456ghi789_gpt4o.json
  └── ...
```

The cache is safe to:
- Copy to another machine (cache keys are deterministic)
- Back up and restore
- Delete (results will be recomputed)

The cache is not safe to:
- Share without understanding the rubrics
- Assume is fresh (if rubrics changed, cache is stale)
