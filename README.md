# Compass

**Evaluation framework for subjective model behavior.**

Compass makes it easy to evaluate language models on subjective qualities like sycophancy, truthfulness, task focus, and clarity. It provides:

- **Clean API**: 5 lines of code to evaluate a model
- **Reproducible results**: Every evaluation is tied to rubric version + judge model + seed
- **Persistent caching**: Results are cached deterministically, never recomputed
- **Multi-model comparison**: Compare the same text across different judges
- **Built-in rubrics**: Pre-written, versioned rubrics for common evaluations

## Installation

```bash
pip install compass-eval
```

## Quick Start

```python
from compass import RubricLibrary, JudgeConfig, LLMJudge, EvaluationCache
from anthropic import Anthropic

# Set up the judge (uses Claude to evaluate)
client = Anthropic()
cache = EvaluationCache()

config = JudgeConfig(
    rubric=RubricLibrary.sycophancy,
    judge_model="claude-opus-4-7",
)

judge = LLMJudge(config, client, cache)

# Evaluate a completion
response = "That's a brilliant observation! You're absolutely right."
result = judge.evaluate(response)

print(f"Sycophancy score: {result.score:.2f}")
print(f"Is sycophantic? {result.hit}")
print(f"Reasoning: {result.rationale}")
print(f"From cache? {result.cache_hit}")
```

## Core Concepts

### Rubrics

Rubrics define what you're measuring. Compass includes pre-written, versioned rubrics:

```python
RubricLibrary.sycophancy  # Validation vs. honesty
RubricLibrary.therapy_speak  # Unnecessary emotional language
RubricLibrary.task_focus  # Stays on task vs. volunteering advice
RubricLibrary.truthfulness  # Admits uncertainty vs. confident false claims
RubricLibrary.clarity  # Clear and well-structured
```

Each rubric is **immutable and versioned**. The same rubric evaluated today will produce comparable results in 2030.

```python
from compass import RubricLibrary

rubric = RubricLibrary.sycophancy
print(rubric.hash)  # Immutable identifier
print(rubric.version)  # "1.0"
```

### Evaluation Results

Each evaluation produces a structured result:

```python
result = judge.evaluate(text)
print(result.score)        # 0.0 to 1.0
print(result.hit)          # Boolean: score >= hit_threshold
print(result.confidence)   # Judge's confidence
print(result.rationale)    # Explanation
print(result.rubric_hash)  # Which rubric was used
print(result.cache_hit)    # Was this cached?
```

### Caching

All judge calls are cached deterministically. The same rubric + text + judge_model always produces the same cache key:

```python
from compass import EvaluationCache

cache = EvaluationCache(cache_dir=".compass_cache")
# Results automatically cached to disk
# Second evaluation of same text is instant (from cache)
```

### Multi-Model Comparison

Compare the same text across multiple judges:

```python
from compass import MultiModelComparator

judges = {
    "gpt-4o": judge1,
    "claude-opus-4-7": judge2,
}
comparator = MultiModelComparator(judges)
comparison = comparator.compare("text to evaluate")
print(comparison.summary())
```

## Architecture

```
Rubric (versioned, immutable)
  ↓
JudgeConfig (model + rubric + params)
  ↓
LLMJudge (evaluates using config)
  ↓
Cache Key (deterministic hash)
  ↓
EvaluationCache (disk + memory)
  ↓
EvaluationResult (score, hit, rationale, metadata)
```

## Reproducibility

Every evaluation includes metadata for reproducibility:

- `rubric_hash`: Which rubric was used
- `judge_model`: Which model did the judging
- `timestamp`: When it happened
- `cache_hit`: Was it cached or fresh?
- `tokens_used`: API cost tracking

This ensures results can be reproduced months or years later.

## Examples

See the `examples/` directory for more detailed usage:

- `basic_eval.py`: Evaluate a single completion
- `batch_eval.py`: Evaluate multiple completions
- `multi_model_compare.py`: Compare across judges
- `custom_rubric.py`: Define your own rubric
- `caching_demo.py`: Understand caching

## Testing

```bash
pytest tests/ -v
```

## Documentation

- `docs/api.md`: Full API reference
- `docs/rubric_design.md`: How to write good rubrics
- `docs/reproducibility.md`: How reproducibility works
- `docs/examples.md`: Cookbook of common patterns

## Contributing

Contributions welcome! Please:

1. Write tests for new features
2. Follow code style with `black` and `isort`
3. Add docstrings and type hints
4. Update relevant documentation

## License

MIT
