# Compass

Evaluation framework for subjective model behavior.

Compass evaluates model outputs on rubric-based qualities such as sycophancy,
truthfulness, task focus, and clarity. It provides:

- Reproducible results tied to rubric version, judge model, and seed
- Deterministic caching for repeated evaluations
- Multi-model comparison on shared texts
- Built-in versioned rubrics for common evaluation tasks

## Installation

```bash
pip install compass-eval
```

Install provider SDKs as needed:

```bash
pip install "compass-eval[anthropic]"
pip install "compass-eval[openai]"
pip install "compass-eval[google]"
```

## Quick Start

```python
from compass import AnthropicClient, EvaluationCache, JudgeConfig, LLMJudge, RubricLibrary

# Set up the judge (uses Claude to evaluate)
client = AnthropicClient(model="claude-opus-4-7")
cache = EvaluationCache()

config = JudgeConfig(
    rubric=RubricLibrary.sycophancy,
    judge_model="claude-opus-4-7",
)

judge = LLMJudge(config, client, cache)

# Evaluate a completion
response = "I agree with your plan and would not change anything."
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

See [ARCHITECTURE.md](ARCHITECTURE.md) for the module layout and request flow.

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

## Long-Running Evaluations

Evaluate large benchmarks without losing progress. The `CheckpointManager` enables pause/resume:

```python
from compass import CheckpointManager, LLMJudge

checkpoint = CheckpointManager("results/benchmark.jsonl")
completed = checkpoint.load()  # Resume if interrupted

for model, prompt, sample in evaluation_set:
    identity = (model, "suite", "detector", prompt, "default", sample)
    if identity in completed:
        continue  # Already done
    
    result = judge.evaluate(prompt)
    checkpoint.save({
        "model": model,
        "suite": "suite",
        "detector": "detector",
        "prompt_id": prompt,
        "condition": "default",
        "sample_idx": sample,
        "score": result.score,
    })
```

**Features:**
- JSONL format with immediate persistence
- Sample-level granularity for partial retries
- Backward compatible with legacy checkpoints
- Robust error handling for corruption recovery

See `docs/CHECKPOINT_SYSTEM.md` for full details.

## Judge Reliability & Auditing

Measure inter-judge disagreement and detect drift:

```python
from compass import JudgeReliabilityAuditor

auditor = JudgeReliabilityAuditor()

# Measure agreement between two judges
agreement = auditor.calculate_agreement(
    judge1_scores=[0.1, 0.8, 0.2],
    judge2_scores=[0.15, 0.75, 0.3],
    hit_threshold=0.5,
)
print(f"Agreement: {agreement['agreement_rate']:.1%}")
print(f"CI: [{agreement['agreement_ci_low']:.1%}, {agreement['agreement_ci_high']:.1%}]")

# Detect judge drift on benign requests
drift = auditor.detect_drift([0.05, 0.10, 0.08], expected_low=True)
print(f"Concern level: {drift['concern_level']}")  # "none", "warning", or "critical"
```

**Features:**
- Wilson score intervals for reliable confidence bounds
- Benign control test for drift detection
- Disagreement sample identification
- Multi-judge system design patterns

See `docs/JUDGE_RELIABILITY.md` for full details.

## Pairwise Model Comparison

Rank models head-to-head on shared evaluations:

```python
from compass import PairwiseRanker

ranker = PairwiseRanker()

# Add evaluations for multiple models on same prompts
for model in ["gpt-4o", "claude-opus", "llama"]:
    for prompt_id in ["p1", "p2", "p3"]:
        result = evaluate(model, prompt_id)
        ranker.add_record(
            suite="task_focus",
            model=model,
            comparison_key=(prompt_id, "neutral"),
            score=result.score,
            metadata={"task_type": "coding"}
        )

# Generate rankings
rankings = ranker.rank("task_focus", min_matches=2)
for model, wins, total in rankings["overall_ranking"]:
    print(f"{model}: {wins:.1f}/{total} wins ({100*wins/total:.0f}%)")

# Segment by task type
segmented = ranker.rank_by_segment("task_focus", segment_by="task_type")
print(segmented["coding"]["overall_ranking"])
```

**Features:**
- Lower score = better (fewer violations)
- Win/loss/tie scoring with confidence intervals
- Segmented analysis by task type or custom metadata
- Pairwise detail breakdowns

See `docs/PAIRWISE_COMPARISON.md` for full details.

## Local Model Evaluation

Evaluate Ollama models for free:

```python
from compass import OllamaClient, OpenAIClient, LLMJudge, JudgeConfig, RubricLibrary

# Generate locally (free)
generator = OllamaClient(model="llama3.1:latest")
completion = generator.complete(
    prompt="Fix this Python bug: ...",
    max_tokens=100,
)
print(f"Cost: ${completion.cost_usd}")  # $0.00

# Judge with cloud (cheap)
config = JudgeConfig(
    rubric=RubricLibrary.task_focus,
    judge_model="gpt-4o-mini",
)
judge_client = OpenAIClient(model="gpt-4o-mini")
judge = LLMJudge(config, judge_client)
result = judge.evaluate(completion.completion)
print(f"Total cost: ${completion.cost_usd + result.cost_usd:.4f}")
```

**Features:**
- Free local inference via Ollama
- No API keys needed for generation
- Hybrid: cheap inference + accurate evaluation
- Works with existing judge infrastructure

See `examples/constitutional_compliance_benchmark.py` for a larger local-vs-cloud setup.

## Documentation

- `docs/api.md`: Full API reference
- `docs/rubric_design.md`: How to write good rubrics
- `docs/reproducibility.md`: How reproducibility works
- `docs/CHECKPOINT_SYSTEM.md`: Resumable evaluations guide
- `docs/JUDGE_RELIABILITY.md`: Judge auditing and drift detection
- `docs/PAIRWISE_COMPARISON.md`: Model ranking methodology
- `docs/CONSTITUTIONAL_COMPLIANCE.md`: Compliance benchmark guide

## Contributing

For development setup and contribution guidelines, see `CONTRIBUTING.md`.
