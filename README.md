# Compass

A framework for repeatable experiments on model behavior.

When a model does something unexpected, define a test suite and compare the
behavior across model versions, prompts, and conditions. Compass provides the
shared clients, scoring, cache, and checkpoints so each experiment can reuse
that infrastructure.

Compass evaluates model outputs on rubric-based qualities such as sycophancy,
truthfulness, task focus, and clarity. It provides:

- Results tied to rubric hashes, judge models, and prompt-template versions
- Deterministic caching for repeated evaluations
- Multi-model comparison on shared texts
- Built-in versioned rubrics for common evaluation tasks
- Reusable `TicSuite` definitions with heuristic and LLM-judge detectors
- Separate generation and scoring, resumable checkpoints, and benchmark quality checks

[Compass Experiments](https://github.com/prathibha-github/helm-eval-lex-quirks)
uses this library for experiment orchestration, scenario loading, and reports.
Compass owns the shared evaluation primitives and does not import the harness.

See [limitations and work left to do](docs/LIMITATIONS.md) for the scoring and
reproducibility issues that still need attention.

## Installation

Python 3.9 or newer is required. To use the code and examples in this checkout:

```bash
git clone https://github.com/prathibha-github/compass.git
cd compass
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Install the extras for the providers you use, from the repository root:

```bash
python -m pip install -e ".[anthropic]"
python -m pip install -e ".[openai]"
python -m pip install -e ".[google]"
```

The distribution is named `compass-eval`; the Python import is `compass`.
Cloud clients read `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GOOGLE_API_KEY`
unless a key is passed explicitly. Install `ollama` separately for local models.

## Quick Start

Install the Anthropic extra and set `ANTHROPIC_API_KEY` before running this
example. It makes a paid API call on a cache miss. Choose a model available to
your account and use the same model ID in the client and judge configuration.

```python
from compass import AnthropicClient, EvaluationCache, JudgeConfig, LLMJudge, RubricLibrary

# Set up the judge (uses Claude to evaluate)
model = "claude-haiku-4-5-20251001"
client = AnthropicClient(model=model)
cache = EvaluationCache()

config = JudgeConfig(
    rubric=RubricLibrary.sycophancy,
    judge_model=model,
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

Rubrics are immutable and versioned. Their hashes let you identify the scoring
definition used for a result; they do not guarantee that a hosted judge will
give the same answer in a later run. Built-in rubrics score unwanted behavior
higher, including `clarity`, where a high score means confusing writing.

```python
from compass import RubricLibrary

rubric = RubricLibrary.sycophancy
print(rubric.hash)  # Immutable identifier
print(rubric.version)  # "1.0"
```

### Evaluation Results

Each evaluation produces a structured result:

```python
result = judge.evaluate("The answer is uncertain; here is what I can establish.")
print(result.score)        # 0.0 to 1.0
print(result.hit)          # Boolean: score >= hit_threshold
print(result.confidence)   # Judge's confidence
print(result.rationale)    # Explanation
print(result.rubric_hash)  # Which rubric was used
print(result.cache_hit)    # Was this cached?
```

For prompt-dependent judgments, pass the original request as context:

```python
result = judge.evaluate(
    "Take a break before working on this.",
    context="Please help me debug this function.",
)
```

### Caching

`LLMJudge` uses a persistent cache by default. Its key combines the judge
configuration hash, a hash of the text, context, and client-reported request
settings, and the prompt-template/scoring versions. The configuration hash
includes the rubric hash, judge model, maximum tokens, temperature, and seed.

```python
from compass import EvaluationCache

cache = EvaluationCache(cache_dir=".compass_cache")
judge = LLMJudge(config, client, cache)
first = judge.evaluate("A response to evaluate.")
second = judge.evaluate("A response to evaluate.")
print(second.cache_hit)  # True: reused the saved evaluation
```

`JudgeConfig.seed` participates in cache identity but is not forwarded to
provider calls by `LLMJudge`. Caching reuses an earlier result; recomputing
against a hosted model is not guaranteed to reproduce it.

### Multi-Model Comparison

Compare the same text across multiple judges using the same rubric. Continuing
from Quick Start, install the OpenAI extra and set `OPENAI_API_KEY` as well:

```python
from compass import MultiModelComparator, OpenAIClient

judges = {
    model: judge,
    "gpt-4o-mini": LLMJudge(
        JudgeConfig(rubric=RubricLibrary.sycophancy, judge_model="gpt-4o-mini"),
        OpenAIClient(model="gpt-4o-mini"),
        cache,
    ),
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

## Benchmarking

Compass provides two related execution APIs:

- `generate_suite_completions` and `evaluate_suite_completions` run `TicSuite`
  experiments with prompt conditions and multiple detectors. They write
  `suite_generations.jsonl` and `suite_evaluations_<judge>.jsonl`. The suite
  evaluator passes the original user prompt as context to LLM judges.
- `BenchmarkSpec` and `SharedBenchmarkRunner` run registered benchmark families
  with run presets, artifact policies, and quality validation.

The benchmark core is organized into:

- `compass.benchmark.specs` for benchmark contracts
- `compass.benchmark.registry` for built-in benchmark registration
- `compass.benchmark.runner` for generation and evaluation loops
- `compass.benchmark.reporting` for summaries and pairwise ranking
- `compass.benchmark.validation` for quality-gated report checks

Use the benchmark playbook when adding new benchmark families:

- [docs/BENCHMARK_PLAYBOOK.md](docs/BENCHMARK_PLAYBOOK.md)
- [docs/CONSTITUTIONAL_COMPLIANCE.md](docs/CONSTITUTIONAL_COMPLIANCE.md)

Benchmark conclusions should be interpreted with quality diagnostics, not raw
violation rates alone. Use:

- `quality_flagged_pct`
- `token_cap_pct`
- `fragment_pct`
- `quality_filtered_hit_rate`

Benchmark output directories also carry the run policy and artifact semantics:

- `generations.jsonl` and `evaluations_<judge>.jsonl` are append-only
  checkpoints
- `benchmark_run_policy.json` records the resolved preset and effective policy
- `benchmark_run_outcome.json` records per-phase expected rows, successful
  rows, runtime failures, and complete/partial status
- shared benchmark analysis reads report artifacts strictly and fails on
  malformed rows rather than skipping them
- runtime model failures are logged and warned about explicitly; successful rows
  are still preserved for resume and inspection

Do not treat a benchmark win as a blanket model-quality claim. Interpretation
is benchmark-local and depends on the prompt set, rubric design, judge model,
and artifact quality diagnostics.

The suite API verifies saved input fingerprints before resuming. Changed prompts,
conditions, temperature, token budgets, detector definitions, or saved completions
require an explicit fresh run or rescore. Legacy suite rows without fingerprints
cannot resume. Use `evaluate_suite_completions(..., resume=False)` to re-score
saved generations without regenerating them. Keep each JSONL file with its
`.manifest.json` coverage file. Failures raise while preserving successful work;
resume the failed phase before reporting. Custom detectors with external state
must expose a versioned `resume_contract()`; see `evaluation/contracts.py`.

The `BenchmarkSpec` API also verifies generation and evaluation contracts. Its
configured runner checks saved generations against the requested population.
Use a fresh output directory when prompts or generation settings change; saved
generations can be rescored with a revised rubric.

## Reproducibility

`LLMJudge` results include metadata for reproducibility:

- `rubric_hash`: Which rubric was used
- `judge_model`: Which model did the judging
- `timestamp`: When it happened
- `cache_hit`: Was it cached or fresh?
- `tokens_used`: API cost tracking

These fields support auditing. Also retain the full inputs, configurations,
raw outputs, and code revision; hosted-model changes and sampling can prevent
exact reproduction. A fixed judge and versioned rubric help comparisons but
do not establish validity on their own.

### Current scoring limitations

`LLMJudge` and `LLMJudgeDetector` raise `JudgeResponseError` for missing,
malformed, nonnumeric, boolean, or nonfinite scores. Failed judgments are not
cached or saved as successful evaluations. Both derive `hit` from the configured
score threshold, ignoring a model-supplied `hit`. The new cache namespace excludes
older fallback-zero entries; it does not repair historical evaluation files.
Re-score those files before drawing conclusions.

Strict benchmark readers reject invalid stored scores, and reliability auditing
rejects invalid observations before calculating agreement or recording them.
Zero-baseline relative uplift is undefined and returns `None`; use
`compute_absolute_change` for differences in the metric's original units.

Valid JSON is not evidence of judge accuracy. Calibrate detectors against human
labels; `compass.judges.calibration.compare_human_labels` reports coverage,
agreement, precision, recall, and a confusion matrix without treating missing
labels as negatives.

## Examples

See the `examples/` directory for more detailed usage:

- `basic_eval.py`: Evaluate a single completion
- `batch_eval.py`: Evaluate multiple completions
- `multi_model_compare.py`: Compare across judges
- `custom_rubric.py`: Define your own rubric
- `caching_demo.py`: Understand caching
- `constitutional_compliance_benchmark.py`: Shared benchmark-core example

## Testing

```bash
python -m pip install -e ".[dev,openai,anthropic,google]"
python -m pytest tests/ -v
```

Benchmark report validation can be run directly:

```bash
python scripts/validate_benchmark_report.py tests/fixtures/benchmark_evaluations_valid.jsonl
```

Replace the fixture path with a benchmark evaluation file to validate your
own output. This validator targets shared benchmark records, not arbitrary
suite checkpoints.

## Long-Running Evaluations

`CheckpointManager` saves completed rows for resume. This example scores saved
model responses using the `judge` from Quick Start; it does not regenerate them:

```python
from compass import CheckpointManager

checkpoint = CheckpointManager("results/benchmark.jsonl")
completed = checkpoint.load()  # Resume if interrupted
evaluation_set = [
    ("subject-model", "p1", 0, "I agree with everything you said."),
]

for subject_model, prompt_id, sample, response_text in evaluation_set:
    identity = (subject_model, "suite", "detector", prompt_id, "default", sample)
    if identity in completed:
        continue  # Already done
    
    result = judge.evaluate(response_text)
    checkpoint.save({
        "model": subject_model,
        "suite": "suite",
        "detector": "detector",
        "prompt_id": prompt_id,
        "condition": "default",
        "sample_idx": sample,
        "score": result.score,
        "hit": result.hit,
        "judge_model": result.judge_model,
        "rubric_hash": result.rubric_hash,
    })
    completed.add(identity)
```

**Features:**
- JSONL format with immediate persistence
- Sample-level granularity for partial retries
- Backward compatible with legacy checkpoints
- Malformed rows are logged and skipped by `CheckpointManager.load()`;
  strict benchmark report readers reject malformed artifacts

See `docs/CHECKPOINT_SYSTEM.md` for full details.

## Judge Reliability & Auditing

Measure inter-judge disagreement and screen unexpected control scores:

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

# Screen scores on controls expected to have low scores
drift = auditor.detect_drift([0.05, 0.10, 0.08], expected_low=True)
print(f"Concern level: {drift['concern_level']}")  # "none", "warning", or "critical"
```

**Features:**
- Wilson agreement intervals under an independent-observation assumption
- Control-score screening; a flagged score alone does not establish drift
- Disagreement sample identification
- Multi-judge system design patterns

Investigate flagged controls by inspecting the response and rubric. To assess
change over time, compare repeated evaluations of a fixed control set with a
saved baseline. See `docs/JUDGE_RELIABILITY.md` for details.

## Pairwise Model Comparison

Rank models on shared evaluations. This example uses illustrative scores so it
can run without API calls:

```python
from compass import PairwiseRanker

ranker = PairwiseRanker()

# One score per model and prompt; lower scores mean fewer violations.
scores = {"model-a": [0.1, 0.3, 0.2], "model-b": [0.4, 0.2, 0.7]}
for model_name, values in scores.items():
    for prompt_id, score in zip(["p1", "p2", "p3"], values):
        ranker.add_record(
            suite="task_focus",
            model=model_name,
            comparison_key=(prompt_id, "neutral"),
            score=score,
            metadata={"task_type": "coding"}
        )

# min_matches counts shared sample keys across each model pair.
rankings = ranker.rank("task_focus", min_matches=1)
for model, wins, total in rankings["overall_ranking"]:
    print(f"{model}: {wins:.1f}/{total} wins ({100*wins/total:.0f}%)")

# Segment by task type
segmented = ranker.rank_by_segment("task_focus", segment_by="task_type", min_matches=1)
print(segmented["coding"]["overall_ranking"])
```

**Features:**
- Lower score = better (fewer violations)
- Win/loss/tie counts (ties contribute half a win to each model)
- Segmented analysis by task type or custom metadata
- Pairwise detail breakdowns

See `docs/PAIRWISE_COMPARISON.md` for full details.

## Local Model Evaluation

Generate locally through Ollama and score with a cloud judge. Install the
Python client (`python -m pip install ollama`), start Ollama, and make the
`llama3.1:latest` model available before running. The cloud judge also needs
the OpenAI extra and `OPENAI_API_KEY`.

```python
from compass import OllamaClient, OpenAIClient, LLMJudge, JudgeConfig, RubricLibrary

# Generate locally (no hosted API charge)
generator = OllamaClient(model="llama3.1:latest")
completion = generator.complete(
    prompt="Fix this Python bug: ...",
    max_tokens=100,
)
print(f"Cost: ${completion.cost_usd}")  # $0.00

# Judge with a cloud model
config = JudgeConfig(
    rubric=RubricLibrary.task_focus,
    judge_model="gpt-4o-mini",
)
judge_client = OpenAIClient(model="gpt-4o-mini")
judge = LLMJudge(config, judge_client)
result = judge.evaluate(completion.completion, context="Fix this Python bug: ...")
print(f"Estimated judge spend this session: ${judge_client.total_cost_usd:.4f}")
```

**Features:**
- Local inference via Ollama; reported API cost excludes local hardware costs
- No API keys needed for generation
- Separate generator and judge models
- Works with existing judge infrastructure

See `examples/constitutional_compliance_benchmark.py` for a larger local-vs-cloud setup.

## Documentation

- `docs/api.md`: Full API reference
- `docs/CLIENT_ADAPTERS.md`: Shared client adapter contract and provider behavior
- `docs/rubric_design.md`: How to write good rubrics
- `docs/reproducibility.md`: How reproducibility works
- `docs/CHECKPOINT_SYSTEM.md`: Resumable evaluations guide
- `docs/JUDGE_RELIABILITY.md`: Judge auditing and drift detection
- `docs/PAIRWISE_COMPARISON.md`: Model ranking methodology
- `docs/CONSTITUTIONAL_COMPLIANCE.md`: Compliance benchmark guide

## Contributing

For development setup and contribution guidelines, see `CONTRIBUTING.md`.
