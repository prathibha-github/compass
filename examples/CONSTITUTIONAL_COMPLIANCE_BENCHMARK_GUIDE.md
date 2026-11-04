# Constitutional Compliance Benchmark for Local Models

Run the registered `constitutional_compliance` benchmark on five behavior
rubrics. The default preset uses Ollama models `llama3.1`, `mistral`, and `phi`,
with `llama3.1` as judge and three samples per prompt. Provider models are also
supported; select them explicitly and provide the corresponding credentials.

## Quick Start

### Prerequisites

1. **Install Ollama** (https://ollama.ai)
   ```bash
   # macOS
   brew install ollama
   
   # Linux/Windows: download from ollama.ai
   ```

2. **Pull the models** (storage requirements depend on the selected tags)
   ```bash
   ollama pull llama3.1
   ollama pull mistral
   ollama pull phi
   ```

3. **Start Ollama server** (in a separate terminal)
   ```bash
   ollama serve
   # Listens on localhost:11434
   ```

4. **Set up API keys only when choosing a provider judge or generation model**
   ```bash
   export OPENAI_API_KEY="sk-..."  # For gpt-4o-mini
   export ANTHROPIC_API_KEY="sk-ant-..."  # For an Anthropic model
   ```

### Run Benchmark

```bash
# Quick test
python examples/constitutional_compliance_benchmark.py --samples 3

# More thorough
python examples/constitutional_compliance_benchmark.py --samples 5

# Custom subset
python examples/constitutional_compliance_benchmark.py --models llama3.1 mistral --samples 5

# Custom output directory
python examples/constitutional_compliance_benchmark.py --output-dir results/my_benchmark --samples 5
```

## What Gets Evaluated

### Constitutional Principles (5 Rubrics)

1. **Task Focus**: Does the model stay on topic vs. volunteer unasked-for advice?
   - Tests: debugging, deadline-related writing, a factual query, API troubleshooting, refactoring

2. **Truthfulness**: Does the model admit uncertainty vs. confidently assert false claims?
   - Tests: technical details, a factual baseline, undisclosed facts, evolving information, predictions

3. **Sycophancy**: Does the model give honest feedback vs. just validate the user?
   - Tests: validation seeking, honest feedback, disagreement, critique, safety concerns

4. **Therapy Speak**: Does the model use technical precision vs. unnecessary emotional language?
   - Tests: a neutral technical query and technical tasks with explicit emotional context

5. **Clarity**: Does the model explain clearly vs. obfuscate?
   - Tests: conceptual explanation, summarization, step-by-step guidance, technology explanations, comparisons

### Models

The default model tags are `llama3.1`, `mistral`, and `phi`. Record the resolved
Ollama model versions and quantizations for a run; tag names alone do not freeze
the installed weights.

**Total Evaluations**: 3 models × 5 rubrics × 5 prompts × N samples
- 3 samples = 225 evaluations
- 5 samples = 375 evaluations
- 10 samples = 750 evaluations

These are planned counts before failures or exclusions. Repeated responses to
one prompt do not replace coverage of independent tasks.

### Evaluation Flow

```
PHASE 1: Generate Completions
├─ For each model, rubric, prompt, sample:
│  ├─ Call Ollama model with prompt
│  ├─ Get completion (model-specific token budget)
│  └─ Save to generations.jsonl (resumable)
│
PHASE 2: Evaluate Completions
├─ For each completion:
│  ├─ Call judge model (llama3.1 by default)
│  ├─ Judge evaluates on rubric scale (0.0-1.0)
│  ├─ Get score + hit status
│  └─ Save to evaluations_<judge>.jsonl (resumable)
│
PHASE 3: Analyze Results
├─ Aggregate hit rates and quality-gated metrics
├─ Print summary table
└─ Return statistics and print coverage/quality diagnostics
│
PHASE 4: Pairwise Ranking
├─ Compare models head-to-head
├─ Calculate win/loss rates
├─ Segment by task type
└─ Print rankings
```

## Output

Results are saved to `results/constitutional_compliance_benchmark/` by default:

```
results/constitutional_compliance_benchmark/
├── generations.jsonl
├── generations.manifest.json
├── evaluations_<judge>.jsonl
├── evaluations_<judge>.manifest.json
├── benchmark_run_policy.json
├── benchmark_run_outcome.json
└── .cache/                # Judge cache
```

Each JSONL file is line-delimited JSON, one record per line.

### Evaluation fields (illustrative excerpt)

```json
{
  "model": "llama3.1",
  "rubric": "truthfulness",
  "prompt_id": "technical_detail",
  "task_type": "uncertainty_handling",
  "sample_idx": 0,
  "score": 0.15,
  "hit": false,
  "confidence": 0.92,
  "rationale": "Illustrative judgment; not a measured model result"
}
```

### Summary Output

Illustrative formatting, not a measured result:

```
==============================================================================================
EVALUATION SUMMARY
==============================================================================================

Model           | Rubric          |  Hit Rate |  Q-Flag |   Cap |  Frag | LegacyCap |  QF Hit | Samples
----------------------------------------------------------------------------------------------------
llama3.1        | task_focus      |     20.0% |   10.0% | 10.0% | 10.0% |      0.0% |   11.1% |      10
```

Validate report quality before interpreting results:

```bash
python scripts/validate_benchmark_report.py results/constitutional_compliance_benchmark/evaluations_llama3.1.jsonl
```

The built-in rubrics use higher scores for more evidence of a violation.
A low score reflects the judge's assessment on that rubric; it does not establish
general compliance or calibrated accuracy.

## Resumable Evaluations

If interrupted, simply re-run the same command—it will resume from the checkpoint:

```bash
# Run 1: Interrupted after 50 evaluations
python examples/constitutional_compliance_benchmark.py --samples 5
# ... stopped mid-way ...

# Run 2: Automatically resumes from checkpoint
python examples/constitutional_compliance_benchmark.py --samples 5
# Resumes: 50 prior evaluations
# Reuses completed identities and resumes the remaining planned work
```

Keep the manifests beside their JSONL files. Resume checks input contracts;
changed prompts, budgets, or judge settings can require a fresh output directory
or explicit rescoring. Missing planned generations block evaluation, and missing
evaluations block reporting. Successful rows remain available for retry. Legacy
artifacts without manifests cannot verify the intended population.

## Cost and timing

The default local generation and judging setup has no provider API charge;
local compute still consumes time and resources. Provider costs depend on the
chosen model, effective token budget, actual usage, retries, and cache reuse.
Use a small pilot and its recorded usage to estimate the larger run. Do not
infer a fixed dollar cost from the evaluation count alone.

## Judge Models

The default preset uses `llama3.1`. Choose a different judge explicitly and
assess it against human labels; provider, price, or model size alone does not
establish judgment quality:

```bash
# Use Claude instead
python examples/constitutional_compliance_benchmark.py \
  --samples 5 \
  --judge-model claude-haiku-4-5

# Use an OpenAI judge
python examples/constitutional_compliance_benchmark.py \
  --samples 5 \
  --judge-model gpt-4o
```

## Interpreting Results

### Hit rate and score

For these built-in rubrics, `hit` means `score >= 0.5`, indicating a detected
violation. The model's own `hit` field does not override that threshold. A 0%
observed hit rate means no threshold crossings in the scored sample, not proof
of correct behavior on other inputs. Report coverage, score distributions,
quality exclusions, and human calibration alongside aggregate rates.

### Pairwise ranking

Lower scores win shared matchups. Rankings describe the matched prompts and
judge setup; they do not establish a general ordering of models. Report wins,
losses, ties, matched counts, and unmatched samples. See the
[pairwise guide](../docs/PAIRWISE_COMPARISON.md) for interpretation limits.

## Advanced Options

### Skip Generation (re-evaluate existing completions)

```bash
# Generate once
python examples/constitutional_compliance_benchmark.py --samples 5

# Try different judge later (reuses generations)
python examples/constitutional_compliance_benchmark.py --samples 5 --skip-generation --judge-model gpt-4o
```

### Skip Ranking (faster for large runs)

```bash
# Skip pairwise comparison analysis
python examples/constitutional_compliance_benchmark.py --samples 10 --skip-ranking
```

### Custom Subset

```bash
# Only evaluate llama3.1
python examples/constitutional_compliance_benchmark.py --models llama3.1 --samples 5

# A rubric subset requires a new BenchmarkSpec; see the authoring playbook.
```

## Troubleshooting

### "✗ llama3.1 unavailable"

Ollama server not running. Start it first:
```bash
# In separate terminal
ollama serve
```

### "No Ollama models available"

Pull models first:
```bash
ollama pull llama3.1
ollama pull mistral
ollama pull phi
```

### Judge API errors

Verify API key is set:
```bash
# For OpenAI
test -n "$OPENAI_API_KEY" && echo "OPENAI_API_KEY is set"

# For Anthropic
test -n "$ANTHROPIC_API_KEY" && echo "ANTHROPIC_API_KEY is set"
```

### Slow generation

- Check the installed model size, quantization, and Ollama hardware usage.
- Reduce samples for an operational smoke test, keeping conclusions exploratory.
- Measure latency on your hardware; this guide does not assume a fixed speedup.

## Next Steps

1. **Run benchmark**: `python examples/constitutional_compliance_benchmark.py --samples 5`
2. **Compare with provider models**: Use the same script with explicit `--models` and credentials; keep token budgets comparable.
3. **Investigate failures**: Read highest-scoring (worst) evaluations
4. **Validate judgments**: Compare apparent failures with independent human labels before treating them as training targets.

## Architecture

The CLI resolves the registered `BenchmarkSpec` and calls `SharedBenchmarkRunner`, using:

- **OllamaClient**: Local model inference (generation)
- **LLMJudge**: Constitutional evaluation (judge)
- **CheckpointManager**: Resumable JSONL checkpoints (robustness)
- **PairwiseRanker**: Head-to-head model comparison (analysis)
- **RubricLibrary**: Pre-written Constitutional rubrics (evaluation criteria)

See [checkpoints](../docs/CHECKPOINT_SYSTEM.md),
[judge reliability](../docs/JUDGE_RELIABILITY.md), and
[pairwise comparison](../docs/PAIRWISE_COMPARISON.md) for technical details.

## Inspecting the command without model calls

```bash
python examples/constitutional_compliance_benchmark.py --help
```

Normal benchmark runs call generation and judge models. Inspect
`benchmark_run_policy.json`, `benchmark_run_outcome.json`, and the coverage
manifests when interpreting a completed run. The abbreviated JSON and table
above illustrate formatting only.
