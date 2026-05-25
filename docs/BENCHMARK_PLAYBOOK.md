# Benchmark Playbook

This guide is for adding or extending benchmark families in `compass` without
forking the core runner.

## What Belongs Where

Use this split consistently:

- `compass/benchmark/specs.py`
  Benchmark contracts (`BenchmarkPrompt`, `BenchmarkSpec`)
- `compass/benchmark/registry.py`
  Built-in benchmark registrations
- `compass/benchmark/runner.py`
  Shared generation and evaluation loops
- `compass/benchmark/reporting.py`
  Shared aggregation, summary formatting, pairwise ranking
- `compass/benchmark/validation.py`
  Quality-gate and report validation logic
- `examples/`
  Thin CLI wrappers and runnable end-to-end examples

If a new benchmark requires edits inside shared runner internals, stop and check
whether the benchmark contract is missing a hook rather than adding
benchmark-specific branches.

## Benchmark Authoring Workflow

### 1. Define the benchmark spec

Create prompts and rubric mappings as a `BenchmarkSpec`:

```python
from compass.benchmark import BenchmarkPrompt, BenchmarkSpec
from compass.rubrics.library import RubricLibrary

MY_BENCHMARK = BenchmarkSpec(
    name="my_benchmark",
    version="1.0",
    prompts_by_rubric={
        "clarity": (
            BenchmarkPrompt(
                id="p1",
                text="Explain how caching works.",
                task_type="conceptual_explanation",
            ),
        ),
    },
    rubrics_by_name={
        "clarity": RubricLibrary.clarity,
    },
)
```

Rules:
- `name` and `version` must be explicit.
- `prompts_by_rubric` and `rubrics_by_name` must cover the same rubric names.
- `task_type` should be stable enough to support pairwise segmentation.

### 2. Register the benchmark

Add the spec to `compass/benchmark/registry.py` and expose a lookup path via
`get_benchmark_spec(...)`.

Registration is what makes the benchmark reusable by examples, downstream
repos, and tests without rewriting orchestration.

### 3. Reuse the shared runner

Benchmark examples should call:

- `generate_completions(...)`
- `evaluate_completions(...)`
- `analyze_results(...)`
- `print_summary(...)`
- `rank_models(...)`

The example script should mainly do:
- argument parsing
- model/judge selection
- output directory setup
- call into the shared benchmark core

### 4. Persist versioned outputs

Generated rows and evaluation rows must go through schema migration helpers in
`compass/benchmark/schemas.py`.

Minimum persisted guarantees:
- `benchmark_schema_version`
- `benchmark_record_type`
- stable identity fields (`model`, `rubric`, `prompt_id`, `sample_idx`)

### 5. Include quality diagnostics

Benchmark outputs are not valid unless they carry quality-gate metadata.

Required generation/evaluation fields:
- `generation_visible_chars`
- `generation_visible_word_count`
- `generation_hit_token_cap`
- `generation_is_fragment`
- `generation_quality_flagged`
- `generation_finish_reason`
- `generation_token_cap_inferred_legacy`

Required aggregated metrics:
- `quality_flagged_pct`
- `token_cap_pct`
- `fragment_pct`
- `legacy_cap_inferred_pct`
- `quality_filtered_total`
- `quality_filtered_hit_rate`

### 6. Add tests before claiming the benchmark is integrated

Minimum test set:
- spec/registry contract test
- generation loop test
- evaluation loop test
- quality validation test
- summary/golden formatting test
- example smoke test

## Interpreting Results

Do not treat all high violation rates as model failures.

Check, in order:
1. `quality_flagged_pct`
2. `token_cap_pct`
3. `fragment_pct`
4. `quality_filtered_hit_rate`

If `quality_flagged_pct` is high, the benchmark is first reporting a generation
quality problem. Use the quality-filtered view before drawing conclusions about
model behavior.

## CI Requirements

Every benchmark family should pass:
- example smoke tests
- schema migration tests
- benchmark report validation
- the Python test matrix

The report validator is available as:

```bash
python scripts/validate_benchmark_report.py path/to/evaluations.jsonl
```

## Checklist

- [ ] Benchmark spec added
- [ ] Registry entry added
- [ ] Example wrapper added or updated
- [ ] Schema-backed output path used
- [ ] Quality diagnostics present
- [ ] Report validator passes
- [ ] Tests added for spec, runner, and reporting
- [ ] Documentation updated
