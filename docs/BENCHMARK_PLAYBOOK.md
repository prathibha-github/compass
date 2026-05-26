# Benchmark Playbook

This guide is for adding or extending benchmark families in `compass` without
forking the core runner.

## What Belongs Where

Use this split consistently:

- `compass/benchmark/specs.py`
  Benchmark contracts (`BenchmarkPrompt`, `BenchmarkSpec`, run presets, runner contract)
- `compass/benchmark/registry.py`
  Built-in benchmark registrations and runner lookup
- `compass/benchmark/runner.py`
  Shared generation and evaluation loops plus the default shared runner
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

### 1. Define the benchmark spec and run preset

Create prompts, rubric mappings, and at least one benchmark-owned run preset:

```python
from compass.benchmark import (
    BenchmarkPolicyDefaults,
    BenchmarkPrompt,
    BenchmarkRunPreset,
    BenchmarkSpec,
)
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
    run_presets={
        "default": BenchmarkRunPreset(
            models=("llama3.1",),
            samples=2,
            judge_model="llama3.1",
            output_dir="results/my_benchmark",
            policy=BenchmarkPolicyDefaults(
                token_budgets={"default": 150, "gpt": 400},
                analysis_lanes=("summary", "pairwise"),
            ),
        ),
    },
)
```

Rules:
- `name` and `version` must be explicit.
- `prompts_by_rubric` and `rubrics_by_name` must cover the same rubric names.
- `task_type` should be stable enough to support pairwise segmentation.
- benchmark defaults belong in `run_presets`, not as CLI-only defaults
- policy defaults should cover token budgets, fairness, quality filtering, and analysis lanes

### 2. Register the benchmark and runner contract

Add the spec to `compass/benchmark/registry.py` and expose both lookup paths:
- `get_benchmark_spec(...)`
- `get_benchmark_runner(...)`

Registration validates that the runner satisfies the `BenchmarkRunner`
contract. If the shared core runner is enough, `register_benchmark_spec(...)`
will supply `SharedBenchmarkRunner(spec)` automatically.
The shared runner also invokes `validate_report(...)` after evaluation so
benchmark quality-gate enforcement does not depend on CLI convention.

The `BenchmarkRunner` contract is registration-time enforced and must expose:
- `spec`
- `validate_run_config(run_config) -> BenchmarkRunConfig`
- `generate(run_config) -> Path`
- `evaluate(generations_path, run_config) -> Path`
- `analyze(evaluations_path, run_config) -> dict`
- `rank(evaluations_path, run_config) -> None`
- `validate_report(evaluations_path, run_config) -> Sequence[str]`

Registration invariants:
- `runner.spec.name` must match `spec.name`
- `runner.spec.version` must match `spec.version`
- report validation must participate through the runner contract, not only the CLI wrapper

### 3. Reuse the shared runner

Benchmark examples should build a run config from the benchmark preset, apply
CLI overrides, and call the registered runner:

- `spec.make_run_config(...)`
- `runner.validate_run_config(...)`
- `runner.generate(...)`
- `runner.evaluate(...)`
- `runner.analyze(...)`
- `runner.rank(...)`

The example script should mainly do:
- argument parsing
- thin override merging on top of a benchmark-owned preset
- model/judge availability checks
- call into the shared benchmark core

### 4. Persist versioned outputs

Generated rows and evaluation rows must go through schema migration helpers in
`compass/benchmark/schemas.py`.

Minimum persisted guarantees:
- `benchmark_name`
- `benchmark_version`
- `benchmark_schema_version`
- `benchmark_record_type`
- stable identity fields (`model`, `rubric`, `prompt_id`, `sample_idx`)

### 5. Make artifact semantics explicit

Benchmark outputs should be self-describing and operationally predictable.

Write and document these artifacts:
- `generations.jsonl`
  Append-only generation checkpoint. Resume and evaluation paths may salvage a
  valid prefix and warn on malformed rows; treat it as an operational
  checkpoint, not as a silently self-healing final report.
- `evaluations_<judge>.jsonl`
  Append-only evaluation checkpoint. Shared summary, ranking, and report
  validation paths read this strictly and fail on malformed rows.
- `benchmark_run_policy.json`
  Atomic sidecar artifact that records the resolved preset and effective run
  policy used to produce the outputs.

Runtime failure behavior should also be explicit:
- generation and evaluation runtime failures are logged per row
- the run emits a final warning when failures occurred
- benchmark artifacts contain successful rows only; do not treat missing rows as
  silent success

### 6. Include quality diagnostics

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

### 7. Add tests before claiming the benchmark is integrated

Minimum test set:
- spec/registry contract test
- preset/run-config override test
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
