# Compass

Compass is a Python library for evaluating subjective LLM behavior. It provides stable, versioned primitives — rubrics, judges, cache, checkpoint, comparison — that experiment runners consume.

## Two-repo architecture

```
compass/                        ← this repo — the library
compass-experiments/            ← experiment runner — consumes compass
```

**compass owns:** rubric definitions, LLM judges, evaluation cache, checkpoint manager, client abstractions, detector/suite definitions, pairwise ranking, judge reliability auditing.

**Compass Experiments owns:** the runner, scenario loaders (MMLU/BoolQ/etc.), reporting, CLI, orchestration.

The dependency direction is one-way: `compass_experiments` imports from `compass`. Compass never imports from the experiment harness.

## Benchmark definition pattern

A benchmark has two parts that live in different places:

- **Definition** (prompts, conditions, rubrics, detector config) → compass, as a versioned `TicSuite` or `Rubric`, registered in the suite/rubric library. This is the stable, reusable, hashable artifact.
- **Execution** (run models, checkpoint, report) → the Compass Experiments runner.

If a benchmark is worth repeating or sharing, define it in compass first, then run it from Compass Experiments.

The harness project is named Compass Experiments; its Python package is
`compass_experiments`. Existing local checkouts may retain their original
directory names.

## Module structure

```
compass/
  rubrics/        versioned, immutable rubric definitions + RubricLibrary
  judges/         LLMJudge, JudgeConfig, EvaluationResult, JudgeReliabilityAuditor
  cache/          EvaluationCache (deterministic, content-addressed)
  evaluation/     CheckpointManager (JSONL-based, resumable)
  clients/        AnthropicClient, OpenAIClient, OpenAIResponsesClient,
                  GoogleAIClient, OllamaClient + shared pricing table
  detectors/      TicSuite framework, builtin suites, heuristic/LLM detectors,
                  quirk detection, persona transfer
  benchmark/      BenchmarkSpec, SharedBenchmarkRunner, registry, reporting,
                  validation, CLI helpers
  comparison/     PairwiseRanker, MultiModelComparator
  reproducibility/ EvaluationMetadata, cost tracking
```

## Key conventions

- Rubrics are frozen dataclasses with a content hash. Never mutate; bump the version instead.
- The cache key combines the judge configuration hash, a hash of response/context/provider settings, and prompt/scoring versions. See `docs/reproducibility.md` for the fields.
- `hit` is always recomputed from `score >= hit_threshold`. The judge's own `hit` field is ignored.
- Client pricing uses exact model names with prefix-based fallback. Update the pricing table when adding new models.
- Shared runner factories configure temperature overrides for GPT-5 and o4 model prefixes. Direct client construction requires an explicit `required_temperature` policy; see `docs/CLIENT_ADAPTERS.md`.
- OpenAI and Anthropic clients retry transient provider failures. Retry and usage policies differ by adapter; check `docs/CLIENT_ADAPTERS.md` and `compass/clients/policy_audit.py`.

## Commit style

No AI co-authorship lines. Concise imperative messages.
