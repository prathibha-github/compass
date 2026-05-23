# Compass

Compass is a Python library for evaluating subjective LLM behavior. It provides stable, versioned primitives — rubrics, judges, cache, checkpoint, comparison — that experiment runners consume.

## Two-repo architecture

```
compass/                        ← this repo — the library
helm_eval_lex_quirks/           ← experiment runner — consumes compass
```

**compass owns:** rubric definitions, LLM judges, evaluation cache, checkpoint manager, client abstractions, detector/suite definitions, pairwise ranking, judge reliability auditing.

**helm_eval_lex_quirks owns:** the runner, scenario loaders (MMLU/BoolQ/etc.), reporting, CLI, orchestration.

The dependency direction is one-way: helm_eval imports from compass. Compass never imports from helm_eval.

## Benchmark definition pattern

A benchmark has two parts that live in different places:

- **Definition** (prompts, conditions, rubrics, detector config) → compass, as a versioned `TicSuite` or `Rubric`, registered in the suite/rubric library. This is the stable, reusable, hashable artifact.
- **Execution** (run models, checkpoint, report) → helm_eval's runner.

If a benchmark is worth repeating or sharing, define it in compass first, then run it from helm_eval.

## Planned refactor (two sessions)

The following work is planned but not yet done:

**Session 1 — upgrade compass:**
- `compass/clients/pricing.py`: shared per-model pricing table (exact model names, not prefix guesses)
- `AnthropicClient` and `OpenAIClient`: add retry/backoff on 429, request throttling, per-model pricing
- `OpenAIResponsesClient`: new client for OpenAI Responses API (GPT-5 / `gpt-5-*` models)
- `compass/detectors/` module: port `TicSuite` framework and all builtin suites from helm_eval
  - `base.py`: `DetectorResult`, `StyleCondition`, `StylePrompt`, `TicDetector`, `TicSuite`, analytics functions
  - `heuristic.py`: `RegexDetector`, `PhraseSetDetector`, `EmojiDetector`, `CharacterCountDetector`
  - `llm_detector.py`: `LLMJudgeDetector` (uses `compass.judges.parsing.parse_judge_response`)
  - `suites.py`: all builtin suites + `SUITE_REGISTRY`
  - `quirk_detection.py`: `find_quirks`, `QUIRK_PROMPTS`
  - `persona_transfer.py`: `analyze_persona_transfer`
- Add tests for all new code

**Session 2 — wire helm_eval to compass:**
- Replace helm_eval's model clients with compass clients
- Replace duplicated rubric text with `compass.RubricLibrary` references
- Replace duplicated `wilson_interval` with `compass.judges.reliability.wilson_interval`
- Replace `LLMJudgeDetector` in helm_eval with `compass.detectors.LLMJudgeDetector`
- Verify tests pass in both repos

## Module structure

```
compass/
  rubrics/        versioned, immutable rubric definitions
  judges/         LLMJudge, JudgeConfig, EvaluationResult, JudgeReliabilityAuditor
  cache/          EvaluationCache (deterministic, content-addressed)
  evaluation/     CheckpointManager (JSONL-based, resumable)
  clients/        AnthropicClient, OpenAIClient, GoogleAIClient, OllamaClient
                  OpenAIResponsesClient (planned)
  detectors/      TicSuite framework + builtin suites (planned)
  comparison/     PairwiseRanker, MultiModelComparator
  reproducibility/ EvaluationMetadata, cost tracking
```

## Key conventions

- Rubrics are frozen dataclasses with a content hash. Never mutate; bump the version instead.
- The cache key is `(rubric_hash, text_hash, judge_model, temperature, seed)`. All five must be stable for a cache hit to be valid.
- `hit` is always recomputed from `score >= hit_threshold`. The judge's own `hit` field is ignored.
- Client pricing uses exact model names with prefix-based fallback. Update the pricing table when adding new models.
- GPT-5 and o4 models require `temperature=1.0` — enforce this in the client, not the caller.
- All clients retry on 429 with exponential backoff and header-based wait times (up to 10 attempts).

## Commit style

No AI co-authorship lines. Concise imperative messages.
