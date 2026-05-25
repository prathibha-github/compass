# Compass Architecture

Compass is the shared evaluation library. It owns rubric definitions, judge and
detector primitives, client wrappers, caching, checkpointing, comparison
utilities, and benchmark runner infrastructure.

## Module Layout

```mermaid
flowchart TD
    R[compass.rubrics<br/>versioned rubric definitions]
    J[compass.judges<br/>LLMJudge parsing reliability]
    C[compass.clients<br/>OpenAI Anthropic Google Ollama]
    K[compass.cache<br/>evaluation cache]
    D[compass.detectors<br/>heuristics suites persona transfer]
    E[compass.evaluation<br/>checkpoint manager]
    P[compass.comparison<br/>multi-model and pairwise ranking]
    B[compass.benchmark<br/>specs runner reporting validation]
    X[examples and downstream repos]

    R --> J
    C --> J
    K --> J
    J --> D
    D --> P
    C --> B
    E --> B
    J --> B
    P --> B
    E --> X
    J --> X
    D --> X
    P --> X
    B --> X
```

## Judge Flow

```mermaid
sequenceDiagram
    participant Caller
    participant Judge as LLMJudge
    participant Cache as EvaluationCache
    participant Client as CompletionClient
    participant Parser as Judge Parser

    Caller->>Judge: evaluate(text)
    Judge->>Cache: lookup(rubric hash, text hash, judge model)
    alt cache hit
        Cache-->>Judge: cached EvaluationResult
        Judge-->>Caller: result
    else cache miss
        Judge->>Client: complete(prompt, system, max_tokens)
        Client-->>Judge: raw completion + token usage
        Judge->>Parser: parse_judge_response(raw)
        Parser-->>Judge: score/confidence/rationale
        Judge->>Cache: store(result)
        Judge-->>Caller: EvaluationResult
    end
```

## Detector and Suite Flow

```mermaid
flowchart LR
    T[Input text] --> H[Heuristic detector]
    T --> L[LLMJudgeDetector]
    L --> J[LLMJudge]
    H --> S[DetectorResult]
    J --> S
    S --> U[TicSuite summary and uplift]
    U --> P[Pairwise or report consumer]
```

## Benchmark Flow

```mermaid
flowchart TD
    S[BenchmarkSpec<br/>name version prompts rubrics]
    R[benchmark.registry<br/>built-in benchmark lookup]
    G[benchmark.runner<br/>generation and evaluation loops]
    I[benchmark.io<br/>schema-backed row loading]
    V[benchmark.validation<br/>quality-gate checks]
    T[benchmark.reporting<br/>summary and rankings]
    C[CLI example or downstream harness]

    S --> R
    R --> C
    C --> G
    G --> I
    G --> V
    I --> T
    V --> T
    T --> C
```

## Benchmark Boundaries

- Benchmark families should be introduced through `BenchmarkSpec` and registry
  entries, not through new branches inside the runner.
- Examples should stay thin wrappers over `compass.benchmark`.
- Report validation is part of the benchmark contract, not an optional extra.
- Quality diagnostics are first-class output fields. They are used to separate
  generation failures from rubric violations.

## Boundaries

- `compass` should stay reusable across projects.
- Downstream repos should import Compass primitives instead of forking them.
- Shared suites, rubrics, and benchmark core primitives belong here.
- Project-specific orchestration belongs outside this repo.
- Provider-specific request quirks belong inside client adapters; the shared
  adapter contract is documented in `docs/CLIENT_ADAPTERS.md`.
