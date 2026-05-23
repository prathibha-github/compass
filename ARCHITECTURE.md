# Compass Architecture

Compass is the shared evaluation library. It owns rubric definitions, judge and
detector primitives, client wrappers, caching, checkpointing, and comparison
utilities.

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
    X[examples and downstream repos]

    R --> J
    C --> J
    K --> J
    J --> D
    D --> P
    E --> X
    J --> X
    D --> X
    P --> X
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

## Boundaries

- `compass` should stay reusable across projects.
- Downstream repos should import Compass primitives instead of forking them.
- Shared suites and rubrics belong here; project-specific orchestration belongs
  outside this repo.
