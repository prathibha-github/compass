# Changelog

All notable changes to Compass are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/en/2.0.0.html).

## [0.1.0] - 2026-05-28

### Added

**Core Framework**
- Immutable, versioned `Rubric` system with deterministic hashing
- `RubricLibrary` with 5 pre-defined rubrics (sycophancy, truthfulness, therapy_speak, clarity, task_focus)
- `EvaluationResult` dataclass with comprehensive metadata tracking
- Deterministic `EvaluationCache` (disk + memory) for reproducibility

**Judge System**
- `Judge` base class abstraction
- `LLMJudge` implementation for evaluating with language models
- JSON parsing with defensive fallbacks for malformed responses
- Automatic caching integrated into judge evaluation
- Support for multiple LLM providers via `CompletionClient` interface

**Multi-Model Comparison**
- `ComparisonResult` for comparing judges across models
- `MultiModelComparator` for head-to-head evaluation
- Agreement scoring (variance-based) and hit agreement metrics
- Aggregated statistics across batches

**Reproducibility & Tracking**
- `EvaluationMetadata` for capturing evaluation context
- `reproducibility_report()` for human-readable summaries
- `cost_summary()` for aggregating token and USD costs
- `cost_per_judge()` for breaking down costs by model
- All results include timestamp, cache hit status, tokens used, cost

**Documentation**
- Comprehensive API reference (`docs/api.md`)
- Rubric design guide with examples and troubleshooting (`docs/rubric_design.md`)
- Reproducibility guide explaining versioning and caching (`docs/reproducibility.md`)
- 5 practical examples (basic eval, batch eval, multi-model comparison, custom rubric, caching demo)
- Contributing guide

**Testing**
- 89 comprehensive tests across all modules
- Coverage for parsing edge cases, caching behavior, multi-model comparison
- Mock client for deterministic testing

### Architecture Highlights

- Clean separation: Rubrics → Judges → Cache → Results
- Immutable rubrics ensure stability and versioning
- Deterministic caching via (rubric_hash, text_hash, judge_model)
- Type-hinted public API
- No external dependencies beyond openai, anthropic, pydantic

### Known Limitations

- Single-provider at a time (client passed explicitly)
- No built-in batch API support (but simple to implement)
- Cache stored as individual JSON files (scales to ~10k evaluations)
- No Web UI or REST API (design for future)

## Future Roadmap

### 0.2.0 (Planned)
- Custom client implementations (Anthropic, OpenAI, local models)
- Async/await support for batch evaluations
- Rubric YAML loading from files
- Pairwise preference scoring for model comparison

### 0.3.0+ (Research Phase)
- Constitutional Compliance Benchmark
- Integration with HELM/HuggingFace evaluation suites
- Cost estimation and budgeting tools
- Cache synchronization across machines
