# A-Grade Codebase Plan

## Target

The path from the current codebase to a real A is mostly about systems rigor:

- make behavior explicit
- make important failures fail closed
- make provider quirks visible instead of implicit
- make benchmark outputs auditable without reconstructing run context

The repo is already in solid shape on modularity, testing, and extension
structure. The remaining work is mostly about tightening invariants and
operational semantics.

## Plan

### 1. Make benchmark execution fail closed

- Change benchmark report consumption to be strict by default.
- Add `strict=True/False` support to `load_generation_records(...)` and
  `load_evaluation_records(...)`.
- Use strict mode in summary, ranking, and validation paths.
- Treat generation checkpoints as strict benchmark inputs once execution moves
  past the resume/write path. Evaluation should fail on malformed generation
  rows rather than silently salvaging a partial benchmark input set.
- Fail the run with a targeted error if benchmark analysis encounters malformed
  rows.
- Keep lenient mode only for migration and debugging utilities.
- Eliminate duplicate validation checkpoints in the runner path so
  `validate_run_config(...)` happens once per execution flow and the validated
  config is what downstream stages consume.

Acceptance criteria:
- User-facing benchmark analysis cannot silently skip malformed rows unless the
  caller explicitly opts into lenient mode.
- Shared benchmark execution does not evaluate a partially corrupted
  `generations.jsonl` file as if it were a valid complete input set.
- Benchmark execution has one clear run-config validation checkpoint rather than
  repeated validation in nested helpers.

### 2. Remove hidden policy from adapters

- Start with a concrete adapter audit pass.
- Enumerate every place a client silently changes caller intent around:
  - token budgets
  - temperature
  - retries/backoff
  - default system prompts
  - unsupported feature downgrades
  - token accounting fallbacks
- Convert fairness-impacting and cost-impacting behavior from implicit adapter
  logic into explicit benchmark or run configuration where possible.
- The main existing example is GPT-5 budget multiplication in the Responses
  adapter, but the item is not complete until the audit list is closed.
- Surface all remaining deliberate adapter deviations in docs and, where
  relevant, output metadata.

Acceptance criteria:
- There is an explicit audited list of adapter-side policy translations.
- Every item on that list is either:
  - moved into explicit config/policy, or
  - documented as an intentional adapter-level exception with test coverage.
- No provider silently changes benchmark fairness or cost semantics without that
  behavior being named in code and docs.

### 3. Replace expensive availability checks with real health checks

- Stop using real completions as model availability probes.
- Add provider-specific lightweight checks where possible.
- For cloud providers, either skip eager probing or use a lightweight auth or
  metadata check if the SDK supports one.
- If no cheap check exists, make quota-consuming probing explicitly opt-in.
- Treat lightweight probing as advisory rather than authoritative.
- Preserve graceful runtime degradation when a model passes a probe but fails
  during generation or evaluation:
  - log the failure clearly
  - do not silently swallow it
  - keep failure accounting explicit in the run outcome

Acceptance criteria:
- Default benchmark startup does not spend model quota or benchmark budget just
  to test connectivity.
- Callers that use availability probing still get clear runtime behavior when a
  model later fails despite passing a lightweight probe.

### 4. Tighten benchmark contract enforcement further

Split this into structural enforcement and behavioral enforcement.

Structural enforcement:
- Keep strengthening registration-time enforcement beyond method-shape checks.
- Enforce invariants that are knowable at registration time, such as:
  - runner/spec identity match
  - required methods and signatures
  - required benchmark-owned config surface

Behavioral enforcement:
- Enforce semantic invariants in CI and shared contract tests, such as:
  - validation hook participation
  - benchmark identity round-tripping through outputs
  - run-config policy being respected by analysis and ranking
- Add startup-time self-checks for built-in benchmarks where useful.

Acceptance criteria:
- Structural contract violations are rejected at registration time.
- Behavioral contract violations are caught by shared benchmark contract tests
  that every registered benchmark must satisfy.

### 5. Make report semantics first-class

- Replace open dict-based reporting payloads in named benchmark-core entrypoints:
  - `analyze_results(...)`
  - `rank_models(...)`
  - `validate_benchmark_report(...)`
- Use explicit typed representations rather than open-ended dicts.
- Prefer `dataclass` for internal structured report objects and `TypedDict`
  only where a dict-shaped boundary is still required.
- Keep rendering as a separate step from analysis.
- Make shared reporting return structured data first, then format it for CLI or
  logs.

Acceptance criteria:
- Report metric shape is explicit and testable.
- Adding or removing a metric causes predictable breakage in tests or types.
- The named benchmark-core reporting functions no longer traffic in loosely
  shaped metric dicts internally.

### 6. Add full golden coverage for benchmark outputs

- Expand snapshot coverage beyond summary formatting.
- Add fixed-input golden tests for:
  - analyzed summary stats
  - rendered textual summary output
  - pairwise ranking output
  - quality-filter policy modes
- Version golden fixtures with schema changes where needed.

Acceptance criteria:
- Benchmark-core behavior changes produce deliberate golden diffs instead of
  silent output drift.

### 7. Strengthen cross-layer invariants

- Add more architecture-boundary tests for:
  - checkpoint schema vs benchmark schema separation
  - benchmark identity fields always appearing together
  - judge cache identity tracking all reproducibility-relevant config
  - token-accounting shape consistency across providers
- Prefer invariant tests over incidental implementation tests.

Acceptance criteria:
- Important cross-layer boundaries are guarded by explicit invariant tests.

### 8. Harden persistence semantics

- Audit all writes for artifacts that matter operationally.
- Extend atomic-write discipline where partial writes could mislead downstream
  readers.
- Make interruption behavior predictable for cache, checkpoints, and benchmark
  outputs.
- Treat single-artifact files and append-only JSONL paths differently:
  - single-file artifacts should use temp-file plus atomic replace
  - append-only JSONL checkpoints should target durable append behavior, plus
    readers that reject or ignore a final partial line deterministically
- Document the intended behavior for partial writes on each persistence path.

Acceptance criteria:
- Interruption leaves either a valid old artifact or a clearly incomplete one
  that strict readers reject.
- Every persistence path has an explicit strategy rather than a generic
  "atomic writes" goal.

### 9. Persist resolved run policy in artifacts

- Record effective benchmark policy alongside outputs or in a sidecar artifact.
- Include:
  - preset name
  - benchmark name and version
  - token budget defaults
  - per-model token overrides
  - quality filter mode
  - analysis lanes
  - legacy token-cap threshold

Acceptance criteria:
- A report can be audited later without reconstructing CLI flags or implicit
  defaults.

### 9a. Persist benchmark run outcome metadata

- Record machine-readable phase outcome metadata beside benchmark artifacts.
- Include:
  - expected row counts per phase
  - successful row counts per phase
  - runtime failure counts per phase
  - whether the run completed cleanly or with partial failures
- Keep this separate from run policy so operational outcome and configured
  intent remain distinct.

Acceptance criteria:
- A reviewer can tell from artifacts alone whether a benchmark run was complete
  or partial without reading logs.

### 10. Reduce implicit CLI behavior further

- Keep CLI wrappers thin.
- Audit example scripts for logic that still belongs in shared benchmark core.
- Move repeated benchmark-run presentation logic into shared helpers where it is
  benchmark-agnostic, including provider/source summaries and cost/policy
  presentation where feasible.
- Restrict CLI responsibilities to:
  - argument parsing
  - applying overrides
  - presenting clean errors

Acceptance criteria:
- Adding a new benchmark family requires almost no benchmark-specific control
  flow in `examples/`.
- Example wrappers do not each reinvent benchmark run banners, provider source
  classification, or other shared presentation rules.

### 11. Improve diagnostics quality

- Standardize benchmark-facing errors for:
  - invalid run config
  - malformed report rows
  - unsupported provider features
  - incomplete or partial benchmark artifacts
- Prefer clean `logger.error(...)` plus `sys.exit(1)` behavior in entrypoints
  over raw tracebacks for expected operational failures.

Acceptance criteria:
- Common user-facing failures are short, actionable, and consistent.

### 12. Finish operational docs

- Round out the documentation that prevents misuse, not just reference docs.
- Cover:
  - benchmark methodology and interpretation
  - adapter contract and cost semantics
  - strict vs lenient artifact handling
  - what conclusions users should not draw from benchmark reports

Acceptance criteria:
- New contributors and report consumers can use the system correctly without
  relying on tribal knowledge.

## Recommended Order

1. Make benchmark loading and reporting fail closed.
2. Replace expensive availability probing.
3. Remove hidden provider budget policy from adapters, starting with the audit
   list.
4. Persist resolved run policy in outputs.
5. Introduce structured report objects and stronger golden coverage.
6. Finish diagnostics and operational docs.

## What “A” Looks Like

The codebase earns an A when:

- benchmark results are trustworthy by construction
- malformed artifacts cannot quietly produce plausible outputs
- provider-specific behavior is explicit, documented, and test-enforced
- shared abstractions own policy; adapters only implement transport details
- outputs are self-describing enough to audit after the fact
