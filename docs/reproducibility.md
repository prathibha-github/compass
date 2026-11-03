# Reproducibility in Compass

Compass records evaluation metadata and caches judge results so earlier work
can be inspected and reused. A cache hit returns a saved evaluation. A new
provider call can return a different score, even with the same visible inputs;
Compass does not guarantee identical model outputs across time or environments.

See [limitations and work left to do](LIMITATIONS.md) for the remaining gaps
in scoring, saved metadata, and resume behavior.

## Cache identity

`LLMJudge` uses these cache coordinates:

```text
(config_hash, text_hash, prompt_version + scoring_version)
```

`config_hash` covers the rubric hash, judge model, maximum tokens, temperature,
and seed. `text_hash` covers separate response, user-request, and generation-system fields.
The last coordinate identifies the judge prompt template and score-validation
contract. Changing either selects a different cache entry; older fallback-zero
entries are excluded by the new scoring version.

Rubrics are frozen dataclasses, but `JudgeConfig` is mutable. Construct a new
configuration for a new experiment and preserve the old one with its results.
`JudgeConfig.seed` affects cache identity; `LLMJudge` does not forward it to the
provider's completion call. Some clients also override or omit requested
sampling parameters for particular models.

The default cache directory is `.compass_cache/`, with one JSON file per hashed
key. Suite and benchmark runners use a `.cache/` directory under their output
path. Back up the cache if you need to preserve the original judgments.
Deleting it forces new calls, which may cost money and produce different results.

## Reuse a saved evaluation

Install the OpenAI provider extra and set `OPENAI_API_KEY` for this example.
The first evaluation makes a provider call if the entry is absent; the second
uses the same evaluation inputs and cache:

```python
from compass import EvaluationCache, JudgeConfig, LLMJudge, OpenAIClient, RubricLibrary

config = JudgeConfig(
    rubric=RubricLibrary.task_focus,
    judge_model="gpt-4o-mini",
    max_tokens=180,
    temperature=0.0,
    seed=42,
)
client = OpenAIClient(model=config.judge_model)
cache = EvaluationCache(cache_dir=".compass_cache")
judge = LLMJudge(config, client, cache)

response = "Use a dictionary to count occurrences."
request = "How do I count repeated values in Python?"
first = judge.evaluate(response, context=request)
second = judge.evaluate(response, context=request)
assert second.cache_hit
assert second.score == first.score
```

This demonstrates reuse, not deterministic regeneration. If a hosted model
changes behind an unchanged ID, the cache still contains the earlier judgment.
Use a separate cache directory when you intend to measure the changed judge.

## What to preserve

Keep the full prompts, system conditions, responses, rubric definitions, judge
configuration, sampling settings, provider model IDs, and both repository commit
SHAs with each experiment. `EvaluationResult` records the rubric hash, judge
model, prompt version, timestamp, cache status, token usage, and cost, but does
not store all inputs or the full configuration.

`EvaluationMetadata.from_result()` currently fills in `seed=42`; it cannot
recover a non-default seed from an `EvaluationResult`. Save the configuration
explicitly rather than treating that helper's default as evidence of the seed
used by the provider.

A report can summarize the results you collected:

```python
from compass import reproducibility_report

print(reproducibility_report([first]))
```

The report's generated instructions still describe seed 42 as deterministic.
Treat them as configuration hints, not a promise of exact reproduction. The
saved artifacts and actual provider settings are the evidence for a run.

## Resume is a separate mechanism

The suite runner checks input fingerprints in addition to checkpoint identities.
Generation resume verifies prompt text, task type, system condition, model,
temperature, and token budget. Evaluation resume verifies saved generation data,
detector parameters and implementation, judge model, and scoring/prompt versions.
It rejects mismatches and legacy rows without fingerprints before provider calls.
Unchanged runs can append samples. Hosted-model alias changes and external custom
client state cannot be inferred from these inputs; use pinned model IDs and fresh
output directories when such state changes.

Generation and evaluation manifests record expected rows. Incomplete generated
artifacts with manifests cannot be scored, and incomplete evaluation files cannot
be summarized. Imported legacy generations without manifests remain scoreable,
but their intended generation population cannot be verified. Retain manifests
beside their JSONL files when moving artifacts.

BenchmarkSpec also checks generation and evaluation contracts before resuming.
Its configured runner requires saved generations to match the requested models,
prompts, sample counts, and token budgets. The low-level `evaluate_completions`
function can rescore saved generations in a fresh output directory. Archived
files without manifests are marked as having unverified generation coverage;
new files with generation contracts must retain their manifests.

Use a new output directory for a changed experiment. To rescore existing suite
generations with changed detector definitions, call
`evaluate_suite_completions(..., resume=False)` with the saved generation path.
To change generation settings, generate fresh completions. See the
[checkpoint guide](CHECKPOINT_SYSTEM.md) for record formats and the
[benchmark playbook](BENCHMARK_PLAYBOOK.md) for the benchmark API.

## Context shown to judges

The suite and BenchmarkSpec runners show the saved user request by default.
They record `judge_context` with each LLM score. The direct detector accepts
`context=...` through `detect_with_judge` or `analyze_text` and uses the same
prompt template as `LLMJudge`. Custom `detect_with_judge` implementations used
with contextual `analyze_text` calls must accept `context` and `system_context`
keyword arguments; the runner does not silently drop them.

For a suite, `judge_context_mode="user_and_system"` also shows the saved system
instructions. Use that deliberately: revealing the condition may affect the
judgment. `judge_context_mode="completion"` omits both fields and is required
when scoring an archive that has no saved request. The historical constitutional
benchmark wrapper explicitly uses this completion-only mode.

Changing context invalidates resumed scores and cached judgments. Re-score the
saved generations to compare modes; no new generation calls are needed.

## Rubric versions

Create a new rubric version when changing its meaning. Keep the old definition
available so previous results remain interpretable:

```python
from dataclasses import replace
from compass import RubricLibrary

original = RubricLibrary.task_focus
revised = replace(
    original,
    version="1.1",
    text=original.text + " Explicitly requested advice is not a violation.",
)
assert revised.hash != original.hash
```

The hash distinguishes definitions; it does not validate them or stop callers
from combining incompatible results. Preserve the judge prompt-template version
and code revision as well.

## Cost accounting

`cost_summary` and `cost_per_judge` count newly incurred usage. A cache hit has
zero cost and tokens; `source_cost_usd` and `source_tokens_used` retain the
original request's estimates. Cached results are independent copies, so a later
cache lookup cannot change an earlier result.

Successful suite and benchmark score rows retain the full `judge_result`,
including raw judge output, settings, and usage. Failed calls go into a separate
`.failures.jsonl` file with sample identity and any available audit record.
These failed attempts never enter scoring or the cache. Retrying preserves the
failure file so its costs can still be inspected.

OpenAI and Anthropic clients retain usage for empty responses that trigger
retries. `tokens_used` describes the final response; `billing_tokens` and
`cost_usd` include all recorded attempts. `usage_complete=False` means that
some usage was estimated or unavailable. Summaries over successful judgments
alone exclude failed attempts; inspect the failure file when accounting for a
whole run.

`requested_seed` records the judge configuration. `effective_seed` is unknown
because the clients do not send it. Metadata from older results also leaves
this field unknown.

Provider bills remain the source for actual charges; local pricing tables and
reported token usage are estimates. Inspect invalid judge responses and
missing evaluations separately from cost and cache statistics. See the
[README's scoring limitations](../README.md#current-scoring-limitations).

## Planned coverage through reporting

Resuming a suite may add samples, but cannot remove or change previously
planned samples, including ones whose calls failed. Use a new output directory
for an intentionally smaller experiment. Keep the original plan to explain
which observations were attempted.

Modern generation and evaluation files require their manifests. Duplicate
manifest identities, missing scores, and stale scoring contracts stop reporting.
`analyze_results`, `rank_models`, and report validation check the evaluation
manifest independently. Saved successful scores remain available for retry.

Archived generations can still be scored, but suite summaries and benchmark
summaries retain unverified generation coverage. Legacy benchmark evaluations
without contracts are also identified as unverified in the text summary.

Provider responses now carry both requested and effective temperature and token
limits. Generation artifacts preserve these settings, and quality checks use
the effective token limit. A missing temperature means the client omitted the
parameter; it does not establish what sampling value the provider used. No seed
is sent by these clients.

Built-in runner policies are included in resume contracts. Judge cache keys also
include the client's reported settings. Custom client factories still need a
fresh output directory when their behavior changes; Compass cannot verify
settings that a custom client does not report.
