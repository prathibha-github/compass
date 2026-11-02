# Reproducibility in Compass

Compass records evaluation metadata and caches judge results so earlier work
can be inspected and reused. A cache hit returns a saved evaluation. A new
provider call can return a different score, even with the same visible inputs;
Compass does not guarantee identical model outputs across time or environments.

## Cache identity

`LLMJudge` uses these cache coordinates:

```text
(config_hash, text_hash, prompt_version)
```

`config_hash` covers the rubric hash, judge model, maximum tokens, temperature,
and seed. `text_hash` covers the response plus optional user context.
`prompt_version` identifies the judge prompt template. Changing any coordinate
selects a different cache entry.

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

Checkpoint identities are smaller than judge-cache identities. Reusing an
output directory after changing prompt text, system conditions, detector
rules, rubrics, or sampling settings can reuse older results. A judge-cache
change does not help when the checkpoint skips the judge call entirely.

Use a new output directory for a changed experiment. To rescore existing suite
generations with changed detector definitions, call
`evaluate_suite_completions(..., resume=False)` with the saved generation path.
To change generation settings, generate fresh completions. See the
[checkpoint guide](CHECKPOINT_SYSTEM.md) for record formats and the
[benchmark playbook](BENCHMARK_PLAYBOOK.md) for the benchmark API.

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

`cost_summary` and `cost_per_judge` add the cost fields in the supplied results.
Cached results retain the original call's token usage and cost, so summing them
again does not measure new API spend. Client counters track calls made by that
client instance. For the example above:

```python
from compass import cost_summary

print(cost_summary([first]))  # Cost recorded on this evaluation
print(client.total_cost_usd)  # Estimated API spend by this client instance
```

Provider bills remain the source for actual charges; local pricing tables and
reported token usage are estimates. Inspect invalid judge responses and
missing evaluations separately from cost and cache statistics. See the
[README's scoring limitations](../README.md#current-scoring-limitations).
