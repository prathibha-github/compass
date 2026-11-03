# Limitations and work left to do

The idea behind Compass is that the next time a model does something unexpected,
I can define a test suite and compare the behavior across model versions,
prompts, and conditions. Rest suggestions are one example. Unprompted safety
guidance could be another. The same infrastructure should support both.

I still need to check the scores against human labels and close a few gaps
in how runs are saved and resumed.

## Judge accuracy and context

A rubric gives the judge instructions, but I haven't checked the rest-suggestion
scores against independent human labels yet. A detector can match “recovery”
in a discussion of a database, or count a quoted suggestion as advice from the
model. LLM judges can make errors too, including errors they share with another
judge. Agreement alone wouldn't catch those.

The [comparison helper](../compass/judges/calibration.py) is in place. I need
people to label examples, including ambiguous cases and counterexamples, and
then check which errors the detectors make. Some prompts should be kept out of
rubric development so the final check uses examples the rubric wasn't tuned on.

The judges also get different inputs depending on the execution path. The
suite runner passes the user request to `LLMJudge`. The direct detector and
BenchmarkSpec paths currently score only the completion. The suite judge does
not receive the generation system prompt as separate context. A judgment about
whether advice was requested needs that distinction to be explicit. Comparing
these paths without accounting for their inputs can look like judge disagreement
when one judge simply has less information.

## Repeated prompts and pooled results

Repeated responses to one prompt can be very similar, especially at low
temperature. Reading a judgment from cache again doesn't provide independent
evidence of judge reliability. The Wilson intervals don't account for
dependence between responses to the same prompt.
I need to report the number of distinct prompts as well as the number of
responses, and use prompt-level uncertainty estimates for those comparisons.

`summarize_suite_evaluations` returns separate results for each model and
condition. Its return shape is now `summary[model][condition]`. Callers that
need the old condition-only shape must explicitly pass `pool_models=True`;
that pools responses, so models with more samples carry more weight.
Task-level breakdowns would help where the sample supports them.

Filtering short or truncated responses is another choice that can change a
result. Reports should show how many responses were excluded from each group
and whether the finding holds with and without the filter.

## Resume and saved metadata

- Suite runs check input fingerprints and expected sample counts. BenchmarkSpec
  resume still uses smaller identifiers, so changing a prompt or rubric can
  leave an old result in place. Until that is fixed, changed benchmark
  configurations need a fresh output directory.
- Old generation files without manifests can be scored, but they don't tell us
  whether every planned sample was collected. The new checks don't repair old
  results; those need a separate review or rerun.
- A provider can change a model behind the same API name. Fingerprints cannot
  detect that, or changes to external state used by a custom client. Keep model
  IDs, code revisions, and actual request settings with the run, and use a new
  output/cache directory when that state changes.

The exact checks and cache behavior are described in
[reproducibility.md](reproducibility.md).

Saved results also leave out some details I would want when investigating a
surprising score. Full judge replies and provider settings are not consistently
retained. `JudgeConfig.seed` affects the cache key but isn't sent to the provider,
and the metadata helper fills in a default seed. That needs correcting before
using the metadata as a record of how a request was made.

Cost reporting needs care too. Cached results carry the cost of the original
call. Adding those fields again overstates what was spent on the current run.

## Next: human labels for the rest-suggestion suite

I want to use human labels to check the measurement, then test explanations for
the behavior. That means saving responses, checking whether they contain actual
rest recommendations, and comparing prompts that differ in a specific way,
such as whether the user mentions fatigue. The
[experiment notes](https://github.com/prathibha-github/helm-eval-lex-quirks/blob/main/RESEARCH_LIMITATIONS.md)
describe that work in more detail.

For Compass itself, the next changes are to extend the resume checks to
BenchmarkSpec, make judge context explicit, and save enough request and judge
data to inspect a result later. Before using a judge on adversarial text, I
would also need to test whether instructions in that text can influence its
score.

Once those checks are in place for a suite, I can reuse it to see whether a
new model version behaves differently or whether a prompt change affects the
result. A new behavior will need its own suite and scoring checks, but it
shouldn't need a new experiment runner.
