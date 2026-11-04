# Limitations and work left to do

The idea behind Compass is that the next time a model does something unexpected,
I can define a test suite and compare the behavior across model versions,
prompts, and conditions. Rest suggestions are one example. Unprompted safety
guidance could be another. The same infrastructure should support both.

The remaining work is checking the measurements against human labels and
establishing which comparisons the experiments can support. Implemented resume,
cache, and audit behavior is documented in [reproducibility.md](reproducibility.md).

## Judge accuracy and context

The rest suites use phrase matching and a prompt-aware LLM judge for different
questions. `rest_suggestion` matches rest-related phrases;
`unsolicited_rest_judge` assesses whether the response recommends rest the user
didn't request. The latter is included in `rest_cai_validation`, `rest_offtask`,
and `rest_intensity`; `rest_hypothesis` remains heuristic-only.

Both approaches need validation against independent human labels. Phrase
matching can count a quoted suggestion as advice, while the separate wellbeing
heuristics can match “recovery” in a database discussion. The LLM rubric tells
the judge to ignore technical uses and consider the request, but that does not
establish its accuracy. Two judges can also share errors that agreement alone
wouldn't catch.

The [comparison helper](../compass/judges/calibration.py) and the harness's
[local labeling page](https://github.com/prathibha-github/compass-experiments/blob/main/CALIBRATION.md#label-responses-in-a-browser)
are available. I need independent labels, including ambiguous cases and
counterexamples, to check which errors the detectors make. Some prompts should
be kept out of rubric development so the final check uses examples the rubric wasn't tuned on.

Suite and BenchmarkSpec runners give judges the saved user request by default.
Direct judge calls need context supplied explicitly. Including generation system
instructions can reveal the condition and affect the judgment; a completion-only
judge lacks context needed for questions such as whether advice was requested.
Compare context modes on the same saved responses and record which mode was used.
The [context guide](reproducibility.md#context-shown-to-judges) describes those options.

## Repeated prompts and pooled results

Repeated responses to one prompt can be very similar, especially at low
temperature. Reading a judgment from cache again doesn't provide independent
evidence of judge reliability. The Wilson agreement intervals do not account for
dependence between responses to the same prompt. Suite and benchmark behavior
summaries report distinct prompt counts and use whole-prompt resampling for
individual rate intervals. Comparisons between conditions or models still need
joint resampling of shared prompts; separate rate intervals do not supply an
interval for the difference.

`summarize_suite_evaluations` returns separate results for each model and
condition. Explicitly setting `pool_models=True` pools responses, so models
with more samples carry more weight. Check task-level differences where the
sample supports them rather than assuming that an aggregate holds across tasks.

Filtering short or truncated responses is another choice that can change a
result. The summaries show rates and counts with and without filtering,
including unknown quality flags. I still need to check whether the finding
holds across those choices.

## Historical files and provider changes

Current suite and BenchmarkSpec runs check input fingerprints and planned
populations. Old files without manifests still cannot tell us whether every
planned sample was collected. Rescoring them marks generation coverage as
unverified; it doesn't recover missing context or establish what was omitted.

The built-in Gemini and Ollama clients use native system-instruction fields,
and provider policies participate in resume and judge-cache identity. A provider
can still change a model behind the same API name. Fingerprints cannot detect
that change or external state used by a custom client. Preserve the model IDs,
code revisions, and effective request settings with each experiment.

Requested seeds are recorded, but the built-in clients don't send them to the
provider. Custom clients can omit settings or usage, which remain unverified.
Missing provider usage and local pricing estimates can also leave cost totals
incomplete. The [cost accounting guide](reproducibility.md#cost-accounting)
explains how to inspect recorded retries, failed attempts, and cache usage;
provider bills remain the source for actual charges.

## Next: human labels for the rest-suggestion suite

I want to use human labels to check the measurement, then test explanations for
the behavior. That means saving responses, checking whether they contain actual
rest recommendations, and comparing prompts that differ in a specific way,
such as whether the user mentions fatigue. The
[experiment notes](https://github.com/prathibha-github/compass-experiments/blob/main/RESEARCH_LIMITATIONS.md)
describe that work in more detail.

Before using a judge on adversarial text, I need to test whether instructions
in that text can influence its score. Saved context and raw judge replies make
those failures easier to inspect, but they do not prevent them.

Once those checks are in place for a suite, I can reuse it to see whether a
new model version behaves differently or whether a prompt change affects the
result. A new behavior will need its own suite and scoring checks, but it
shouldn't need a new experiment runner.
