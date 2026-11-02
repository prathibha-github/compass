# Evaluation limits and next steps

Compass provides infrastructure for collecting and scoring model outputs. Its
tests establish behavior of that infrastructure; they do not establish the
validity of a rubric, the accuracy of a judge, or a model's safety. Conclusions
need evidence from the specific experiment and population being studied.

This document describes the implementation as of November 2, 2026. Proposed
work below has not been completed or assigned a delivery date. Experiment-level
limitations and the rest-suggestion pilot are documented in
[Compass Experiments](https://github.com/prathibha-github/helm-eval-lex-quirks/blob/main/RESEARCH_LIMITATIONS.md).

## What the current safeguards establish

The judge paths reject invalid scores and derive hits from the configured
threshold. The suite runner verifies input fingerprints, records expected
sample identities, and rejects incomplete artifacts in its scoring and summary
paths. These checks prevent specific forms of silent measurement corruption.
They do not repair historical results or establish that a valid numeric score
matches a human judgment. See the [scoring contract](../README.md#current-scoring-limitations)
and [resume behavior](reproducibility.md#resume-is-a-separate-mechanism).

## Limits that affect interpretation

| Dimension | Current boundary | Evidence or work needed |
| --- | --- | --- |
| Construct validity | Rubrics define intended behavior, but a name such as `truthfulness` or `task_focus` does not establish what the detector actually measures. Keyword matches can reflect quotation, negation, or appropriate task advice. | Annotate counterexamples and matched controls. Validate each rubric against the intended construct before interpreting its hit rate. |
| Judge reliability | Parsing checks establish a usable score. Judge agreement, confidence, and deterministic cache reuse do not establish correctness; judges may share errors. | Compare with independent human labels on held-out prompts. Report precision, recall, disagreement, abstentions, and coverage by task and condition. The [calibration helper](../compass/judges/calibration.py) supports comparison; labels and validation evidence must come from the study. |
| Prompt context | The suite runner passes the saved user prompt to `LLMJudge`. The direct detector path and the current BenchmarkSpec evaluator score completion text without that context. The suite judge does not receive the generation system condition as separate context. | Specify which context the construct requires. Compare judgments on the same saved responses with controlled context changes before treating execution paths as equivalent. |
| Statistical uncertainty | Wilson intervals summarize binary observations. Repeated responses to a shared prompt and reused cached judgments can be dependent. `summarize_suite_evaluations` pools models within each condition. | Report prompt counts and response counts separately. Preserve model/task strata, use uncertainty estimates appropriate to the sampling unit, and account for exploratory comparisons. Prompt-level resampling is proposed work, not a current guarantee. |
| Resume and reproducibility | Suite fingerprints cover recorded inputs, not changes behind provider model aliases or arbitrary external client state. Legacy generations without manifests cannot verify intended coverage. BenchmarkSpec resume still relies on smaller identities. | Extend compatibility checks to BenchmarkSpec. Preserve code revisions, effective provider settings, model IDs, and manifests. Use fresh output/cache directories when unrecorded state changes. See [reproducibility](reproducibility.md). |
| Auditability and cost | Result records do not consistently retain full raw judge replies or every provider setting. Cached results retain their original cost fields. `JudgeConfig.seed` is not forwarded to providers, and the metadata helper's default seed is not evidence of an applied seed. | Preserve raw replies and effective request metadata with explicit retention controls. Separate newly incurred spend from historical cached cost. Correct seed-reporting semantics before claiming replayability. |
| Quality filtering | Token-cap and fragment flags identify possible output-quality problems. Filtering can change the evaluated population and affect model comparisons. | Report rates and denominators before and after filtering, with exclusions by model and condition. Test whether conclusions depend on the filtering rule. |
| Adversarial and external validity | The framework accepts supplied prompts and rubrics. This does not establish resistance to instructions embedded in judged text, coverage of dangerous capabilities, or transfer to agentic and deployment settings. | Define a threat model and held-out challenge set for each proposed use. Measure evaluator manipulation and distribution shifts against a human-reviewed baseline. |

## Proposed order of work

1. **Validate one measurement target.** Freeze a rubric, create independent human
   labels with adjudication, and reserve prompts for held-out assessment.
   Publish errors and coverage alongside agreement. Completion means an
   inspectable validation result, including negative findings, rather than
   meeting a threshold selected after seeing the data.
2. **Close the remaining persistence and provenance gaps.** Extend fingerprints
   and coverage checks to BenchmarkSpec, make context handling explicit, and
   retain enough judge/request data to audit a result. Validate changed-input
   and partial-failure cases without paid model calls where possible.
3. **Test stability beyond the development set.** Hold generated responses fixed
   when comparing judges or scoring contexts; generate new responses only for
   generation-side interventions. Evaluate held-out prompt families, provider
   versions, and controlled adversarial inputs. Report prompt-level uncertainty
   and whether substantive conclusions survive those changes.

## Research scope

The present contribution is behavioral evaluation infrastructure. It can
support experiments about evaluator reliability, instruction sensitivity, and
selected safeguards. A study must establish how its measured behavior relates
to the safety question it claims to address. For example, a rest suggestion is
not inherently a harmful response or an alignment failure.

Scalable oversight is a possible research direction for judge-reliability work,
but the current framework does not demonstrate reliable supervision of tasks
beyond human capability. Model-organism training, mechanistic interpretability,
frontier capability red-teaming, AI welfare measurement, and training/RL systems
are outside the demonstrated scope. Supporting those areas would require their
own methods, datasets, access, and validation.
