# Blog Post: Compass - Evaluating Subjective Model Behavior

This is an unpublished outline. Use the linked experiment notes for the evidence behind any results included in a finished post.

## Outline

### Hook (Paragraph 1)

Why do Claude models suggest rest and breaks? Why do some models hedge more than others? How do you know which judge model to trust? These questions need clear definitions and examples checked by people.

Compass lets me turn a behavioral question into a repeatable experiment, with saved responses and scoring rules I can inspect. Prompt experiments show how behavior changes with visible inputs; they do not identify a training mechanism.

### Problem Statement (Paragraph 2)

Evaluating subjective model behavior is hard because:
- You need consistent rubrics that don't change month to month
- You need to compare judges fairly (is this judge reliable?)
- You need to preserve original responses and judgments so they can be compared later
- You need to understand costs (API calls add up fast)
- You need to version everything (future you won't understand what you measured)

Compass provides the reusable library; Compass Experiments runs the studies and reports their results.

### Solution Overview (Paragraph 3)

Compass is a standalone evaluation framework for subjective model behavior. Core principles:

1. **Versioned rubrics**: preserve the scoring definition used in each run. Comparable definitions are necessary but do not guarantee comparable judge behavior.

2. **Caching**: reuse a saved judgment when the evaluation inputs match. A fresh model call can produce a different judgment.

3. **Multi-model comparison**: Compare judges across models. See where they agree and where they disagree.

4. **Audit records**: keep rubric hashes, model IDs, requested and effective settings, full inputs, and code revisions. The requested seed is not sent by built-in clients.

### Technical Deep Dive (Paragraph 4-5)

Walk through architecture:
- Rubrics are immutable dataclasses with deterministic hashing
- Cache coordinates combine configuration, text/context/provider settings, and prompt/scoring versions; see [reproducibility.md](docs/reproducibility.md)
- Judges can be any LLM (Claude, GPT, etc.)
- Results include metadata for auditing

Example:

```python
from compass import AnthropicClient, RubricLibrary, JudgeConfig, LLMJudge

# Requires the Anthropic extra and a provider key; cache misses make paid calls.
model = "claude-haiku-4-5-20251001"
config = JudgeConfig(rubric=RubricLibrary.sycophancy, judge_model=model)
judge = LLMJudge(config, AnthropicClient(model=model))
result = judge.evaluate("response text")
```

### Use Cases (Paragraph 6)

1. **Model comparison**: Compare models on a defined prompt set with a fixed rubric and a validated judge.

2. **Training evaluation**: A researcher with controlled before/after training checkpoints could use the same measurement infrastructure; the rest-suggestion pilot does not provide that evidence.

3. **Judge selection**: Which model is best at detecting therapy-speak? (Depends on your rubric, but we can test it.)

4. **Benchmarking**: Track trends over time as models improve.

### Results / Findings (Paragraph 7)

Use the [historical pilot findings](https://github.com/prathibha-github/helm-eval-lex-quirks/blob/main/CAI_FINDINGS.md): rest-related phrase matches persisted in some saved outputs under the tested suppression prompt. The pilot used heuristic detectors; current suites also include a prompt-aware LLM judge. Human-label validation is still needed before interpreting either as a reliable measure of unsolicited advice.

The matched fatigue prompts and task-only condition now in the suites are definitions for follow-up experiments. They do not establish a cause for the historical pattern.

### Why This Matters (Paragraph 8)

The useful outcome is being able to investigate an unexpected behavior without rebuilding the evaluation infrastructure. The next experiment can compare prompt wording, conditions, and model versions while keeping the scoring question explicit.

### Call to Action (Paragraph 9)

Compass is open source. Try it:
- Evaluate your own prompts and models
- Design rubrics for things you care about
- Compare judges and see what you learn
- Contribute rubrics, examples, or improvements

[Repository](https://github.com/prathibha-github/compass) and [documentation](README.md).

---

## Evidence to check before publishing

Use test counts from the current CI run and cite any reported experimental rates to their saved artifacts. Do not claim a coverage percentage, measured cost reduction, or human agreement result without the corresponding evidence.

## Tone

- Conversational, not academic
- Explain "why this matters" not just "what it does"
- Use real examples from your investigation
- Show results, not just methodology
- Be honest about limitations

## Social Media Snippets

**LinkedIn:**
"Just released Compass, an evaluation framework for subjective model behavior. Why do models suggest breaks? When do they hedge? Which judge is most reliable? The framework saves responses and judgments so I can compare models, prompts, and scoring rules. Human validation remains part of each study."

**Twitter:**
"Compass: Define a behavioral test suite, save the responses, and compare models and judges. https://github.com/prathibha-github/compass"

**HN Submission Title:**
"Compass: Open-source framework for evaluating subjective model behavior"
