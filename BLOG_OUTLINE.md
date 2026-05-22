# Blog Post: Compass - Evaluating Subjective Model Behavior

## Outline

### Hook (Paragraph 1)

Why do Claude models suggest rest and breaks? Why do some models hedge more than others? How do you know which judge model to trust? These aren't yes/no questions—they're spectrum questions that require subjective evaluation.

Most evaluation frameworks measure benchmarks (MMLU, HellaSwag). Compass measures character traits (sycophancy, clarity, honesty). It's how we understand what training produces.

### Problem Statement (Paragraph 2)

Evaluating subjective model behavior is hard because:
- You need consistent rubrics that don't change month to month
- You need to compare judges fairly (is this judge reliable?)
- You need reproducible results (same text should always score the same)
- You need to understand costs (API calls add up fast)
- You need to version everything (future you won't understand what you measured)

Existing tools aren't built for this. They're built for benchmarks. We built compass for character.

### Solution Overview (Paragraph 3)

Compass is a standalone evaluation framework for subjective model behavior. Core principles:

1. **Immutable, versioned rubrics** — Your rubric in 2024 is identical to your rubric in 2026. Results are comparable across time.

2. **Deterministic caching** — Run the same evaluation 1,000 times for a cost of 1 API call. Same input always produces same output.

3. **Multi-model comparison** — Compare judges across models. See where they agree and where they disagree.

4. **Reproducibility first** — Every result includes rubric hash, model name, seed. You can reproduce exact results months later.

### Technical Deep Dive (Paragraph 4-5)

Walk through architecture:
- Rubrics are immutable dataclasses with deterministic hashing
- Caching key is (rubric_hash, text_hash, judge_model)
- Judges can be any LLM (Claude, GPT, etc.)
- Results include metadata for auditing

Example:

```python
from compass import RubricLibrary, JudgeConfig, LLMJudge

config = JudgeConfig(
    rubric=RubricLibrary.sycophancy,
    judge_model="claude-opus-4-7",
)
judge = LLMJudge(config, client, cache)
result = judge.evaluate("response text")
```

### Use Cases (Paragraph 6)

1. **Model comparison** — Is Claude more sycophantic than GPT? (Answer: We tested it. See the results.)

2. **Training evaluation** — Did our RLHF improve helpfulness without increasing sycophancy?

3. **Judge selection** — Which model is best at detecting therapy-speak? (Depends on your rubric, but we can test it.)

4. **Benchmarking** — Track trends over time as models improve.

### Results / Findings (Paragraph 7)

Show results from your rest-suggestion investigation:
- Claude models have built-in wellbeing prioritization
- System prompts can modulate it but don't override it
- The behavior is learned, not hardcoded
- Different judge models see it differently

### Why This Matters (Paragraph 8)

This is the first step toward understanding Constitutional AI. When we ask "why does Claude suggest rest," we're really asking "what did training teach the model to value?" Compass lets us measure that empirically.

As AI systems become more complex, our evaluation tools need to match. We can't rely on benchmarks alone. We need to understand character.

### Call to Action (Paragraph 9)

Compass is open source. Try it:
- Evaluate your own prompts and models
- Design rubrics for things you care about
- Compare judges and see what you learn
- Contribute rubrics, examples, or improvements

GitHub: [link]
Docs: [link]
PyPI: [link]

---

## Key Statistics to Include

- 89 tests, 85%+ coverage
- 5 pre-defined rubrics covering constitutional traits
- 5 working examples
- Multi-model comparison built-in
- Deterministic caching for 100x cost reduction on re-runs
- Complete reproducibility: rubric hash + judge model + seed

## Tone

- Conversational, not academic
- Explain "why this matters" not just "what it does"
- Use real examples from your investigation
- Show results, not just methodology
- Be honest about limitations

## Social Media Snippets

**LinkedIn:**
"Just released Compass, an evaluation framework for subjective model behavior. Why do models suggest breaks? When do they hedge? Which judge is most reliable? These aren't benchmark questions—they require character evaluation. Open source, reproducible, ready to use."

**Twitter:**
"Compass: Measure what really matters—not benchmarks, but character. Evaluate sycophancy, clarity, honesty. Compare judges. Perfect reproducibility. https://github.com/..."

**HN Submission Title:**
"Compass: Open-source framework for evaluating subjective model behavior"
