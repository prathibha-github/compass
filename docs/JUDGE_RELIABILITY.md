# Judge Reliability & Inter-Judge Disagreement

## Overview

LLM judges are not perfect. They disagree with each other, drift over time, and may be calibrated incorrectly. This guide explains how to measure and detect these issues.

## When Judges Matter

**High-stakes domains:**
- Self-harm/suicide risk detection
- Bias and fairness evaluation
- Toxicity classification
- Regulatory compliance

**Cost-sensitive evaluations:**
- Safety validation (false negatives are costly)
- Model selection (wrong choice → wrong models deployed)

## Measuring Inter-Judge Agreement

### What It Means

Two judges evaluate the same completion. Agreement occurs when they both hit or both miss the detection threshold:

```
Judge 1 score: 0.3 (miss, < 0.5 threshold)
Judge 2 score: 0.35 (miss, < 0.5 threshold)
→ AGREE (both are misses)

Judge 1 score: 0.7 (hit, >= 0.5 threshold)
Judge 2 score: 0.3 (miss, < 0.5 threshold)
→ DISAGREE
```

### Calculating Agreement

```python
from compass import JudgeReliabilityAuditor

auditor = JudgeReliabilityAuditor()

judge1_scores = [0.1, 0.8, 0.2, 0.9]
judge2_scores = [0.15, 0.75, 0.3, 0.85]

agreement = auditor.calculate_agreement(
    judge1_scores,
    judge2_scores,
    hit_threshold=0.5,
)

print(f"Agreement rate: {agreement['agreement_rate']:.1%}")
print(f"Confidence interval: [{agreement['agreement_ci_low']:.1%}, {agreement['agreement_ci_high']:.1%}]")
print(f"Mean score difference: {agreement['mean_difference']:.3f}")
```

### Interpreting Results

Agreement measures whether judges make the same classifications. Two judges
can agree and still make the same mistake. Report agreement alongside each
judge's performance against independent human labels; no agreement threshold
establishes accuracy on its own.

### Confidence Intervals

The `agreement_ci_low` and `agreement_ci_high` bounds reflect statistical uncertainty:

```
With 10 samples and 8 agreements:
  Point estimate: 80%
  Confidence interval: [49%, 94%]
  
With 100 samples and 80 agreements:
  Point estimate: 80%
  Confidence interval: [71%, 87%]
```

Choose the annotation budget from the comparisons and precision needed.
These intervals treat responses as independent observations. Repeated responses
to the same prompt need prompt-level uncertainty estimates when the question
concerns generalization across tasks.

## Detecting Judge Drift

### What Is Drift?

A judge systematically changes its behavior over time or becomes miscalibrated. Common signals:

- **False positives:** Flagging benign requests as problematic
- **False negatives:** Missing actual problems
- **Score inflation:** Giving higher scores than trained for
- **Score deflation:** Giving lower scores than trained for

### Benign Control Test

A benign control can reveal false positives for a particular rubric. It does
not measure false negatives or establish change over time. Compare repeated
measurements of fixed controls with a recorded baseline to investigate drift:

```python
from compass import JudgeReliabilityAuditor

auditor = JudgeReliabilityAuditor()

# Benign control: a legitimate request (e.g., writing a safety training handout)
benign_control_scores = [0.05, 0.10, 0.08, 0.12, 0.07]

drift = auditor.detect_drift(benign_control_scores, expected_low=True)

print(f"Has drift: {drift['has_drift']}")
print(f"Mean score: {drift['mean_score']:.2f}")
print(f"Concern level: {drift['concern_level']}")
```

### Concern Levels

| Concern | Mean Score | Action |
|---------|-----------|--------|
| **none** | < 0.15 | No alert under this heuristic; check human labels separately |
| **warning** | 0.15 ≤ mean < 0.40 | Inspect examples and compare with the recorded baseline |
| **critical** | mean ≥ 0.40 | Investigate why expected-negative controls receive high scores |

These are the helper's alert thresholds, not validated acceptance criteria.
Choose controls and tolerances for the specific rubric before interpreting an
alert as evidence of drift.

## Disagreements near the threshold

Scores on opposite sides of a threshold produce different classifications.
That disagreement does not, by itself, measure either judge's confidence:

```
Clear positive (score 0.9):  J1=0.9, J2=0.85 → Agree
Near the threshold:        J1=0.4, J2=0.6 → Disagree
Clear negative (score 0.1):  J1=0.1, J2=0.15 → Agree
```

### Inspecting classification disagreements

```python
judge1 = [0.1, 0.4, 0.9]  # Includes a disagreement near the threshold
judge2 = [0.15, 0.6, 0.85]

agreement = auditor.calculate_agreement(judge1, judge2, hit_threshold=0.5)

# disagreement_samples = [1]  ← Index 1 is borderline
for idx in agreement['disagreement_samples']:
    print(f"Borderline case {idx}: J1={judge1[idx]}, J2={judge2[idx]}")
```

Inspect these cases with independent human labels. A third judge is another
measurement to assess, not a substitute for validating the label.

## Multi-Judge System Design

### Two judges

```
Request → Judge A → Score A
          Judge B → Score B
          
Average scores or use tiebreaker rule for disagreement
```

This requires two judge calls per response. Whether combining them improves
accuracy needs to be checked against human labels.

### A third judge for disagreements

```
Request → Judge A → Score A
          Judge B → Score B
          Judge C → Score C (only if A != B)
          
Majority vote on final decision
```

This requires a third call for disagreements. A majority vote can still be
wrong when judges share errors.

### Heterogeneous Judges

```
Request → Model family A
          Model family B
          Task-specific judge
          
Ensemble: vote or weighted average
```

Different model families may make different errors. Measure that on labeled
examples before paying for an ensemble across the full run.

## Cause Analysis

### Why Do Judges Disagree?

1. **Inherent Ambiguity:** The task itself is genuinely ambiguous (one possible explanation)
2. **Judge Miscalibration:** One judge's threshold is wrong
3. **Rubric Ambiguity:** The rubric text is unclear
4. **Training Data:** Judges trained on different data
5. **Model Differences:** Different base models have different sensitivities

### Debugging Disagreement

```python
# Collect disagreement cases
disagreements = []
for i, (s1, s2) in enumerate(zip(judge1_scores, judge2_scores)):
    if (s1 >= 0.5) != (s2 >= 0.5):
        disagreements.append(i)

# Sample a few and manually review
for idx in disagreements[:3]:
    print(f"Case {idx}: J1={judge1_scores[idx]}, J2={judge2_scores[idx]}")
    print(f"  Request: {requests[idx]}")
    print(f"  Completion: {completions[idx]}")
    print()

# Ask: Is the rubric clear? Is the case actually ambiguous?
```

## Auditing Checklist

Before relying on a judge's scores, record the target being labeled, the rubric
version, the context shown, and performance against held-out human labels.
Keep disagreements and unresolved examples for inspection. Check positive and
negative controls, including cases where the judges agree but people disagree.

For repeated runs, preserve the baseline judgments and request settings so a
change can be investigated without regenerating responses. Select the number
of judges from measured error reduction and cost, rather than assuming that
more judges are more accurate.

## When to Re-Audit

- **After retraining judges:** New model = new calibration
- **After rubric changes:** Rewording can shift meaning
- **Periodically in production:** Choose a cadence based on observed changes and risk
- **After major model upgrades:** New base models may behave differently
- **When accuracy drops:** Check if judge drift, not model regression

## Testing

```python
import unittest
from compass import JudgeReliabilityAuditor

class TestJudgeReliability(unittest.TestCase):
    def test_perfect_agreement(self):
        auditor = JudgeReliabilityAuditor()
        j1 = [0.1, 0.9] * 5  # Clear cases
        j2 = [0.15, 0.85] * 5
        
        result = auditor.calculate_agreement(j1, j2, hit_threshold=0.5)
        self.assertEqual(result['agreement_rate'], 1.0)  # Perfect
    
    def test_benign_control_no_drift(self):
        auditor = JudgeReliabilityAuditor()
        scores = [0.05, 0.10, 0.08]
        
        drift = auditor.detect_drift(scores, expected_low=True)
        self.assertFalse(drift['has_drift'])
```

## Further Reading

- **Statistical foundations:** Wilson score intervals for proportions
- **Detector calibration:** Comparison with independent human labels; answer-confidence ECE is a separate measurement
- **Agreement:** Krippendorff's alpha, Cohen's kappa
- **Interrater reliability:** Consensus coding standards in NLP
