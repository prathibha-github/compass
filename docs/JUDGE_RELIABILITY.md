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

| Agreement | Interpretation |
|-----------|-----------------|
| **> 90%** | Excellent agreement; judges are well-calibrated |
| **80-90%** | Good agreement; minor disagreements on borderline cases |
| **70-80%** | Acceptable agreement; monitor for drift |
| **< 70%** | Poor agreement; investigate calibration or redefine detection |

### Confidence Intervals

The `agreement_ci_low` and `agreement_ci_high` bounds reflect statistical uncertainty:

```
With 10 samples and 8 agreements:
  Point estimate: 80%
  Confidence interval: [44%, 97%]  ← Very wide!
  
With 100 samples and 80 agreements:
  Point estimate: 80%
  Confidence interval: [71%, 87%]  ← Narrow (reliable)
```

**Lesson:** Small sample sizes produce unreliable estimates. Audit agreement on at least 30-50 samples per judge pair.

## Detecting Judge Drift

### What Is Drift?

A judge systematically changes its behavior over time or becomes miscalibrated. Common signals:

- **False positives:** Flagging benign requests as problematic
- **False negatives:** Missing actual problems
- **Score inflation:** Giving higher scores than trained for
- **Score deflation:** Giving lower scores than trained for

### Benign Control Test

The most reliable drift detector: evaluate the judge on a **legitimate request it should pass**:

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
| **none** | < 0.15 | No drift detected; judge is well-calibrated |
| **warning** | 0.15-0.40 | Monitor closely; consider retraining |
| **critical** | > 0.40 | Immediate investigation; don't use in production |

## Low-Confidence Regions

Judges naturally disagree on **borderline cases**:

```
Clear positive (score 0.9):  J1=0.9, J2=0.85 → Agree
Borderline case (score 0.5): J1=0.4, J2=0.6 → Disagree ← Problem!
Clear negative (score 0.1):  J1=0.1, J2=0.15 → Agree
```

### Identifying Low-Confidence Cases

```python
judge1 = [0.1, 0.5, 0.9]  # Easy cases (extremes)
judge2 = [0.15, 0.6, 0.85]

agreement = auditor.calculate_agreement(judge1, judge2, hit_threshold=0.5)

# disagreement_samples = [1]  ← Index 1 is borderline
for idx in agreement['disagreement_samples']:
    print(f"Borderline case {idx}: J1={judge1[idx]}, J2={judge2[idx]}")
```

**Action:** Create a third "tiebreaker" judge for low-confidence cases, or require human review.

## Multi-Judge System Design

### 2-Judge System (Recommended)

```
Request → Judge A → Score A
          Judge B → Score B
          
Average scores or use tiebreaker rule for disagreement
```

Trade-off: 2x cost, ~10% higher accuracy.

### 3-Judge System (High-Stakes)

```
Request → Judge A → Score A
          Judge B → Score B
          Judge C → Score C (only if A != B)
          
Majority vote on final decision
```

Trade-off: Up to 3x cost, 20% higher accuracy on borderline cases.

### Heterogeneous Judges

```
Request → GPT-4o (high capability, expensive)
          Claude-Opus (different training, good agreement)
          Specialized model (task-specific, accurate)
          
Ensemble: vote or weighted average
```

Trade-off: Very high cost, maximum accuracy.

## Cause Analysis

### Why Do Judges Disagree?

1. **Inherent Ambiguity:** The task itself is genuinely ambiguous (most common)
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

Before deploying a judge system:

- [ ] **Agreement rate > 75%** on test set
- [ ] **Confidence intervals don't include < 70%**
- [ ] **Benign control test passes** (mean score < 0.15 for non-drift tasks)
- [ ] **Low-confidence cases identified** and handled (tiebreaker judge or manual review)
- [ ] **Agreement documented** for each rubric
- [ ] **Drift detection plan** (periodic benign control re-tests)
- [ ] **Disagreement cases** manually reviewed and understood
- [ ] **Judge selection justified** (capability, cost, agreement)

## When to Re-Audit

- **After retraining judges:** New model = new calibration
- **After rubric changes:** Rewording can shift meaning
- **Quarterly (production):** Detect slow drift
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
- **Calibration:** Expected Calibration Error (ECE)
- **Agreement:** Krippendorff's alpha, Cohen's kappa
- **Interrater reliability:** Consensus coding standards in NLP
