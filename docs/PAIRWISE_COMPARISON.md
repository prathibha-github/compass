# Pairwise Model Comparison & Rankings

## Overview

Compare multiple models head-to-head on a shared evaluation set. Generates win/loss/tie statistics and ranks models by pairwise preference.

**Key insight:** Lower score is better (fewer violations = more compliant).

## Concepts

### Win/Loss/Tie

For each shared evaluation (same prompt, condition, sample):

```
Model A score: 0.3
Model B score: 0.7

A wins (lower score): 0.3 < 0.7 ✓
```

Scoring rules:
- **Win:** score < opponent's score (worth 1 point)
- **Tie:** score == opponent's score (worth 0.5 points each)
- **Loss:** score > opponent's score (worth 0 points)

### Preference Score

A model's preference score is the sum of its win fractions:

```
Model A: 3 wins + 0.5 tie = 3.5 preference points
Model B: 1 win + 0.5 tie = 1.5 preference points

A ranks higher (3.5 > 1.5)
```

### Minimum Matches

Only rank pairs with sufficient shared evaluations (minimum 2 by default):

```
GPT-4o vs Claude:
  Shared evaluations: 40
  Win rate: 60%
  ✓ Rank this pair

GPT-4o vs Llama:
  Shared evaluations: 1
  Win rate: 100% (but n=1!)
  ✗ Flag as low-n, don't rank
```

## Usage

### Basic Ranking

```python
from compass import PairwiseRanker

ranker = PairwiseRanker()

# Add evaluation records
for model, completion, score in evaluations:
    ranker.add_record(
        suite="task_focus",
        model=model,
        comparison_key=(prompt_id, condition),  # Shared across models
        score=score,
    )

# Calculate rankings
rankings = ranker.rank("task_focus")

for model, wins, total in rankings["overall_ranking"]:
    win_rate = 100 * wins / total if total > 0 else 0
    print(f"{model}: {wins:.1f}/{total} wins ({win_rate:.0f}%)")
```

### Pairwise Matchups

```python
for (model_a, model_b), result in rankings["pairwise_results"].items():
    wins_a = result["wins_a"]
    wins_b = result["wins_b"]
    total = result["matches"]
    
    print(f"{model_a} vs {model_b}: {wins_a}-{wins_b} ({total} matches)")
```

## Segmented Analysis

Break down rankings by **task type, condition, or any metadata**:

```python
# Rank by task type
segmented = ranker.rank_by_segment("task_focus", segment_by="task_type")

for task_type, results in segmented.items():
    print(f"\n{task_type}:")
    for model, wins, total in results["overall_ranking"]:
        print(f"  {model}: {100*wins/total if total else 0:.0f}%")
```

This reveals **model strengths/weaknesses**:

```
CODING TASKS:
  1. Claude-Opus (75%)
  2. GPT-4o (60%)
  3. Llama (40%)

WRITING TASKS:
  1. GPT-4o (70%)
  2. Claude-Opus (65%)
  3. Llama (30%)
```

## Interpreting Results

### Win Rate Interpretation

| Win Rate | Meaning |
|----------|---------|
| **> 70%** | Clear winner; dominant performance |
| **55-70%** | Consistent advantage |
| **45-55%** | Comparable performance; slight edge |
| **30-45%** | Consistent disadvantage |
| **< 30%** | Clear loser; poor performance |

### Confidence Levels

With n matches, confidence in win rate grows:

```
n=5:   ±40% CI (very uncertain)
n=10:  ±30% CI
n=30:  ±18% CI (reliable)
n=100: ±10% CI (very reliable)
```

Always report match count with win rates.

### Tied Models

If two models show similar win rates:

```
Model A: 50.2% (51/101)
Model B: 49.8% (50/101)

Practically equivalent; differences could be noise
```

The differences are within statistical noise. Treat as tied unless confidence intervals clearly separate.

## What Pairwise Rankings Do Not Prove

Keep these limits in mind when reading rankings:

- A pairwise winner on this evaluation set is not automatically a better model
  in general. The ranking is local to the shared prompts, rubrics, and judge.
- A model with a narrow win rate edge is not meaningfully ahead if the match
  count is small or the confidence intervals overlap.
- Pairwise preference does not replace absolute quality diagnostics. A model can
  "win" more matchups while both models still fail the underlying benchmark at
  unacceptable rates.
- Rankings across separate runs are not directly comparable unless the
  benchmark policy, prompt set, and judge model are held constant.

## Avoiding Pitfalls

### 1. Different Data Sets

```python
# ✓ Good: Compare on shared evaluations
ranker.add_record("task_focus", "gpt-4o", ("p1", "default"), 0.5)
ranker.add_record("task_focus", "claude", ("p1", "default"), 0.6)

# ✗ Bad: Comparing different prompts
ranker.add_record("task_focus", "gpt-4o", ("p1", "default"), 0.5)
ranker.add_record("task_focus", "claude", ("p2", "formal"), 0.6)
# → Different prompts; not comparable
```

### 2. Sample Size Too Small

```python
# ✓ Good: Many shared evaluations
rankings = ranker.rank("task_focus", min_matches=30)

# ✗ Bad: Very few matches
rankings = ranker.rank("task_focus", min_matches=1)
# → High noise, unreliable rankings
```

### 3. Ignoring Ties

```python
# Ties are meaningful!
# If 30% of cases are identical, models are more similar than win % suggests

Model A: 40 wins, 10 ties, 50 total
Model B: 40 losses, 10 ties, 50 total

A and B are not directly comparable; tie rate is high
```

### 4. Cross-Suite Comparison

```
Task focus scores (0-1):   low score = better
Truthfulness scores (0-1): low score = better
Sycophancy scores (0-1):   low score = better

Can we compare model A's task-focus score (0.3) to its sycophancy score (0.8)?
→ Only if measured on same scale and detector
```

## Advanced: Normalization

For comparing across suites with different score ranges:

```python
# Option 1: Z-score within each suite
def normalize_within_suite(scores_by_suite):
    normalized = {}
    for suite, scores in scores_by_suite.items():
        mean = sum(scores) / len(scores)
        std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
        normalized[suite] = [(s - mean) / std for s in scores]
    return normalized

# Option 2: Rank-based (0-1)
def rank_normalize(scores):
    sorted_scores = sorted(scores)
    rank_map = {s: i / len(scores) for i, s in enumerate(sorted_scores)}
    return [rank_map[s] for s in scores]
```

**Caveat:** Normalization assumes score ranges are comparable. Better to evaluate on the same scale.

## Statistical Rigor

### Binomial Test (Is the difference significant?)

```python
from scipy.stats import binom_test

# Model A: 60 wins, 40 losses (60% win rate)
# Is this significantly better than 50% baseline?

p_value = binom_test(60, 100, 0.5)
if p_value < 0.05:
    print("Significant advantage (p < 0.05)")
else:
    print("Not statistically significant")
```

### Confidence Intervals

Use Wilson intervals for win rate bounds:

```python
from compass.judges.reliability import wilson_interval

hits = 60
total = 100

low, high = wilson_interval(hits, total)
print(f"Win rate: 60% [CI: {low:.1%}, {high:.1%}]")
```

Overlapping CIs mean the models are probably equivalent.

## Testing

```python
import unittest
from compass import PairwiseRanker

class TestPairwiseRanking(unittest.TestCase):
    def test_clear_winner(self):
        ranker = PairwiseRanker()
        
        # Model A always wins
        ranker.add_record("test", "A", ("p1", "c"), 0.2)
        ranker.add_record("test", "B", ("p1", "c"), 0.8)
        ranker.add_record("test", "A", ("p2", "c"), 0.3)
        ranker.add_record("test", "B", ("p2", "c"), 0.9)
        
        results = ranker.rank("test")
        self.assertEqual(results["overall_ranking"][0][0], "A")
        self.assertEqual(results["overall_ranking"][1][0], "B")
```

## Best Practices

1. **Match count first:** Ensure enough shared evaluations
2. **Report confidence intervals:** Not just point estimates
3. **Segment by task type:** Understand where models differ
4. **Manual review:** Spot-check surprising results
5. **Document:** Record which models/suites were compared, when, on what data
6. **Iterate:** Rerun after new models/prompts/judges are added
