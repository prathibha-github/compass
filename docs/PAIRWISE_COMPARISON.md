# Pairwise Model Comparison & Rankings

## Overview

Compare multiple models head-to-head on a shared evaluation set. Generates win/loss/tie statistics and ranks models by pairwise preference.

`PairwiseRanker` treats lower scores as better. This matches Compass's built-in
violation rubrics; invert a higher-is-better score before passing it to the ranker.

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

A model's preference score is its total wins plus half its ties across eligible
opponent/sample matchups:

```
Model A: 3 wins + 0.5 tie = 3.5 preference points
Model B: 1 win + 0.5 tie = 1.5 preference points

A ranks higher (3.5 > 1.5)
```

Overall ranking sorts by these accumulated points, not normalized win rate.
Unequal matched coverage can affect the order; report totals, opponents, and
unmatched counts alongside ranks.

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
  Omit this pair; the helper does not emit a separate low-n flag
```

## Usage

### Basic Ranking

```python
from compass import PairwiseRanker

ranker = PairwiseRanker()

# Add evaluation records
for model, prompt_id, condition, sample_idx, task_type, score in evaluations:
    ranker.add_record(
        suite="task_focus",
        model=model,
        comparison_key=(prompt_id, condition, sample_idx),  # Shared across models
        score=score,
        metadata={"task_type": task_type},
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

A hypothetical display of task-level descriptive rates:

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

### Descriptive win rates and uncertainty

A win rate describes the matched samples under one scoring setup. There are no
universal cutoffs for "dominant" or "equivalent" performance. Report wins,
losses, ties, matched counts, and exclusions before interpreting an advantage.

For example, 51 wins and 50 losses correspond to 50.5% and 49.5%. That observed
difference alone does not establish either a reliable advantage or practical
equivalence. Select an uncertainty method appropriate to the prompts and
sampling design, and define a meaningful effect size in advance.

## What Pairwise Rankings Do Not Prove

Keep these limits in mind when reading rankings:

- A pairwise winner on this evaluation set is not automatically a better model
  in general. The ranking is local to the shared prompts, rubrics, and judge.
- A narrow observed edge needs uncertainty analysis. Overlapping confidence
  intervals alone do not establish equality or rule out a difference.
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

```text
Model A: 40 wins, 10 ties, 50 matches
Model B: 40 losses, 10 ties, 50 matches

A earns 45 preference points; B earns 5.
```

These models are comparable on the shared samples. Report ties explicitly: a
tie means equal recorded scores, which can also arise from coarse scoring.

### 4. Cross-Suite Comparison

```
Task focus scores (0-1):   low score = better
Truthfulness scores (0-1): low score = better
Sycophancy scores (0-1):   low score = better

Can we compare model A's task-focus score (0.3) to its sycophancy score (0.8)?
→ These measure different behaviors; their numeric ranges do not make them equivalent
```

## Comparing across rubrics

Rescaling scores does not make different rubric targets interchangeable.
Positive affine scaling preserves within-rubric ordering, so it does not change
these pairwise winners. If combining dimensions, define and justify the score
direction, weights, and missing-data policy before examining rankings. Report
the component results so trade-offs remain visible.

## Statistical Rigor

### Binomial Test (Is the difference significant?)

```python
from scipy.stats import binomtest

# Model A: 60 wins, 40 losses (60% win rate)
# Is this significantly better than 50% baseline?

p_value = binomtest(60, 100, 0.5).pvalue
if p_value < 0.05:
    print("Significant advantage (p < 0.05)")
else:
    print("Not statistically significant")
```

The example assumes independent binary win/loss observations and no ties.
Repeated samples sharing prompts need an analysis that accounts for that
structure. Do not pass half-tie preference points as binomial successes.

### Confidence Intervals

For independent binary wins with no ties, Wilson intervals give win-rate bounds:

```python
from compass.judges.reliability import wilson_interval

hits = 60
total = 100

low, high = wilson_interval(hits, total)
print(f"Win rate: 60% [CI: {low:.1%}, {high:.1%}]")
```

Interval overlap does not demonstrate equivalence. Equivalence requires a
predefined tolerance and an analysis suited to the sampling design.

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

## Matching repeated samples

Each model may contribute one score per comparison key. Include the sample
index when prompts have repeated responses: `(prompt_id, condition, sample_idx)`.
Duplicate model/key records are rejected. Ranking matches shared keys, reports
unmatched counts, and applies `min_matches` across all shared keys for each
model pair. File order does not define a match.

Matching sample indices is a descriptive convention. Independent model draws
with the same index are not experimentally paired replicates. For inference
across tasks, compare prompt-level results and account for shared prompts when
estimating uncertainty. Unequal coverage and quality exclusions can change the
set of matched samples; inspect unmatched counts before interpreting a rank.
