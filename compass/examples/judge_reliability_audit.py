#!/usr/bin/env python3
"""Example: Audit judge reliability and detect disagreement/drift.

Demonstrates how to:
- Measure inter-judge agreement on safety-critical evaluations
- Detect judge drift using benign_control tests
- Identify low-confidence regions (high variance)
- Report on judge reliability

This is essential for:
- Safety-critical domains (self-harm detection, bias, toxicity)
- Multi-judge systems (reduce false positives/negatives)
- Model selection (ensure consistent evaluation)
"""

from compass import JudgeReliabilityAuditor


def demo_inter_judge_agreement():
    """Measure agreement between two judges on sycophancy detection."""

    auditor = JudgeReliabilityAuditor()

    # Simulate scores from two judges on 10 completions
    # (Lower is better - fewer tics)
    judge1_scores = [0.1, 0.3, 0.8, 0.2, 0.6, 0.0, 0.9, 0.4, 0.7, 0.15]
    judge2_scores = [0.2, 0.2, 0.75, 0.3, 0.5, 0.1, 0.85, 0.3, 0.8, 0.25]

    # Calculate agreement
    agreement = auditor.calculate_agreement(
        judge1_scores,
        judge2_scores,
        hit_threshold=0.5,  # score >= 0.5 is a "hit" (sycophantic)
    )

    print("Inter-Judge Agreement on Sycophancy Detection")
    print("=" * 60)
    print(f"Agreement rate:        {agreement['agreement_rate']:.1%}")
    print(f"Confidence interval:   [{agreement['agreement_ci_low']:.1%}, {agreement['agreement_ci_high']:.1%}]")
    print(f"Mean score difference: {agreement['mean_difference']:.3f}")
    print(f"Disagreement indices:  {agreement['disagreement_samples']}")
    print()

    if agreement["agreement_rate"] < 0.80:
        print("⚠️  Low agreement detected!")
        print("   Consider: retraining judges, adjusting threshold, or deeper analysis")
    else:
        print("✓ Judges show good agreement")


def demo_judge_drift_detection():
    """Detect judge drift using benign_control test.

    The benign_control prompt is a legitimate request that should NOT
    be flagged as sycophantic. If it is, the judge has drifted.
    """

    auditor = JudgeReliabilityAuditor()

    print("\nJudge Drift Detection (Benign Control Test)")
    print("=" * 60)

    # Scenario 1: Judge is well-calibrated (low scores on benign control)
    print("\nScenario 1: Well-calibrated judge")
    benign_scores_good = [0.05, 0.10, 0.08, 0.12, 0.07]  # All low (correct)
    drift_good = auditor.detect_drift(benign_scores_good, expected_low=True)

    print(f"  Benign control scores: {benign_scores_good}")
    print(f"  Mean score:            {drift_good['mean_score']:.2f}")
    print(f"  Concern level:         {drift_good['concern_level']}")
    print(f"  Has drift:             {drift_good['has_drift']}")
    if not drift_good["has_drift"]:
        print("  ✓ Judge is well-calibrated")

    # Scenario 2: Judge has warning-level drift
    print("\nScenario 2: Judge with warning-level drift")
    benign_scores_warning = [0.25, 0.30, 0.35, 0.28, 0.32]  # Moderately high
    drift_warning = auditor.detect_drift(benign_scores_warning, expected_low=True)

    print(f"  Benign control scores: {benign_scores_warning}")
    print(f"  Mean score:            {drift_warning['mean_score']:.2f}")
    print(f"  Concern level:         {drift_warning['concern_level']}")
    print(f"  Has drift:             {drift_warning['has_drift']}")
    if drift_warning["has_drift"]:
        print("  ⚠️  Judge shows drift! Review its training and calibration")

    # Scenario 3: Judge has critical drift
    print("\nScenario 3: Judge with critical drift")
    benign_scores_critical = [0.70, 0.80, 0.75, 0.85, 0.78]  # Very high (wrong!)
    drift_critical = auditor.detect_drift(benign_scores_critical, expected_low=True)

    print(f"  Benign control scores: {benign_scores_critical}")
    print(f"  Mean score:            {drift_critical['mean_score']:.2f}")
    print(f"  Concern level:         {drift_critical['concern_level']}")
    print(f"  Has drift:             {drift_critical['has_drift']}")
    if drift_critical["has_drift"]:
        print("  🔴 CRITICAL: Judge is flagging benign requests as problematic!")
        print("    This will cause high false-positive rates in production")


def demo_low_confidence_regions():
    """Identify where judges disagree (low confidence regions)."""

    auditor = JudgeReliabilityAuditor()

    print("\nLow-Confidence Regions (High Disagreement)")
    print("=" * 60)

    # Judges agree on easy cases, disagree on borderline cases
    judge1 = [0.1,  0.2,  0.5,  0.8,  0.9]  # Clear tics
    judge2 = [0.15, 0.25, 0.6,  0.75, 0.85]  # Similar pattern

    agreement = auditor.calculate_agreement(judge1, judge2, hit_threshold=0.5)

    print("\nScores and agreement:")
    for i, (s1, s2) in enumerate(zip(judge1, judge2)):
        h1 = "HIT" if s1 >= 0.5 else "miss"
        h2 = "HIT" if s2 >= 0.5 else "miss"
        agree = "✓" if h1 == h2 else "✗"
        print(f"  Sample {i}: J1={s1:.2f} ({h1})  J2={s2:.2f} ({h2})  {agree}")

    print(f"\nDisagreement indices: {agreement['disagreement_samples']}")
    print("  → These are borderline cases. They need manual review.")
    print("  → Consider creating a third judge for tiebreaking.")


def demo_multi_judge_summary():
    """Summarize results across multiple judge pairs."""

    print("\nMulti-Judge Reliability Summary")
    print("=" * 60)

    auditor = JudgeReliabilityAuditor()

    # Add observations from comparing three judge pairs
    judge_pairs = [
        ("gpt-4o", "gpt-4o-mini", [0.75, 0.82, 0.79]),  # High agreement
        ("gpt-4o", "claude-opus", [0.68, 0.72, 0.65]),   # Medium agreement
        ("gpt-4o-mini", "claude-opus", [0.60, 0.65, 0.58]),  # Lower agreement
    ]

    for j1, j2, score_list in judge_pairs:
        agreement = auditor.calculate_agreement(
            [score_list[0]] * 5,
            [s * 1.1 for s in score_list] * 1,
            hit_threshold=0.5,
        )
        print(f"\n{j1:15} vs {j2:15}")
        print(f"  Agreement: {agreement['agreement_rate']:.1%}")
        print(f"  Confidence interval: [{agreement['agreement_ci_low']:.1%}, {agreement['agreement_ci_high']:.1%}]")


if __name__ == "__main__":
    print("=" * 70)
    print("Example: Judge Reliability Audit")
    print("=" * 70)

    demo_inter_judge_agreement()
    demo_judge_drift_detection()
    demo_low_confidence_regions()
    demo_multi_judge_summary()

    print("\n" + "=" * 70)
    print("When to audit judges:")
    print("  • Safety-critical domains (self-harm, bias, toxicity)")
    print("  • Multi-judge systems (reduce false positives/negatives)")
    print("  • After retraining or prompt changes")
    print("  • When model selection depends on judge scores")
