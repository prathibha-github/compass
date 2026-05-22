"""Tests for judge reliability auditing."""

import unittest

from compass.judges.reliability import JudgeReliabilityAuditor, wilson_interval


class TestWilsonInterval(unittest.TestCase):
    """Test Wilson score interval calculation."""

    def test_perfect_agreement(self):
        """Wilson interval for 10/10 successes."""
        low, high = wilson_interval(hits=10, total=10)

        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)
        self.assertGreater(high, 0.9)  # Should be high

    def test_perfect_failure(self):
        """Wilson interval for 0/10 successes."""
        low, high = wilson_interval(hits=0, total=10)

        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)
        self.assertLess(low, 0.1)  # Should be low

    def test_even_split(self):
        """Wilson interval for 5/10 (50%)."""
        low, high = wilson_interval(hits=5, total=10)

        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)
        self.assertLess(low, 0.5)  # Low bound below 50%
        self.assertGreater(high, 0.5)  # High bound above 50%

    def test_large_sample_narrow_ci(self):
        """Wilson interval narrower with larger sample."""
        low_small, high_small = wilson_interval(hits=8, total=10)
        low_large, high_large = wilson_interval(hits=80, total=100)

        ci_width_small = high_small - low_small
        ci_width_large = high_large - low_large

        # Larger sample should have narrower CI
        self.assertLess(ci_width_large, ci_width_small)

    def test_confidence_level_effects_ci_width(self):
        """Higher confidence level widens CI."""
        low_90, high_90 = wilson_interval(hits=8, total=10, confidence=0.90)
        low_95, high_95 = wilson_interval(hits=8, total=10, confidence=0.95)

        ci_width_90 = high_90 - low_90
        ci_width_95 = high_95 - low_95

        # 95% CI should be wider than 90% CI
        self.assertGreater(ci_width_95, ci_width_90)


class TestInterJudgeAgreement(unittest.TestCase):
    """Test inter-judge agreement measurement."""

    def test_perfect_agreement_same_scores(self):
        """Judges with identical scores have 100% agreement."""
        auditor = JudgeReliabilityAuditor()

        judge1_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        judge2_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        result = auditor.calculate_agreement(judge1_scores, judge2_scores, hit_threshold=0.5)

        self.assertEqual(result['agreement_rate'], 1.0)
        self.assertEqual(result['mean_difference'], 0.0)
        self.assertEqual(len(result['disagreement_samples']), 0)

    def test_perfect_agreement_different_scores(self):
        """Judges with different scores but same threshold classification agree."""
        auditor = JudgeReliabilityAuditor()

        judge1_scores = [0.1, 0.8, 0.2, 0.9]  # Clear cases
        judge2_scores = [0.15, 0.75, 0.25, 0.85]  # Slightly different but same classification

        result = auditor.calculate_agreement(judge1_scores, judge2_scores, hit_threshold=0.5)

        self.assertEqual(result['agreement_rate'], 1.0)

    def test_complete_disagreement(self):
        """Judges always on opposite sides of threshold."""
        auditor = JudgeReliabilityAuditor()

        judge1_scores = [0.3, 0.7, 0.2, 0.8]
        judge2_scores = [0.7, 0.3, 0.8, 0.2]

        result = auditor.calculate_agreement(judge1_scores, judge2_scores, hit_threshold=0.5)

        self.assertEqual(result['agreement_rate'], 0.0)

    def test_partial_agreement(self):
        """Judges agree on some but not all cases."""
        auditor = JudgeReliabilityAuditor()

        judge1_scores = [0.1, 0.9, 0.1, 0.9]
        judge2_scores = [0.1, 0.9, 0.7, 0.3]  # Disagreement on samples 2 and 3

        result = auditor.calculate_agreement(judge1_scores, judge2_scores, hit_threshold=0.5)

        self.assertEqual(result['agreement_rate'], 0.5)  # 2 agreements out of 4
        self.assertEqual(len(result['disagreement_samples']), 2)

    def test_mean_difference_calculation(self):
        """Mean difference reflects average absolute score difference."""
        auditor = JudgeReliabilityAuditor()

        judge1_scores = [0.1, 0.5, 0.9]
        judge2_scores = [0.2, 0.4, 0.8]

        result = auditor.calculate_agreement(judge1_scores, judge2_scores, hit_threshold=0.5)

        expected_mean_diff = (abs(0.1 - 0.2) + abs(0.5 - 0.4) + abs(0.9 - 0.8)) / 3
        self.assertAlmostEqual(result['mean_difference'], expected_mean_diff, places=5)

    def test_disagreement_samples_identifies_borderline_cases(self):
        """Disagreement samples correctly identified."""
        auditor = JudgeReliabilityAuditor()

        judge1_scores = [0.1, 0.4, 0.9, 0.6]
        judge2_scores = [0.1, 0.7, 0.9, 0.3]  # Disagreement on samples 1 and 3

        result = auditor.calculate_agreement(judge1_scores, judge2_scores, hit_threshold=0.5)

        # Samples 1 and 3 should be flagged as disagreement
        self.assertIn(1, result['disagreement_samples'])
        self.assertIn(3, result['disagreement_samples'])
        self.assertEqual(len(result['disagreement_samples']), 2)

    def test_ci_contains_point_estimate(self):
        """Confidence interval contains agreement rate."""
        auditor = JudgeReliabilityAuditor()

        judge1_scores = [0.1] * 25 + [0.9] * 25
        judge2_scores = [0.15] * 20 + [0.7] * 5 + [0.85] * 25

        result = auditor.calculate_agreement(judge1_scores, judge2_scores, hit_threshold=0.5)

        agreement_rate = result['agreement_rate']
        ci_low = result['agreement_ci_low']
        ci_high = result['agreement_ci_high']

        self.assertGreaterEqual(agreement_rate, ci_low)
        self.assertLessEqual(agreement_rate, ci_high)


class TestJudgeDriftDetection(unittest.TestCase):
    """Test detection of judge drift/miscalibration."""

    def test_no_drift_low_scores(self):
        """Well-calibrated judge on benign cases scores low."""
        auditor = JudgeReliabilityAuditor()

        benign_scores = [0.05, 0.08, 0.06, 0.07, 0.09]

        drift = auditor.detect_drift(benign_scores, expected_low=True)

        self.assertFalse(drift['has_drift'])
        self.assertEqual(drift['concern_level'], 'none')

    def test_warning_level_drift(self):
        """Moderate miscalibration: concern level is warning."""
        auditor = JudgeReliabilityAuditor()

        benign_scores = [0.25, 0.30, 0.28, 0.26, 0.29]  # Elevated but not critical

        drift = auditor.detect_drift(benign_scores, expected_low=True)

        self.assertEqual(drift['concern_level'], 'warning')

    def test_critical_level_drift(self):
        """Severe miscalibration: concern level is critical."""
        auditor = JudgeReliabilityAuditor()

        benign_scores = [0.60, 0.65, 0.58, 0.62, 0.64]  # Very high for benign cases

        drift = auditor.detect_drift(benign_scores, expected_low=True)

        self.assertEqual(drift['concern_level'], 'critical')
        self.assertTrue(drift['has_drift'])

    def test_mean_score_reported(self):
        """Drift report includes mean score."""
        auditor = JudgeReliabilityAuditor()

        scores = [0.1, 0.2, 0.3, 0.2, 0.1]
        drift = auditor.detect_drift(scores, expected_low=True)

        expected_mean = sum(scores) / len(scores)
        self.assertAlmostEqual(drift['mean_score'], expected_mean, places=5)


class TestAddComparison(unittest.TestCase):
    """Test recording of audit observations."""

    def test_add_observation(self):
        """add_comparison records observations."""
        auditor = JudgeReliabilityAuditor()

        auditor.add_comparison(
            judge1_score=0.3,
            judge2_score=0.35,
            metadata={"judge_pair": ("gpt-4o", "claude"), "suite": "task_focus"},
        )

        summary = auditor.summary()
        self.assertGreater(summary['agreement_rate'], 0.0)

    def test_multiple_comparisons_tracked(self):
        """Multiple audit runs tracked separately."""
        auditor = JudgeReliabilityAuditor()

        auditor.add_comparison(
            judge1_score=0.3,
            judge2_score=0.35,
            metadata={"judge_pair": ("gpt-4o", "claude"), "suite": "task_focus"},
        )
        auditor.add_comparison(
            judge1_score=0.7,
            judge2_score=0.75,
            metadata={"judge_pair": ("gpt-4o", "llama"), "suite": "truthfulness"},
        )

        # Both observations recorded
        self.assertEqual(len(auditor.observations), 2)

    def test_summary_aggregates_data(self):
        """summary() returns aggregated audit data."""
        auditor = JudgeReliabilityAuditor()

        auditor.add_comparison(
            judge1_score=0.3,
            judge2_score=0.35,
            metadata={"judge_pair": ("gpt-4o", "claude"), "suite": "task_focus"},
        )

        summary = auditor.summary()

        self.assertIn('agreement_rate', summary)
        self.assertIn('mean_difference', summary)


class TestDriftWithExpectedHigh(unittest.TestCase):
    """Test drift detection with expected_high=True."""

    def test_high_drift_detection_high_is_good(self):
        """With expected_high=True, high scores are no drift."""
        auditor = JudgeReliabilityAuditor()

        # High scores are expected
        high_scores = [0.85, 0.90, 0.88, 0.92, 0.87]

        drift = auditor.detect_drift(high_scores, expected_low=False)

        # These should indicate no drift (high is correct)
        self.assertEqual(drift['concern_level'], 'none')

    def test_low_drift_detection_high_is_expected(self):
        """With expected_low=False, low scores indicate drift."""
        auditor = JudgeReliabilityAuditor()

        # Low scores when high expected (> 0.85 is none, 0.60-0.85 is warning, < 0.60 is critical)
        low_scores = [0.10, 0.15, 0.12, 0.08, 0.11]

        drift = auditor.detect_drift(low_scores, expected_low=False)

        # Mean is ~0.11, which is < 0.60, so critical
        self.assertEqual(drift['concern_level'], 'critical')


if __name__ == "__main__":
    unittest.main()
