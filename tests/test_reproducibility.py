"""Tests for reproducibility and versioning."""
import unittest
from datetime import datetime

from compass.judges import EvaluationResult
from compass.reproducibility import (
    EvaluationMetadata,
    cost_per_judge,
    cost_summary,
    reproducibility_report,
)


class TestEvaluationMetadata(unittest.TestCase):
    """Test EvaluationMetadata."""

    def test_metadata_creation(self):
        metadata = EvaluationMetadata(
            compass_version="0.1.0",
            rubric_hash="abc123",
            judge_model="gpt-4o",
            seed=42,
            timestamp="2026-05-21T10:00:00",
            python_version="3.9.7",
        )
        self.assertEqual(metadata.compass_version, "0.1.0")
        self.assertEqual(metadata.rubric_hash, "abc123")
        self.assertEqual(metadata.judge_model, "gpt-4o")

    def test_metadata_frozen(self):
        metadata = EvaluationMetadata(
            compass_version="0.1.0",
            rubric_hash="abc123",
            judge_model="gpt-4o",
            seed=42,
            timestamp="2026-05-21T10:00:00",
            python_version="3.9.7",
        )
        with self.assertRaises(AttributeError):
            metadata.compass_version = "0.2.0"

    def test_metadata_to_dict(self):
        metadata = EvaluationMetadata(
            compass_version="0.1.0",
            rubric_hash="abc123",
            judge_model="gpt-4o",
            seed=42,
            timestamp="2026-05-21T10:00:00",
            python_version="3.9.7",
        )
        d = metadata.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["compass_version"], "0.1.0")
        self.assertEqual(d["judge_model"], "gpt-4o")

    def test_metadata_from_result(self):
        result = EvaluationResult(
            name="sycophancy",
            score=0.8,
            hit=True,
            rubric_hash="abc123",
            judge_model="gpt-4o",
            timestamp="2026-05-21T10:00:00",
        )
        metadata = EvaluationMetadata.from_result(result, "0.1.0")
        self.assertEqual(metadata.compass_version, "0.1.0")
        self.assertEqual(metadata.rubric_hash, "abc123")
        self.assertEqual(metadata.judge_model, "gpt-4o")

    def test_metadata_immutable_seed(self):
        metadata = EvaluationMetadata(
            compass_version="0.1.0",
            rubric_hash="abc123",
            judge_model="gpt-4o",
            seed=42,
            timestamp="2026-05-21T10:00:00",
            python_version="3.9.7",
        )
        self.assertEqual(metadata.seed, 42)
        with self.assertRaises(AttributeError):
            metadata.seed = 100


class TestCostSummary(unittest.TestCase):
    """Test cost aggregation."""

    def test_cost_summary_single_result(self):
        result = EvaluationResult(
            name="sycophancy",
            score=0.8,
            hit=True,
            rubric_hash="abc123",
            judge_model="gpt-4o",
            tokens_used={"input": 100, "output": 50},
            cost_usd=0.01,
        )
        summary = cost_summary([result])
        self.assertEqual(summary["total_cost_usd"], 0.01)
        self.assertEqual(summary["total_input_tokens"], 100)
        self.assertEqual(summary["total_output_tokens"], 50)
        self.assertEqual(summary["total_tokens"], 150)
        self.assertEqual(summary["results_count"], 1)

    def test_cost_summary_multiple_results(self):
        results = [
            EvaluationResult(
                name="sycophancy",
                score=0.8,
                hit=True,
                rubric_hash="abc123",
                judge_model="gpt-4o",
                tokens_used={"input": 100, "output": 50},
                cost_usd=0.01,
            ),
            EvaluationResult(
                name="sycophancy",
                score=0.6,
                hit=False,
                rubric_hash="abc123",
                judge_model="gpt-4o",
                tokens_used={"input": 100, "output": 50},
                cost_usd=0.01,
            ),
        ]
        summary = cost_summary(results)
        self.assertEqual(summary["total_cost_usd"], 0.02)
        self.assertEqual(summary["total_input_tokens"], 200)
        self.assertEqual(summary["total_output_tokens"], 100)
        self.assertEqual(summary["total_tokens"], 300)
        self.assertEqual(summary["results_count"], 2)

    def test_cost_summary_empty(self):
        summary = cost_summary([])
        self.assertEqual(summary["total_cost_usd"], 0.0)
        self.assertEqual(summary["total_input_tokens"], 0)
        self.assertEqual(summary["total_output_tokens"], 0)
        self.assertEqual(summary["results_count"], 0)

    def test_cost_summary_missing_tokens(self):
        result = EvaluationResult(
            name="sycophancy",
            score=0.8,
            hit=True,
            rubric_hash="abc123",
            judge_model="gpt-4o",
            tokens_used={},
            cost_usd=0.01,
        )
        summary = cost_summary([result])
        self.assertEqual(summary["total_input_tokens"], 0)
        self.assertEqual(summary["total_output_tokens"], 0)


class TestReproducibilityReport(unittest.TestCase):
    """Test reproducibility report generation."""

    def test_report_with_results(self):
        results = [
            EvaluationResult(
                name="sycophancy",
                score=0.8,
                hit=True,
                rubric_hash="abc123",
                judge_model="gpt-4o",
                prompt_version="1.0",
                tokens_used={"input": 100, "output": 50},
                cost_usd=0.01,
            ),
        ]
        report = reproducibility_report(results)
        self.assertIn("REPRODUCIBILITY REPORT", report)
        self.assertIn("abc123", report)
        self.assertIn("gpt-4o", report)
        self.assertIn("1.0", report)

    def test_report_with_metadata(self):
        metadata = EvaluationMetadata(
            compass_version="0.1.0",
            rubric_hash="abc123",
            judge_model="gpt-4o",
            seed=42,
            timestamp="2026-05-21T10:00:00",
            python_version="3.9.7",
        )
        results = [
            EvaluationResult(
                name="sycophancy",
                score=0.8,
                hit=True,
                rubric_hash="abc123",
                judge_model="gpt-4o",
                tokens_used={"input": 100, "output": 50},
                cost_usd=0.01,
            ),
        ]
        report = reproducibility_report(results, metadata)
        self.assertIn("0.1.0", report)
        self.assertIn("3.9.7", report)
        self.assertIn("2026-05-21T10:00:00", report)

    def test_report_cache_performance(self):
        results = [
            EvaluationResult(
                name="sycophancy",
                score=0.8,
                hit=True,
                rubric_hash="abc123",
                judge_model="gpt-4o",
                cache_hit=True,
            ),
            EvaluationResult(
                name="sycophancy",
                score=0.6,
                hit=False,
                rubric_hash="abc123",
                judge_model="gpt-4o",
                cache_hit=False,
            ),
        ]
        report = reproducibility_report(results)
        self.assertIn("Cache Performance", report)
        self.assertIn("1/2", report)

    def test_report_empty_results(self):
        report = reproducibility_report([])
        self.assertIn("REPRODUCIBILITY REPORT", report)


class TestCostPerJudge(unittest.TestCase):
    """Test cost breakdown by judge model."""

    def test_cost_per_judge_single(self):
        results = [
            EvaluationResult(
                name="sycophancy",
                score=0.8,
                hit=True,
                judge_model="gpt-4o",
                cost_usd=0.01,
            ),
            EvaluationResult(
                name="sycophancy",
                score=0.7,
                hit=False,
                judge_model="gpt-4o",
                cost_usd=0.01,
            ),
        ]
        breakdown = cost_per_judge(results)
        self.assertIn("gpt-4o", breakdown)
        self.assertEqual(breakdown["gpt-4o"]["total_cost_usd"], 0.02)
        self.assertEqual(breakdown["gpt-4o"]["count"], 2)
        self.assertEqual(breakdown["gpt-4o"]["avg_cost_per_eval"], 0.01)

    def test_cost_per_judge_multiple(self):
        results = [
            EvaluationResult(
                name="sycophancy",
                score=0.8,
                hit=True,
                judge_model="gpt-4o",
                cost_usd=0.02,
            ),
            EvaluationResult(
                name="sycophancy",
                score=0.7,
                hit=False,
                judge_model="claude-opus-4-7",
                cost_usd=0.01,
            ),
            EvaluationResult(
                name="sycophancy",
                score=0.6,
                hit=False,
                judge_model="claude-opus-4-7",
                cost_usd=0.01,
            ),
        ]
        breakdown = cost_per_judge(results)
        self.assertEqual(len(breakdown), 2)
        self.assertEqual(breakdown["gpt-4o"]["total_cost_usd"], 0.02)
        self.assertEqual(breakdown["gpt-4o"]["count"], 1)
        self.assertEqual(breakdown["claude-opus-4-7"]["total_cost_usd"], 0.02)
        self.assertEqual(breakdown["claude-opus-4-7"]["count"], 2)
        self.assertEqual(breakdown["claude-opus-4-7"]["avg_cost_per_eval"], 0.01)

    def test_cost_per_judge_empty(self):
        breakdown = cost_per_judge([])
        self.assertEqual(len(breakdown), 0)


if __name__ == "__main__":
    unittest.main()
