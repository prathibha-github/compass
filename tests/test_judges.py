"""Tests for judge abstractions."""
import unittest

from compass.judges import EvaluationResult, JudgeConfig
from compass.rubrics import RubricLibrary


class TestEvaluationResult(unittest.TestCase):
    """Test EvaluationResult dataclass."""

    def test_result_creation(self):
        result = EvaluationResult(
            name="sycophancy",
            score=0.75,
            hit=True,
            confidence=0.9,
            rationale="Test rationale",
        )
        self.assertEqual(result.name, "sycophancy")
        self.assertEqual(result.score, 0.75)
        self.assertTrue(result.hit)
        self.assertEqual(result.confidence, 0.9)

    def test_result_to_dict(self):
        result = EvaluationResult(
            name="sycophancy",
            score=0.75,
            hit=True,
            confidence=0.9,
            rationale="Test",
            rubric_hash="abc123",
            judge_model="gpt-4o",
        )
        d = result.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["name"], "sycophancy")
        self.assertEqual(d["score"], 0.75)
        self.assertEqual(d["rubric_hash"], "abc123")

    def test_result_default_confidence_none(self):
        result = EvaluationResult(
            name="test",
            score=0.5,
            hit=False,
        )
        self.assertIsNone(result.confidence)

    def test_result_default_rationale_empty(self):
        result = EvaluationResult(
            name="test",
            score=0.5,
            hit=False,
        )
        self.assertEqual(result.rationale, "")

    def test_result_repr(self):
        result = EvaluationResult(
            name="sycophancy",
            score=0.75,
            hit=True,
        )
        repr_str = repr(result)
        self.assertIn("sycophancy", repr_str)
        self.assertIn("0.75", repr_str)

    def test_result_with_token_usage(self):
        result = EvaluationResult(
            name="test",
            score=0.5,
            hit=False,
            tokens_used={"input": 100, "output": 50},
            cost_usd=0.001,
        )
        self.assertEqual(result.tokens_used["input"], 100)
        self.assertEqual(result.cost_usd, 0.001)


class TestJudgeConfig(unittest.TestCase):
    """Test JudgeConfig dataclass."""

    def test_config_creation(self):
        rubric = RubricLibrary.sycophancy
        config = JudgeConfig(
            rubric=rubric,
            judge_model="gpt-4o",
            max_tokens=180,
        )
        self.assertEqual(config.rubric.name, "sycophancy")
        self.assertEqual(config.judge_model, "gpt-4o")
        self.assertEqual(config.max_tokens, 180)

    def test_config_hash_deterministic(self):
        rubric = RubricLibrary.sycophancy
        config1 = JudgeConfig(
            rubric=rubric,
            judge_model="gpt-4o",
        )
        config2 = JudgeConfig(
            rubric=rubric,
            judge_model="gpt-4o",
        )
        self.assertEqual(config1.config_hash, config2.config_hash)

    def test_config_hash_changes_with_model(self):
        rubric = RubricLibrary.sycophancy
        config1 = JudgeConfig(
            rubric=rubric,
            judge_model="gpt-4o",
        )
        config2 = JudgeConfig(
            rubric=rubric,
            judge_model="claude-opus-4-7",
        )
        self.assertNotEqual(config1.config_hash, config2.config_hash)

    def test_config_hash_changes_with_rubric(self):
        config1 = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        config2 = JudgeConfig(
            rubric=RubricLibrary.therapy_speak,
            judge_model="gpt-4o",
        )
        self.assertNotEqual(config1.config_hash, config2.config_hash)

    def test_config_defaults(self):
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        self.assertEqual(config.max_tokens, 180)
        self.assertEqual(config.temperature, 0.0)
        self.assertEqual(config.seed, 42)

    def test_config_repr(self):
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        repr_str = repr(config)
        self.assertIn("sycophancy", repr_str)
        self.assertIn("gpt-4o", repr_str)


if __name__ == "__main__":
    unittest.main()
