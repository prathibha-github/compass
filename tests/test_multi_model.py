"""Tests for multi-model comparison."""
import tempfile
import unittest

from compass.cache import EvaluationCache
from compass.clients import CompletionClient, CompletionResponse
from compass.comparison import ComparisonResult, MultiModelComparator
from compass.judges import JudgeConfig, LLMJudge
from compass.rubrics import RubricLibrary


class MockClient(CompletionClient):
    """Mock client returning fixed response."""

    def __init__(self, response_text: str):
        self.response_text = response_text

    def complete(self, prompt, max_tokens=180, temperature=0.0, system=None):
        return CompletionResponse(completion=self.response_text)


class TestComparisonResult(unittest.TestCase):
    """Test ComparisonResult dataclass."""

    def test_comparison_creation(self):
        result1 = self._result(0.8, "model1")
        result2 = self._result(0.9, "model2")

        comparison = ComparisonResult(
            rubric_name="sycophancy",
            text="test",
            judges={"model1": result1, "model2": result2},
        )

        self.assertEqual(comparison.rubric_name, "sycophancy")
        self.assertEqual(len(comparison.judges), 2)

    def test_agreement_score_perfect(self):
        """Perfect agreement when all judges agree."""
        result1 = self._result(0.75, "model1")
        result2 = self._result(0.75, "model2")
        result3 = self._result(0.75, "model3")

        comparison = ComparisonResult(
            rubric_name="test",
            text="text",
            judges={"m1": result1, "m2": result2, "m3": result3},
        )

        self.assertAlmostEqual(comparison.agreement_score(), 1.0, places=2)

    def test_agreement_score_no_agreement(self):
        """Low agreement when judges disagree."""
        result1 = self._result(0.0, "model1")
        result2 = self._result(1.0, "model2")

        comparison = ComparisonResult(
            rubric_name="test",
            text="text",
            judges={"m1": result1, "m2": result2},
        )

        agreement = comparison.agreement_score()
        self.assertLess(agreement, 0.5)

    def test_agreement_score_single_judge(self):
        """Single judge has perfect agreement (trivial)."""
        result = self._result(0.5, "model1")
        comparison = ComparisonResult(
            rubric_name="test",
            text="text",
            judges={"m1": result},
        )

        self.assertEqual(comparison.agreement_score(), 1.0)

    def test_hit_agreement_all_agree_true(self):
        """Hit agreement when all judges hit."""
        result1 = self._result_with_hit(0.8, True, "model1")
        result2 = self._result_with_hit(0.7, True, "model2")

        comparison = ComparisonResult(
            rubric_name="test",
            text="text",
            judges={"m1": result1, "m2": result2},
        )

        self.assertEqual(comparison.hit_agreement(), 1.0)

    def test_hit_agreement_all_agree_false(self):
        """Hit agreement when all judges miss."""
        result1 = self._result_with_hit(0.3, False, "model1")
        result2 = self._result_with_hit(0.2, False, "model2")

        comparison = ComparisonResult(
            rubric_name="test",
            text="text",
            judges={"m1": result1, "m2": result2},
        )

        self.assertEqual(comparison.hit_agreement(), 1.0)

    def test_hit_agreement_split(self):
        """Hit agreement when judges split."""
        result1 = self._result_with_hit(0.8, True, "model1")
        result2 = self._result_with_hit(0.2, False, "model2")

        comparison = ComparisonResult(
            rubric_name="test",
            text="text",
            judges={"m1": result1, "m2": result2},
        )

        agreement = comparison.hit_agreement()
        self.assertGreater(agreement, 0.0)
        self.assertLess(agreement, 1.0)

    def test_score_range(self):
        """Score range returns min and max."""
        result1 = self._result(0.3, "model1")
        result2 = self._result(0.8, "model2")
        result3 = self._result(0.5, "model3")

        comparison = ComparisonResult(
            rubric_name="test",
            text="text",
            judges={"m1": result1, "m2": result2, "m3": result3},
        )

        min_score, max_score = comparison.score_range()
        self.assertEqual(min_score, 0.3)
        self.assertEqual(max_score, 0.8)

    def test_summary_readable(self):
        """Summary is human-readable."""
        result1 = self._result_with_hit(0.8, True, "gpt-4o")
        result2 = self._result_with_hit(0.6, False, "claude-opus-4-7")

        comparison = ComparisonResult(
            rubric_name="sycophancy",
            text="test response",
            judges={"gpt-4o": result1, "claude-opus-4-7": result2},
        )

        summary = comparison.summary()
        self.assertIn("sycophancy", summary)
        self.assertIn("gpt-4o", summary)
        self.assertIn("claude-opus-4-7", summary)
        self.assertIn("Agreement", summary)

    @staticmethod
    def _result(score: float, model: str):
        from compass.judges import EvaluationResult

        return EvaluationResult(
            name="test",
            score=score,
            hit=score >= 0.5,
            rubric_hash="abc123",
            judge_model=model,
        )

    @staticmethod
    def _result_with_hit(score: float, hit: bool, model: str):
        from compass.judges import EvaluationResult

        return EvaluationResult(
            name="test",
            score=score,
            hit=hit,
            rubric_hash="abc123",
            judge_model=model,
        )


class TestMultiModelComparator(unittest.TestCase):
    """Test MultiModelComparator."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cache = EvaluationCache(cache_dir=self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_comparator_creation(self):
        judge1 = self._judge("model1")
        judge2 = self._judge("model2")

        comparator = MultiModelComparator({"model1": judge1, "model2": judge2})
        self.assertEqual(len(comparator.judges), 2)

    def test_comparator_compare(self):
        judge1 = self._judge("model1", '{"score": 0.8, "hit": true}')
        judge2 = self._judge("model2", '{"score": 0.6, "hit": false}')

        comparator = MultiModelComparator({"model1": judge1, "model2": judge2})
        comparison = comparator.compare("test text")

        self.assertEqual(len(comparison.judges), 2)
        self.assertIn("model1", comparison.judges)
        self.assertIn("model2", comparison.judges)
        self.assertEqual(comparison.judges["model1"].score, 0.8)
        self.assertEqual(comparison.judges["model2"].score, 0.6)

    def test_comparator_requires_judges(self):
        comparator = MultiModelComparator({})
        with self.assertRaises(ValueError):
            comparator.compare("test")

    def test_comparator_compare_batch(self):
        judge1 = self._judge("model1", '{"score": 0.5}')
        judge2 = self._judge("model2", '{"score": 0.7}')

        comparator = MultiModelComparator({"model1": judge1, "model2": judge2})
        texts = ["text1", "text2", "text3"]
        comparisons = comparator.compare_batch(texts)

        self.assertEqual(len(comparisons), 3)
        self.assertEqual(comparisons[0].text, "text1")
        self.assertEqual(comparisons[1].text, "text2")
        self.assertEqual(comparisons[2].text, "text3")

    def test_agreement_stats(self):
        judge1 = self._judge("model1", '{"score": 0.8}')
        judge2 = self._judge("model2", '{"score": 0.7}')

        comparator = MultiModelComparator({"model1": judge1, "model2": judge2})
        texts = ["text1", "text2", "text3"]
        comparisons = comparator.compare_batch(texts)

        stats = comparator.agreement_stats(comparisons)

        self.assertIn("mean_agreement", stats)
        self.assertIn("std_agreement", stats)
        self.assertIn("min_agreement", stats)
        self.assertIn("max_agreement", stats)
        self.assertIn("n_comparisons", stats)
        self.assertEqual(stats["n_comparisons"], 3)

    def test_agreement_stats_empty(self):
        comparator = MultiModelComparator({})
        stats = comparator.agreement_stats([])
        self.assertEqual(stats, {})

    def _judge(self, model: str, response: str = '{"score": 0.5}'):
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model=model,
        )
        client = MockClient(response)
        return LLMJudge(config, client, self.cache)


if __name__ == "__main__":
    unittest.main()
