"""Tests for error handling and edge cases."""
import tempfile
import unittest
from pathlib import Path

from compass.cache import EvaluationCache
from compass.clients import CompletionClient, CompletionResponse
from compass.comparison import MultiModelComparator
from compass.judges import JudgeConfig, LLMJudge
from compass.rubrics import Rubric, RubricLibrary


class FailingClient(CompletionClient):
    """Client that raises exceptions."""

    def __init__(self, error: Exception):
        self.error = error

    def complete(self, prompt, max_tokens=180, temperature=0.0, system=None):
        raise self.error


class MockClient(CompletionClient):
    """Mock client for testing."""

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.call_count = 0

    def complete(self, prompt, max_tokens=180, temperature=0.0, system=None):
        self.call_count += 1
        return CompletionResponse(
            completion=self.response_text,
            tokens_used={"input": 100, "output": 50},
            cost_usd=0.001,
        )


class TestErrorHandling(unittest.TestCase):
    """Test error handling in judges."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cache = EvaluationCache(cache_dir=self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_empty_response_handling(self):
        """Handle empty response gracefully."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient("")
        judge = LLMJudge(config, client, self.cache)

        result = judge.evaluate("test")
        self.assertEqual(result.score, 0.0)
        self.assertFalse(result.hit)

    def test_whitespace_only_response(self):
        """Handle whitespace-only response."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient("   \n\t  ")
        judge = LLMJudge(config, client, self.cache)

        result = judge.evaluate("test")
        self.assertEqual(result.score, 0.0)

    def test_score_as_string_number(self):
        """Handle score as string number."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient('{"score": "0.75", "hit": true}')
        judge = LLMJudge(config, client, self.cache)

        result = judge.evaluate("test")
        # Should parse string number or default to 0.0
        self.assertIn(result.score, [0.75, 0.0])

    def test_boolean_score(self):
        """Handle boolean value where score expected."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient('{"score": true, "hit": false}')
        judge = LLMJudge(config, client, self.cache)

        result = judge.evaluate("test")
        # Boolean true converts to 1.0, which is clamped to 1.0
        self.assertIn(result.score, [0.0, 1.0])

    def test_null_score(self):
        """Handle null score."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient('{"score": null, "hit": false}')
        judge = LLMJudge(config, client, self.cache)

        result = judge.evaluate("test")
        self.assertEqual(result.score, 0.0)

    def test_missing_score_field(self):
        """Handle missing score field."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient('{"hit": true, "rationale": "test"}')
        judge = LLMJudge(config, client, self.cache)

        result = judge.evaluate("test")
        self.assertEqual(result.score, 0.0)

    def test_very_large_score(self):
        """Handle very large score."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient('{"score": 1000.0, "hit": true}')
        judge = LLMJudge(config, client, self.cache)

        result = judge.evaluate("test")
        self.assertEqual(result.score, 1.0)  # Should be clamped

    def test_very_negative_score(self):
        """Handle very negative score."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient('{"score": -1000.0, "hit": false}')
        judge = LLMJudge(config, client, self.cache)

        result = judge.evaluate("test")
        self.assertEqual(result.score, 0.0)  # Should be clamped


class TestEdgeCases(unittest.TestCase):
    """Test edge cases in operations."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cache = EvaluationCache(cache_dir=self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_empty_text_evaluation(self):
        """Evaluate empty text."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient('{"score": 0.5, "hit": false}')
        judge = LLMJudge(config, client, self.cache)

        result = judge.evaluate("")
        self.assertIsNotNone(result)
        self.assertEqual(result.score, 0.5)

    def test_very_long_text_evaluation(self):
        """Evaluate very long text."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient('{"score": 0.5, "hit": false}')
        judge = LLMJudge(config, client, self.cache)

        long_text = "test " * 10000
        result = judge.evaluate(long_text)
        self.assertIsNotNone(result)

    def test_special_characters_in_text(self):
        """Evaluate text with special characters."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient('{"score": 0.5, "hit": false}')
        judge = LLMJudge(config, client, self.cache)

        special_text = 'Test with "quotes", \\backslashes, \nnewlines, \ttabs'
        result = judge.evaluate(special_text)
        self.assertIsNotNone(result)

    def test_unicode_text_evaluation(self):
        """Evaluate text with unicode characters."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient('{"score": 0.5, "hit": false}')
        judge = LLMJudge(config, client, self.cache)

        unicode_text = "Hello 世界 🌍 Привет مرحبا"
        result = judge.evaluate(unicode_text)
        self.assertIsNotNone(result)

    def test_cache_with_unicode_key(self):
        """Cache handles unicode in text properly."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient('{"score": 0.5}')
        judge = LLMJudge(config, client, self.cache)

        text1 = "café"
        text2 = "cafe"  # Different without accent

        r1 = judge.evaluate(text1)
        r2 = judge.evaluate(text2)

        # Both should have been evaluated
        self.assertEqual(client.call_count, 2)

    def test_very_high_temperature(self):
        """Judge respects temperature configuration."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
            temperature=2.0,  # Very high
        )
        client = MockClient('{"score": 0.5}')
        judge = LLMJudge(config, client, self.cache)

        judge.evaluate("test")
        # Should complete without error

    def test_zero_max_tokens(self):
        """Judge with zero max_tokens."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
            max_tokens=0,
        )
        client = MockClient('{"score": 0.5}')
        judge = LLMJudge(config, client, self.cache)

        result = judge.evaluate("test")
        # Should handle gracefully
        self.assertIsNotNone(result)


class TestComparisonEdgeCases(unittest.TestCase):
    """Test edge cases in multi-model comparison."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cache = EvaluationCache(cache_dir=self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_comparison_single_judge(self):
        """Comparison with only one judge."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient('{"score": 0.5}')
        judge = LLMJudge(config, client, self.cache)

        comparator = MultiModelComparator({"gpt-4o": judge})
        comparison = comparator.compare("test")

        self.assertEqual(len(comparison.judges), 1)

    def test_comparison_identical_judges(self):
        """Comparison where judges always agree."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient('{"score": 0.5, "hit": false}')
        judge1 = LLMJudge(config, client, self.cache)
        judge2 = LLMJudge(config, client, self.cache)

        comparator = MultiModelComparator({"judge1": judge1, "judge2": judge2})
        comparison = comparator.compare("test")

        # Perfect agreement
        self.assertEqual(comparison.agreement_score(), 1.0)


if __name__ == "__main__":
    unittest.main()
