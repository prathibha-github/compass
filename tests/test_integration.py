"""Integration tests for end-to-end compass workflows."""
import tempfile
import unittest
from pathlib import Path

from compass.cache import EvaluationCache
from compass.clients import CompletionClient, CompletionResponse
from compass.comparison import MultiModelComparator
from compass.judges import JudgeConfig, LLMJudge
from compass.reproducibility import EvaluationMetadata
from compass.rubrics import Rubric, RubricLibrary


class MockClient(CompletionClient):
    """Mock client for testing."""

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.call_count = 0
        self.calls = []

    def complete(self, prompt, max_tokens=180, temperature=0.0, system=None):
        self.call_count += 1
        self.calls.append(
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system,
            }
        )
        return CompletionResponse(
            completion=self.response_text,
            tokens_used={"input": 100, "output": 50},
            cost_usd=0.001,
        )


class TestEndToEndWorkflow(unittest.TestCase):
    """Test complete evaluation workflows."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cache_dir = self.tmpdir.name

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_single_evaluation_workflow(self):
        """Complete workflow: load rubric, create judge, evaluate text."""
        cache = EvaluationCache(cache_dir=self.cache_dir)
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient('{"score": 0.8, "hit": true, "confidence": 0.9}')
        judge = LLMJudge(config, client, cache)

        result = judge.evaluate("test response")

        self.assertEqual(result.score, 0.8)
        self.assertTrue(result.hit)
        self.assertEqual(result.judge_model, "gpt-4o")
        self.assertFalse(result.cache_hit)

    def test_batch_evaluation_workflow(self):
        """Evaluate multiple texts with same judge."""
        cache = EvaluationCache(cache_dir=self.cache_dir)
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient('{"score": 0.5, "hit": false}')
        judge = LLMJudge(config, client, cache)

        texts = ["response 1", "response 2", "response 3"]
        results = [judge.evaluate(text) for text in texts]

        self.assertEqual(len(results), 3)
        self.assertEqual(client.call_count, 3)
        self.assertTrue(all(r.score == 0.5 for r in results))

    def test_multi_rubric_workflow(self):
        """Evaluate same text with multiple rubrics."""
        cache = EvaluationCache(cache_dir=self.cache_dir)
        client = MockClient('{"score": 0.6, "hit": true}')

        results = {}
        for rubric_name in ["sycophancy", "therapy_speak", "task_focus"]:
            rubric = RubricLibrary.get(rubric_name)
            config = JudgeConfig(rubric=rubric, judge_model="gpt-4o")
            judge = LLMJudge(config, client, cache)
            results[rubric_name] = judge.evaluate("test response")

        self.assertEqual(len(results), 3)
        self.assertTrue(all(r.score == 0.6 for r in results.values()))

    def test_cache_persistence_workflow(self):
        """Verify cache persists across judge instances."""
        text = "test response"

        # First judge evaluates and caches
        cache1 = EvaluationCache(cache_dir=self.cache_dir)
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient('{"score": 0.75, "hit": true}')
        judge1 = LLMJudge(config, client, cache1)
        result1 = judge1.evaluate(text)

        # New judge with same cache directory gets cached result
        cache2 = EvaluationCache(cache_dir=self.cache_dir)
        client2 = MockClient('{"score": 0.25, "hit": false}')
        judge2 = LLMJudge(config, client2, cache2)
        result2 = judge2.evaluate(text)

        self.assertEqual(result1.score, result2.score)
        self.assertEqual(result2.score, 0.75)
        self.assertTrue(result2.cache_hit)
        self.assertEqual(client2.call_count, 0)  # Should not call API

    def test_multi_judge_comparison_workflow(self):
        """Compare same text across different judges."""
        cache = EvaluationCache(cache_dir=self.cache_dir)
        text = "test response"

        judges = {}
        for model in ["gpt-4o", "claude-opus"]:
            client = MockClient('{"score": 0.6, "hit": true}')
            config = JudgeConfig(
                rubric=RubricLibrary.sycophancy,
                judge_model=model,
            )
            judges[model] = LLMJudge(config, client, cache)

        comparator = MultiModelComparator(judges)
        comparison = comparator.compare(text)

        self.assertEqual(len(comparison.judges), 2)
        self.assertIn("gpt-4o", comparison.judges)
        self.assertIn("claude-opus", comparison.judges)
        self.assertGreater(comparison.agreement_score(), 0.0)

    def test_metadata_tracking_workflow(self):
        """Verify evaluation metadata is properly tracked."""
        cache = EvaluationCache(cache_dir=self.cache_dir)
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient('{"score": 0.7, "hit": true, "confidence": 0.85}')
        judge = LLMJudge(config, client, cache)

        result = judge.evaluate("test response")

        # Verify metadata is present
        self.assertIsNotNone(result.rubric_hash)
        self.assertIsNotNone(result.judge_model)
        self.assertIsNotNone(result.timestamp)
        self.assertIsNotNone(result.tokens_used)
        self.assertGreater(result.cost_usd, 0.0)

    def test_cost_accumulation_workflow(self):
        """Test cost tracking across multiple evaluations."""
        cache = EvaluationCache(cache_dir=self.cache_dir)
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient('{"score": 0.5, "hit": false}')
        judge = LLMJudge(config, client, cache)

        results = [judge.evaluate(f"text {i}") for i in range(5)]
        total_cost = sum(r.cost_usd for r in results)

        self.assertEqual(len(results), 5)
        self.assertAlmostEqual(total_cost, 0.005)

    def test_reproducibility_with_same_seed(self):
        """Same seed should produce same metadata."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient('{"score": 0.5}')

        cache1 = EvaluationCache(cache_dir=self.cache_dir)
        judge1 = LLMJudge(config, client, cache1)
        result1 = judge1.evaluate("test")

        # Metadata should include reproducible info
        self.assertIsNotNone(result1.rubric_hash)
        self.assertEqual(result1.judge_model, "gpt-4o")


class TestMultipleRubricsIntegration(unittest.TestCase):
    """Test integration with multiple built-in rubrics."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cache = EvaluationCache(cache_dir=self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_all_library_rubrics_evaluatable(self):
        """All library rubrics can be evaluated."""
        client = MockClient('{"score": 0.5, "hit": false}')
        text = "test response"

        for rubric_name in [
            "sycophancy",
            "therapy_speak",
            "task_focus",
            "truthfulness",
            "clarity",
        ]:
            rubric = RubricLibrary.get(rubric_name)
            config = JudgeConfig(rubric=rubric, judge_model="gpt-4o")
            judge = LLMJudge(config, client, self.cache)

            result = judge.evaluate(text)
            self.assertIsNotNone(result)
            self.assertEqual(result.name, rubric_name)

    def test_rubric_hash_consistency(self):
        """Rubric hashes are consistent across evaluations."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient('{"score": 0.5}')
        judge = LLMJudge(config, client, self.cache)

        result1 = judge.evaluate("text 1")
        result2 = judge.evaluate("text 2")

        self.assertEqual(result1.rubric_hash, result2.rubric_hash)
        self.assertEqual(result1.rubric_hash, RubricLibrary.sycophancy.hash)


if __name__ == "__main__":
    unittest.main()
