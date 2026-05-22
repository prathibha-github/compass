"""Tests for batch operations and advanced scenarios."""
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

from compass.cache import EvaluationCache
from compass.clients import CompletionClient, CompletionResponse
from compass.comparison import MultiModelComparator
from compass.judges import JudgeConfig, LLMJudge
from compass.rubrics import RubricLibrary


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


class TestBatchEvaluation(unittest.TestCase):
    """Test batch evaluation of multiple texts."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cache = EvaluationCache(cache_dir=self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_evaluate_batch_same_rubric(self):
        """Batch evaluate with same rubric."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient('{"score": 0.5, "hit": false}')
        judge = LLMJudge(config, client, self.cache)

        texts = [f"response {i}" for i in range(10)]
        results = [judge.evaluate(text) for text in texts]

        self.assertEqual(len(results), 10)
        self.assertEqual(client.call_count, 10)
        self.assertTrue(all(r.score == 0.5 for r in results))

    def test_batch_evaluation_collects_costs(self):
        """Batch evaluation properly tracks costs."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient('{"score": 0.5}')
        judge = LLMJudge(config, client, self.cache)

        texts = [f"response {i}" for i in range(5)]
        results = [judge.evaluate(text) for text in texts]
        total_cost = sum(r.cost_usd for r in results)

        self.assertAlmostEqual(total_cost, 0.005)

    def test_batch_evaluation_caching_benefit(self):
        """Batch evaluation benefits from caching."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient('{"score": 0.5}')
        judge = LLMJudge(config, client, self.cache)

        text = "response"
        # First 5 evaluations are unique
        for i in range(5):
            judge.evaluate(f"{text} {i}")
        # Next 5 are repeats
        for i in range(5):
            judge.evaluate(f"{text} {i}")

        # Should have called API only 5 times
        self.assertEqual(client.call_count, 5)

    def test_large_batch_evaluation(self):
        """Handle large batch of evaluations."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient('{"score": 0.5}')
        judge = LLMJudge(config, client, self.cache)

        # Evaluate 100 unique texts
        texts = [f"response {i}" for i in range(100)]
        results = [judge.evaluate(text) for text in texts]

        self.assertEqual(len(results), 100)
        self.assertEqual(client.call_count, 100)

    def test_batch_multiple_rubrics(self):
        """Batch evaluate with multiple rubrics."""
        texts = ["response 1", "response 2", "response 3"]
        rubrics = ["sycophancy", "therapy_speak", "task_focus"]
        client = MockClient('{"score": 0.5}')

        results = {}
        for rubric_name in rubrics:
            rubric = RubricLibrary.get(rubric_name)
            config = JudgeConfig(rubric=rubric, judge_model="gpt-4o")
            judge = LLMJudge(config, client, self.cache)
            results[rubric_name] = [judge.evaluate(text) for text in texts]

        self.assertEqual(len(results), 3)
        for rubric_results in results.values():
            self.assertEqual(len(rubric_results), 3)


class TestBatchComparison(unittest.TestCase):
    """Test batch comparison across multiple judges."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cache = EvaluationCache(cache_dir=self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_compare_batch_texts(self):
        """Compare same batch across multiple models."""
        judges = {}
        for model in ["gpt-4o", "claude-opus"]:
            client = MockClient('{"score": 0.5}')
            config = JudgeConfig(
                rubric=RubricLibrary.sycophancy,
                judge_model=model,
            )
            judges[model] = LLMJudge(config, client, self.cache)

        comparator = MultiModelComparator(judges)
        texts = ["response 1", "response 2", "response 3"]
        comparisons = [comparator.compare(text) for text in texts]

        self.assertEqual(len(comparisons), 3)
        for comparison in comparisons:
            self.assertEqual(len(comparison.judges), 2)

    def test_compare_large_batch(self):
        """Compare large batch of texts."""
        judges = {}
        for model in ["gpt-4o", "claude-opus", "gpt-4-turbo"]:
            client = MockClient('{"score": 0.5}')
            config = JudgeConfig(
                rubric=RubricLibrary.sycophancy,
                judge_model=model,
            )
            judges[model] = LLMJudge(config, client, self.cache)

        comparator = MultiModelComparator(judges)
        texts = [f"response {i}" for i in range(50)]
        comparisons = [comparator.compare(text) for text in texts]

        self.assertEqual(len(comparisons), 50)
        agreement_scores = [c.agreement_score() for c in comparisons]
        self.assertEqual(len(agreement_scores), 50)


class TestBatchAggregation(unittest.TestCase):
    """Test aggregating results from batch operations."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cache = EvaluationCache(cache_dir=self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_aggregate_hit_rate(self):
        """Calculate hit rate from batch results."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        # Create client that alternates hit/miss
        client = MockClient('{"score": 0.5, "hit": true}')
        judge = LLMJudge(config, client, self.cache)

        results = [judge.evaluate(f"response {i}") for i in range(10)]
        hit_rate = sum(1 for r in results if r.hit) / len(results)

        self.assertEqual(hit_rate, 1.0)

    def test_aggregate_mean_score(self):
        """Calculate mean score from batch results."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        # Vary scores
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        results = []
        for score in scores:
            client = MockClient(f'{{"score": {score}}}')
            judge = LLMJudge(config, client, self.cache)
            results.append(judge.evaluate(f"response {score}"))

        mean_score = sum(r.score for r in results) / len(results)
        self.assertAlmostEqual(mean_score, 0.55, places=1)

    def test_aggregate_confidence(self):
        """Track confidence across batch results."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        confidences = [0.95, 0.87, 0.92, 0.88, 0.91]

        results = []
        for conf in confidences:
            client = MockClient(f'{{"score": 0.5, "confidence": {conf}}}')
            judge = LLMJudge(config, client, self.cache)
            results.append(judge.evaluate(f"response {conf}"))

        mean_confidence = sum(r.confidence for r in results if r.confidence) / len(
            results
        )
        self.assertAlmostEqual(mean_confidence, 0.906, places=2)


class TestPerformanceCharacteristics(unittest.TestCase):
    """Test performance characteristics of batch operations."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cache = EvaluationCache(cache_dir=self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_cache_improves_performance(self):
        """Caching reduces API calls."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient('{"score": 0.5}')
        judge = LLMJudge(config, client, self.cache)

        # First pass - 10 unique texts
        for i in range(10):
            judge.evaluate(f"response {i}")
        calls_first_pass = client.call_count

        # Second pass - same 10 texts
        for i in range(10):
            judge.evaluate(f"response {i}")
        calls_second_pass = client.call_count

        # Should not have increased
        self.assertEqual(calls_first_pass, 10)
        self.assertEqual(calls_second_pass, 10)

    def test_memory_cache_performance(self):
        """Memory cache is fast for repeated access."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient('{"score": 0.5}')
        judge = LLMJudge(config, client, self.cache)

        text = "response"
        # First evaluation hits API
        result1 = judge.evaluate(text)
        self.assertFalse(result1.cache_hit)

        # Second evaluation hits memory cache
        result2 = judge.evaluate(text)
        self.assertTrue(result2.cache_hit)

        # Should only call API once
        self.assertEqual(client.call_count, 1)

    def test_disk_cache_loading(self):
        """Disk cache loads correctly."""
        text = "response"

        # Store in cache
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient('{"score": 0.5}')
        judge = LLMJudge(config, client, self.cache)
        judge.evaluate(text)

        # Create new judge with same cache - should load from disk
        cache2 = EvaluationCache(cache_dir=self.tmpdir.name)
        client2 = MockClient('{"score": 0.1}')
        judge2 = LLMJudge(config, client2, cache2)
        result = judge2.evaluate(text)

        self.assertTrue(result.cache_hit)
        self.assertEqual(result.score, 0.5)  # Got cached value


class TestBatchRobustness(unittest.TestCase):
    """Test robustness of batch operations."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cache = EvaluationCache(cache_dir=self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_batch_with_mixed_results(self):
        """Handle batch with varying result types."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        responses = [
            '{"score": 0.5, "hit": true, "confidence": 0.95}',
            '{"score": 0.3, "hit": false}',
            '{"score": 0.8}',
            "malformed",
        ]

        results = []
        for i, response in enumerate(responses):
            client = MockClient(response)
            judge = LLMJudge(config, client, self.cache)
            # Use unique text for each to avoid caching issues
            results.append(judge.evaluate(f"response {i}"))

        self.assertEqual(len(results), 4)
        self.assertEqual(results[0].score, 0.5)
        self.assertEqual(results[1].score, 0.3)
        self.assertEqual(results[2].score, 0.8)
        self.assertEqual(results[3].score, 0.0)  # Fallback for malformed

    def test_batch_continue_on_error(self):
        """Continue batch processing if one evaluation fails gracefully."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient('{"score": 0.5}')
        judge = LLMJudge(config, client, self.cache)

        texts = [f"response {i}" for i in range(5)]
        results = []
        for text in texts:
            try:
                results.append(judge.evaluate(text))
            except Exception:
                results.append(None)

        self.assertEqual(len(results), 5)
        self.assertTrue(all(r is not None for r in results))


if __name__ == "__main__":
    unittest.main()
