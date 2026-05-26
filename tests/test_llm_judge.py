"""Tests for LLMJudge."""
import tempfile
import unittest

from compass.cache import EvaluationCache
from compass.clients import CompletionClient, CompletionResponse
from compass.judges import JudgeConfig, LLMJudge
from compass.rubrics import RubricLibrary


class MockClient(CompletionClient):
    """Mock client for testing."""

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.call_count = 0
        self.last_prompt = None

    def complete(self, prompt, max_tokens=180, temperature=0.0, system=None):
        self.call_count += 1
        self.last_prompt = prompt
        return CompletionResponse(
            completion=self.response_text,
            tokens_used={"input": 100, "output": 50},
            cost_usd=0.001,
        )


class TestLLMJudge(unittest.TestCase):
    """Test LLMJudge evaluation."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cache = EvaluationCache(cache_dir=self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_judge_evaluate_well_formed_response(self):
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient(
            '{"score": 0.75, "hit": true, "confidence": 0.9, "rationale": "test"}'
        )
        judge = LLMJudge(config, client, self.cache)

        result = judge.evaluate("test response")

        self.assertEqual(result.name, "sycophancy")
        self.assertEqual(result.score, 0.75)
        self.assertTrue(result.hit)
        self.assertEqual(result.confidence, 0.9)
        self.assertEqual(result.rubric_hash, RubricLibrary.sycophancy.hash)
        self.assertEqual(result.judge_model, "gpt-4o")

    def test_judge_caches_results(self):
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient('{"score": 0.5, "hit": false, "confidence": 0.8}')
        judge = LLMJudge(config, client, self.cache)

        text = "test response"

        # First call
        result1 = judge.evaluate(text)
        self.assertFalse(result1.cache_hit)
        self.assertEqual(client.call_count, 1)

        # Second call (cached)
        result2 = judge.evaluate(text)
        self.assertTrue(result2.cache_hit)
        self.assertEqual(client.call_count, 1)  # No new call

        # Results should be identical
        self.assertEqual(result1.score, result2.score)

    def test_judge_config_change_busts_cache(self):
        base_config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
            temperature=0.0,
        )
        variant_config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
            temperature=0.7,
        )
        first_client = MockClient('{"score": 0.5, "hit": false}')
        second_client = MockClient('{"score": 0.8, "hit": true}')

        first_judge = LLMJudge(base_config, first_client, self.cache)
        second_judge = LLMJudge(variant_config, second_client, self.cache)

        first_result = first_judge.evaluate("test response")
        second_result = second_judge.evaluate("test response")

        self.assertEqual(first_client.call_count, 1)
        self.assertEqual(second_client.call_count, 1)
        self.assertFalse(first_result.cache_hit)
        self.assertFalse(second_result.cache_hit)
        self.assertNotEqual(first_result.score, second_result.score)

    def test_judge_rubric_change_busts_cache(self):
        base_config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        variant_config = JudgeConfig(
            rubric=RubricLibrary.clarity,
            judge_model="gpt-4o",
        )
        first_client = MockClient('{"score": 0.5, "hit": false}')
        second_client = MockClient('{"score": 0.8, "hit": true}')

        first_judge = LLMJudge(base_config, first_client, self.cache)
        second_judge = LLMJudge(variant_config, second_client, self.cache)

        first_result = first_judge.evaluate("test response")
        second_result = second_judge.evaluate("test response")

        self.assertEqual(first_client.call_count, 1)
        self.assertEqual(second_client.call_count, 1)
        self.assertFalse(first_result.cache_hit)
        self.assertFalse(second_result.cache_hit)
        self.assertNotEqual(first_result.rubric_hash, second_result.rubric_hash)

    def test_judge_seed_change_busts_cache(self):
        base_config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
            seed=42,
        )
        variant_config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
            seed=99,
        )
        first_client = MockClient('{"score": 0.5, "hit": false}')
        second_client = MockClient('{"score": 0.8, "hit": true}')

        first_judge = LLMJudge(base_config, first_client, self.cache)
        second_judge = LLMJudge(variant_config, second_client, self.cache)

        first_result = first_judge.evaluate("test response")
        second_result = second_judge.evaluate("test response")

        self.assertEqual(first_client.call_count, 1)
        self.assertEqual(second_client.call_count, 1)
        self.assertFalse(first_result.cache_hit)
        self.assertFalse(second_result.cache_hit)
        self.assertNotEqual(first_judge.config.config_hash, second_judge.config.config_hash)

    def test_judge_max_tokens_change_busts_cache(self):
        base_config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
            max_tokens=180,
        )
        variant_config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
            max_tokens=250,
        )
        first_client = MockClient('{"score": 0.5, "hit": false}')
        second_client = MockClient('{"score": 0.8, "hit": true}')

        first_judge = LLMJudge(base_config, first_client, self.cache)
        second_judge = LLMJudge(variant_config, second_client, self.cache)

        first_result = first_judge.evaluate("test response")
        second_result = second_judge.evaluate("test response")

        self.assertEqual(first_client.call_count, 1)
        self.assertEqual(second_client.call_count, 1)
        self.assertFalse(first_result.cache_hit)
        self.assertFalse(second_result.cache_hit)
        self.assertNotEqual(first_judge.config.config_hash, second_judge.config.config_hash)

    def test_prompt_version_change_busts_cache(self):
        class _VersionedJudge(LLMJudge):
            PROMPT_VERSION = "2.0"

        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        first_client = MockClient('{"score": 0.5, "hit": false}')
        second_client = MockClient('{"score": 0.8, "hit": true}')

        first_judge = LLMJudge(config, first_client, self.cache)
        second_judge = _VersionedJudge(config, second_client, self.cache)

        first_result = first_judge.evaluate("test response")
        second_result = second_judge.evaluate("test response")

        self.assertEqual(first_client.call_count, 1)
        self.assertEqual(second_client.call_count, 1)
        self.assertFalse(first_result.cache_hit)
        self.assertFalse(second_result.cache_hit)
        self.assertNotEqual(
            first_judge._cache_coordinates("test response"),
            second_judge._cache_coordinates("test response"),
        )

    def test_judge_different_texts_different_cache(self):
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient('{"score": 0.5, "hit": false}')
        judge = LLMJudge(config, client, self.cache)

        judge.evaluate("text 1")
        judge.evaluate("text 2")

        # Both should have called the API
        self.assertEqual(client.call_count, 2)

    def test_judge_score_clamped(self):
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )

        # Score > 1.0 should be clamped
        client = MockClient('{"score": 1.5, "hit": true}')
        judge = LLMJudge(config, client, self.cache)
        result = judge.evaluate("test")
        self.assertEqual(result.score, 1.0)

        # Score < 0.0 should be clamped
        client = MockClient('{"score": -0.5, "hit": false}')
        judge = LLMJudge(config, client, self.cache)
        result = judge.evaluate("test 2")
        self.assertEqual(result.score, 0.0)

    def test_judge_hit_defaults_to_threshold(self):
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,  # threshold is 0.5
            judge_model="gpt-4o",
        )

        # Score 0.6 >= 0.5, so hit should be true
        client = MockClient('{"score": 0.6}')
        judge = LLMJudge(config, client, self.cache)
        result = judge.evaluate("test")
        self.assertTrue(result.hit)

        # Score 0.3 < 0.5, so hit should be false
        client = MockClient('{"score": 0.3}')
        judge = LLMJudge(config, client, self.cache)
        result = judge.evaluate("test 2")
        self.assertFalse(result.hit)

    def test_judge_hit_determined_by_threshold_not_judge(self):
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,  # threshold is 0.5
            judge_model="gpt-4o",
        )

        # Judge returns hit=true but score 0.2 < threshold 0.5; threshold wins
        client = MockClient('{"score": 0.2, "hit": true}')
        judge = LLMJudge(config, client, self.cache)
        result = judge.evaluate("test")
        self.assertFalse(result.hit)

    def test_judge_malformed_response(self):
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient("Not JSON at all")
        judge = LLMJudge(config, client, self.cache)

        result = judge.evaluate("test")

        # Should return safe defaults
        self.assertEqual(result.score, 0.0)
        self.assertFalse(result.hit)
        self.assertEqual(result.rationale, "")

    def test_judge_missing_fields(self):
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )

        # Missing confidence and rationale
        client = MockClient('{"score": 0.5, "hit": false}')
        judge = LLMJudge(config, client, self.cache)
        result = judge.evaluate("test")

        self.assertIsNone(result.confidence)
        self.assertEqual(result.rationale, "")

    def test_judge_non_numeric_score(self):
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )

        # Non-numeric score should default to 0.0
        client = MockClient('{"score": "high", "hit": false}')
        judge = LLMJudge(config, client, self.cache)
        result = judge.evaluate("test")

        self.assertEqual(result.score, 0.0)

    def test_judge_tracks_tokens(self):
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient('{"score": 0.5, "hit": false}')
        judge = LLMJudge(config, client, self.cache)

        result = judge.evaluate("test")

        self.assertEqual(result.tokens_used["input"], 100)
        self.assertEqual(result.tokens_used["output"], 50)
        self.assertEqual(result.cost_usd, 0.001)
        self.assertEqual(result.prompt_version, LLMJudge.PROMPT_VERSION)

    def test_judge_deterministic_prompt(self):
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        client = MockClient('{"score": 0.5}')
        judge = LLMJudge(config, client, self.cache)

        judge.evaluate("test")
        prompt1 = client.last_prompt

        judge.evaluate("test")
        prompt2 = client.last_prompt

        # Same text should produce same prompt
        self.assertEqual(prompt1, prompt2)
        self.assertIn(RubricLibrary.sycophancy.text, prompt1)


if __name__ == "__main__":
    unittest.main()
