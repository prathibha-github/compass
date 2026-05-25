"""Tests for evaluation cache."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from compass.cache import EvaluationCache
from compass.judges import EvaluationResult


class TestEvaluationCache(unittest.TestCase):
    """Test EvaluationCache."""

    def setUp(self):
        """Create temporary cache directory for each test."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cache = EvaluationCache(cache_dir=self.tmpdir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        self.tmpdir.cleanup()

    def test_cache_put_get(self):
        """Store and retrieve result from cache."""
        result = EvaluationResult(
            name="test",
            score=0.75,
            hit=True,
            rubric_hash="abc123",
            judge_model="gpt-4o",
        )
        self.cache.put("config123", "text_hash", "1.0", result)

        retrieved = self.cache.get("config123", "text_hash", "1.0")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "test")
        self.assertEqual(retrieved.score, 0.75)

    def test_cache_miss_returns_none(self):
        """Cache returns None for missing key."""
        result = self.cache.get("missing", "hash", "1.0")
        self.assertIsNone(result)

    def test_cache_key_deterministic(self):
        """Same inputs produce same cache key."""
        key1 = self.cache._cache_key("config123", "def", "1.0")
        key2 = self.cache._cache_key("config123", "def", "1.0")
        self.assertEqual(key1, key2)

    def test_cache_key_different_for_different_inputs(self):
        """Different inputs produce different keys."""
        key1 = self.cache._cache_key("config123", "def", "1.0")
        key2 = self.cache._cache_key("config123", "def", "1.1")
        self.assertNotEqual(key1, key2)

    def test_cache_persistence(self):
        """Results persist to disk."""
        result = EvaluationResult(
            name="test",
            score=0.5,
            hit=False,
            rubric_hash="hash1",
            judge_model="gpt-4o",
        )
        self.cache.put("config123", "text", "1.0", result)

        # Create new cache instance from same directory
        cache2 = EvaluationCache(cache_dir=self.tmpdir.name)
        retrieved = cache2.get("config123", "text", "1.0")

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.score, 0.5)

    def test_cache_memory_and_disk(self):
        """Cache works for both memory and disk."""
        result1 = EvaluationResult(
            name="test1",
            score=0.5,
            hit=False,
            rubric_hash="hash1",
            judge_model="gpt-4o",
        )
        result2 = EvaluationResult(
            name="test2",
            score=0.75,
            hit=True,
            rubric_hash="hash2",
            judge_model="gpt-4o",
        )

        self.cache.put("config123", "text", "1.0", result1)
        self.cache.put("config456", "text", "1.0", result2)

        # Get from memory
        r1 = self.cache.get("config123", "text", "1.0")
        self.assertIsNotNone(r1)

        # Get from disk after clearing memory
        self.cache.memory.clear()
        r1_disk = self.cache.get("config123", "text", "1.0")
        self.assertIsNotNone(r1_disk)
        self.assertEqual(r1_disk.name, "test1")

    def test_cache_stats(self):
        """Cache statistics are correct."""
        result = EvaluationResult(
            name="test",
            score=0.5,
            hit=False,
            rubric_hash="hash1",
            judge_model="gpt-4o",
        )
        self.cache.put("config123", "text", "1.0", result)

        stats = self.cache.stats()
        self.assertEqual(stats["cached_files"], 1)
        self.assertEqual(stats["memory_items"], 1)
        self.assertTrue("cache_dir" in stats)

    def test_cache_clear(self):
        """Clear removes all cached results."""
        result = EvaluationResult(
            name="test",
            score=0.5,
            hit=False,
            rubric_hash="hash1",
            judge_model="gpt-4o",
        )
        self.cache.put("config123", "text", "1.0", result)

        stats_before = self.cache.stats()
        self.assertGreater(stats_before["cached_files"], 0)

        self.cache.clear()
        stats_after = self.cache.stats()
        self.assertEqual(stats_after["cached_files"], 0)
        self.assertEqual(stats_after["memory_items"], 0)

    def test_cache_handles_corrupted_files(self):
        """Cache gracefully handles corrupted files."""
        # Manually write corrupted JSON
        cache_dir = Path(self.tmpdir.name)
        corrupt_file = cache_dir / "corrupt.json"
        corrupt_file.write_text("{invalid json}")

        # Cache should handle this gracefully
        result = self.cache.get("somekey", "hash", "1.0")
        self.assertIsNone(result)

    def test_cache_cleans_up_temp_file_on_replace_failure(self):
        """Atomic writes should not leave temp files behind on failure."""
        result = EvaluationResult(
            name="test",
            score=0.5,
            hit=False,
            rubric_hash="hash1",
            judge_model="gpt-4o",
        )

        with self.assertLogs("compass.cache.evaluation_cache", level="WARNING") as logs:
            with patch("compass.cache.evaluation_cache.os.replace", side_effect=OSError("boom")):
                self.cache.put("config123", "text", "1.0", result)

        self.assertIn("Cache write failed", logs.output[0])
        self.assertEqual(list(Path(self.tmpdir.name).glob("*.tmp")), [])
        self.assertEqual(list(Path(self.tmpdir.name).glob("*.json")), [])


if __name__ == "__main__":
    unittest.main()
