"""Tests for benchmark generation/evaluation record schemas."""

import unittest

from compass.benchmark.schemas import (
    BENCHMARK_SCHEMA_VERSION,
    EVALUATION_RECORD_TYPE,
    GENERATION_RECORD_TYPE,
    VALID_BENCHMARK_RECORD_TYPES,
    evaluation_identity,
    generation_identity,
    migrate_evaluation_record,
    migrate_generation_record,
)


class TestBenchmarkSchemas(unittest.TestCase):
    def test_generation_migration_defaults_legacy_fields(self):
        legacy = {
            "model": "llama3.1",
            "rubric": "clarity",
            "prompt_id": "p1",
            "completion": "answer",
        }
        row = migrate_generation_record(legacy)
        self.assertEqual(row["benchmark_schema_version"], BENCHMARK_SCHEMA_VERSION)
        self.assertEqual(row["benchmark_record_type"], GENERATION_RECORD_TYPE)
        self.assertEqual(row["sample_idx"], 0)

    def test_evaluation_migration_infers_legacy_type(self):
        legacy = {
            "model": "llama3.1",
            "rubric": "clarity",
            "prompt_id": "p1",
            "sample_idx": "2",
            "score": 0.25,
            "hit": False,
        }
        row = migrate_evaluation_record(legacy)
        self.assertEqual(row["benchmark_schema_version"], BENCHMARK_SCHEMA_VERSION)
        self.assertEqual(row["benchmark_record_type"], EVALUATION_RECORD_TYPE)
        self.assertEqual(row["sample_idx"], 2)

    def test_generation_identity(self):
        identity = generation_identity(
            {
                "model": "mistral",
                "rubric": "truthfulness",
                "prompt_id": "p5",
                "sample_idx": "1",
                "completion": "ok",
            }
        )
        self.assertEqual(identity, ("mistral", "truthfulness", "p5", 1))

    def test_evaluation_identity(self):
        identity = evaluation_identity(
            {
                "model": "mistral",
                "rubric": "truthfulness",
                "prompt_id": "p5",
                "sample_idx": 1,
                "score": 0.9,
                "hit": True,
            }
        )
        self.assertEqual(identity, ("mistral", "truthfulness", "p5", 1))

    def test_unsupported_schema_version_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported benchmark schema version"):
            migrate_generation_record(
                {
                    "model": "llama3.1",
                    "rubric": "clarity",
                    "prompt_id": "p1",
                    "completion": "x",
                    "benchmark_schema_version": 99,
                }
            )

    def test_record_type_set_is_immutable(self):
        self.assertIsInstance(VALID_BENCHMARK_RECORD_TYPES, frozenset)
        self.assertEqual(
            VALID_BENCHMARK_RECORD_TYPES,
            frozenset({GENERATION_RECORD_TYPE, EVALUATION_RECORD_TYPE}),
        )


if __name__ == "__main__":
    unittest.main()

