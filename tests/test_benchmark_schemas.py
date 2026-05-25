"""Tests for benchmark generation/evaluation record schemas."""

import unittest

from compass.benchmark.schemas import (
    BENCHMARK_NAME_FIELD,
    BENCHMARK_SCHEMA_VERSION,
    BENCHMARK_SCHEMA_VERSION_FIELD,
    BENCHMARK_RECORD_TYPE_FIELD,
    BENCHMARK_VERSION_FIELD,
    EVALUATION_RECORD_TYPE,
    GENERATION_RECORD_TYPE,
    VALID_BENCHMARK_RECORD_TYPES,
    evaluation_identity,
    generation_identity,
    migrate_evaluation_record,
    migrate_generation_record,
)
from compass.evaluation.record_schema import (
    CHECKPOINT_RECORD_TYPE_FIELD,
    CHECKPOINT_SCHEMA_VERSION_FIELD,
    VALID_RECORD_TYPES,
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
        self.assertNotIn(BENCHMARK_NAME_FIELD, row)
        self.assertNotIn(BENCHMARK_VERSION_FIELD, row)

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
        self.assertNotIn(BENCHMARK_NAME_FIELD, row)
        self.assertNotIn(BENCHMARK_VERSION_FIELD, row)

    def test_generation_migration_preserves_benchmark_identity(self):
        row = migrate_generation_record(
            {
                "benchmark_name": "constitutional_compliance",
                "benchmark_version": "1.0",
                "model": "llama3.1",
                "rubric": "clarity",
                "prompt_id": "p1",
                "completion": "answer",
            }
        )
        self.assertEqual(row[BENCHMARK_NAME_FIELD], "constitutional_compliance")
        self.assertEqual(row[BENCHMARK_VERSION_FIELD], "1.0")

    def test_benchmark_identity_requires_name_and_version_together(self):
        with self.assertRaisesRegex(ValueError, "both benchmark_name and benchmark_version"):
            migrate_generation_record(
                {
                    "benchmark_name": "constitutional_compliance",
                    "model": "llama3.1",
                    "rubric": "clarity",
                    "prompt_id": "p1",
                    "completion": "answer",
                }
            )
        with self.assertRaisesRegex(ValueError, "both benchmark_name and benchmark_version"):
            migrate_generation_record(
                {
                    "benchmark_version": "1.0",
                    "model": "llama3.1",
                    "rubric": "clarity",
                    "prompt_id": "p1",
                    "completion": "answer",
                }
            )

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

    def test_benchmark_schema_field_names_are_distinct_from_checkpoint_fields(self):
        self.assertNotEqual(
            BENCHMARK_SCHEMA_VERSION_FIELD,
            CHECKPOINT_SCHEMA_VERSION_FIELD,
        )
        self.assertNotEqual(
            BENCHMARK_RECORD_TYPE_FIELD,
            CHECKPOINT_RECORD_TYPE_FIELD,
        )

    def test_benchmark_record_type_domain_does_not_overlap_checkpoint_domain(self):
        self.assertTrue(
            VALID_BENCHMARK_RECORD_TYPES.isdisjoint(VALID_RECORD_TYPES)
        )


if __name__ == "__main__":
    unittest.main()
