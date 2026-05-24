"""Benchmark schemas and shared helpers."""

from compass.benchmark.schemas import (
    BENCHMARK_SCHEMA_VERSION,
    EVALUATION_RECORD_TYPE,
    GENERATION_RECORD_TYPE,
    evaluation_identity,
    generation_identity,
    migrate_evaluation_record,
    migrate_generation_record,
)

__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "GENERATION_RECORD_TYPE",
    "EVALUATION_RECORD_TYPE",
    "migrate_generation_record",
    "migrate_evaluation_record",
    "generation_identity",
    "evaluation_identity",
]

