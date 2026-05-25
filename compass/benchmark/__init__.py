"""Benchmark schemas and shared helpers."""

from compass.benchmark.io import load_evaluation_records, load_generation_records
from compass.benchmark.registry import (
    CONSTITUTIONAL_COMPLIANCE_BENCHMARK,
    get_benchmark_spec,
    list_benchmark_specs,
    register_benchmark_spec,
)
from compass.benchmark.reporting import analyze_results, print_summary, rank_models
from compass.benchmark.runner import (
    _compute_generation_quality,
    _default_max_tokens_for_model,
    _generation_quality_from_record,
    compute_token_budget_by_model,
    evaluate_completions,
    generate_completions,
    setup_output_dir,
    test_model_connection,
    validate_token_budget_policy,
)
from compass.benchmark.schemas import (
    BENCHMARK_SCHEMA_VERSION,
    EVALUATION_RECORD_TYPE,
    GENERATION_RECORD_TYPE,
    evaluation_identity,
    generation_identity,
    migrate_evaluation_record,
    migrate_generation_record,
)
from compass.benchmark.specs import (
    BenchmarkPrompt,
    BenchmarkSpec,
    build_benchmark_spec,
)
from compass.benchmark.validation import (
    validate_benchmark_report,
    validate_evaluation_records_for_quality,
    validate_stats_for_quality,
)

__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "GENERATION_RECORD_TYPE",
    "EVALUATION_RECORD_TYPE",
    "BenchmarkPrompt",
    "BenchmarkSpec",
    "CONSTITUTIONAL_COMPLIANCE_BENCHMARK",
    "build_benchmark_spec",
    "get_benchmark_spec",
    "list_benchmark_specs",
    "register_benchmark_spec",
    "migrate_generation_record",
    "migrate_evaluation_record",
    "generation_identity",
    "evaluation_identity",
    "load_generation_records",
    "load_evaluation_records",
    "compute_token_budget_by_model",
    "validate_token_budget_policy",
    "setup_output_dir",
    "test_model_connection",
    "generate_completions",
    "evaluate_completions",
    "analyze_results",
    "print_summary",
    "rank_models",
    "validate_evaluation_records_for_quality",
    "validate_stats_for_quality",
    "validate_benchmark_report",
]
