"""Benchmark schemas and shared helpers."""

from compass.benchmark.config import (
    DEFAULT_TOKEN_BUDGETS,
    LEGACY_TOKEN_CAP_FALLBACK,
    default_max_tokens_for_model,
)
from compass.benchmark.io import load_evaluation_records, load_generation_records
from compass.benchmark.registry import (
    CONSTITUTIONAL_COMPLIANCE_BENCHMARK,
    CONSTITUTIONAL_COMPLIANCE_PRESET,
    CONSTITUTIONAL_COMPLIANCE_RUNNER,
    get_benchmark_runner,
    get_benchmark_spec,
    list_benchmark_specs,
    register_benchmark_spec,
)
from compass.benchmark.reporting import analyze_results, print_summary, rank_models
from compass.benchmark.runner import (
    _compute_generation_quality,
    _default_max_tokens_for_model,
    _generation_quality_from_record,
    SharedBenchmarkRunner,
    compute_token_budget_by_model,
    evaluate_completions,
    generate_completions,
    setup_output_dir,
    test_model_connection,
    validate_token_budget_policy,
)
from compass.benchmark.schemas import (
    BENCHMARK_SCHEMA_VERSION,
    BENCHMARK_SCHEMA_VERSION_FIELD,
    EVALUATION_RECORD_TYPE,
    BENCHMARK_RECORD_TYPE_FIELD,
    GENERATION_RECORD_TYPE,
    evaluation_identity,
    generation_identity,
    migrate_evaluation_record,
    migrate_generation_record,
)
from compass.benchmark.specs import (
    BenchmarkPolicyDefaults,
    BenchmarkPrompt,
    BenchmarkRunConfig,
    BenchmarkRunPreset,
    BenchmarkRunner,
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
    "BENCHMARK_SCHEMA_VERSION_FIELD",
    "BENCHMARK_RECORD_TYPE_FIELD",
    "GENERATION_RECORD_TYPE",
    "EVALUATION_RECORD_TYPE",
    "BenchmarkPolicyDefaults",
    "BenchmarkPrompt",
    "BenchmarkRunConfig",
    "BenchmarkRunPreset",
    "BenchmarkRunner",
    "BenchmarkSpec",
    "CONSTITUTIONAL_COMPLIANCE_BENCHMARK",
    "CONSTITUTIONAL_COMPLIANCE_PRESET",
    "CONSTITUTIONAL_COMPLIANCE_RUNNER",
    "DEFAULT_TOKEN_BUDGETS",
    "LEGACY_TOKEN_CAP_FALLBACK",
    "SharedBenchmarkRunner",
    "build_benchmark_spec",
    "default_max_tokens_for_model",
    "get_benchmark_runner",
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
