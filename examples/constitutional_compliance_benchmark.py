#!/usr/bin/env python3
"""Constitutional compliance benchmark CLI."""

import argparse
import logging
import sys
from pathlib import Path

from compass.benchmark import (
    CONSTITUTIONAL_COMPLIANCE_BENCHMARK,
    SharedBenchmarkRunner,
    _compute_generation_quality,
    _default_max_tokens_for_model,
    _generation_quality_from_record,
    analyze_results,
    default_max_tokens_for_model,
    get_benchmark_runner,
    load_evaluation_records,
    load_generation_records,
    print_summary,
    setup_output_dir,
    test_model_connection,
    validate_token_budget_policy,
)
from compass.benchmark.specs import build_benchmark_spec
from compass.rubrics.library import RubricLibrary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BENCHMARK_SPEC = CONSTITUTIONAL_COMPLIANCE_BENCHMARK
BENCHMARK_RUNNER = get_benchmark_runner(BENCHMARK_SPEC.name)
DEFAULT_PRESET = BENCHMARK_SPEC.get_run_preset()
PROMPTS = BENCHMARK_SPEC.as_prompt_dict()


def _coerce_spec(prompts_by_rubric: dict):
    if prompts_by_rubric == PROMPTS:
        return BENCHMARK_SPEC
    return build_benchmark_spec(
        name=BENCHMARK_SPEC.name,
        version=BENCHMARK_SPEC.version,
        prompts_by_rubric=prompts_by_rubric,
        rubrics_by_name={
            rubric_name: RubricLibrary.get(rubric_name)
            for rubric_name in prompts_by_rubric
        },
        pairwise_segment_field=BENCHMARK_SPEC.pairwise_segment_field,
    )


def _runner_for_spec(benchmark_spec):
    if benchmark_spec == BENCHMARK_SPEC:
        return BENCHMARK_RUNNER
    return SharedBenchmarkRunner(benchmark_spec)


def _parse_max_tokens_by_model_args(items):
    """Parse CLI entries like model=1000 into a token-budget map."""
    budgets = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(
                f"Invalid --max-tokens-by-model entry {item!r}; expected MODEL=TOKENS."
            )
        model, raw_budget = item.split("=", 1)
        model = model.strip()
        if not model:
            raise ValueError("Model name cannot be empty in --max-tokens-by-model.")
        try:
            budget = int(raw_budget)
        except ValueError as exc:
            raise ValueError(
                f"Invalid token budget {raw_budget!r} for model {model!r}."
            ) from exc
        if budget <= 0:
            raise ValueError(
                f"Invalid token budget {budget!r} for model {model!r}; must be > 0."
            )
        budgets[model] = budget
    return budgets


def generate_completions(
    models: list,
    prompts_by_rubric: dict,
    samples: int,
    output_dir: Path,
    max_tokens_by_model: dict = None,
    allow_mixed_token_budgets: bool = None,
) -> Path:
    """Backward-compatible wrapper around shared benchmark generation."""
    benchmark_spec = _coerce_spec(prompts_by_rubric)
    runner = _runner_for_spec(benchmark_spec)
    run_config = benchmark_spec.make_run_config(
        models=models,
        samples=samples,
        output_dir=str(output_dir),
        max_tokens_by_model=max_tokens_by_model,
        allow_mixed_token_budgets=allow_mixed_token_budgets,
    )
    return runner.generate(run_config)


def evaluate_completions(
    generations_path: Path,
    judge_model: str,
    output_dir: Path,
    legacy_token_cap_threshold: int = DEFAULT_PRESET.legacy_token_cap_threshold,
) -> Path:
    """Backward-compatible wrapper around shared benchmark evaluation."""
    run_config = BENCHMARK_SPEC.make_run_config(
        judge_model=judge_model,
        output_dir=str(output_dir),
        legacy_token_cap_threshold=legacy_token_cap_threshold,
    )
    return BENCHMARK_RUNNER.evaluate(generations_path, run_config)


def rank_models(evaluations_path: Path, output_dir: Path) -> None:
    """Backward-compatible wrapper around shared pairwise ranking."""
    run_config = BENCHMARK_SPEC.make_run_config(output_dir=str(output_dir))
    BENCHMARK_RUNNER.rank(evaluations_path, run_config)


def create_parser():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run local model benchmark on all Constitutional principles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Evaluates: task_focus, truthfulness, sycophancy, therapy_speak, clarity

Examples:
  # Full benchmark (3 models × 25 prompts × 5 samples = 375 evals, ~$0.40)
  python examples/local_benchmark.py --samples 5

  # Quick test (3 models × 25 prompts × 2 samples = 150 evals, ~$0.15)
  python examples/local_benchmark.py --samples 2

  # Subset of models
  python examples/local_benchmark.py --models llama3.1 mistral --samples 3

  # Custom output directory
  python examples/local_benchmark.py --output-dir results/constitutional

  # Uniform generous token budget for all models
  python examples/constitutional_compliance_benchmark.py --samples 5 --max-tokens 1000

  # Explicit per-model budgets
  python examples/constitutional_compliance_benchmark.py --samples 5 \
    --max-tokens-by-model claude-haiku-4-5=1000 \
    --max-tokens-by-model gpt-5.4-mini=1000
        """,
    )
    parser.add_argument(
        "--preset",
        choices=BENCHMARK_SPEC.preset_names,
        default=BENCHMARK_SPEC.default_preset,
        help="Benchmark-owned run preset to start from (default: %(default)s)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to evaluate. Local: llama3.1, mistral, phi (Ollama). Cloud: gemini-1.5-flash, gemini-1.5-pro (Google AI free tier)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples per prompt. Defaults to the selected preset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results. Defaults to the selected preset.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Judge model for evaluation. Defaults to the selected preset.",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip generation, only evaluate existing checkpoint",
    )
    parser.add_argument(
        "--skip-ranking",
        action="store_true",
        help="Skip pairwise ranking analysis",
    )
    budget_group = parser.add_mutually_exclusive_group()
    budget_group.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "Use the same max token budget for every evaluated model. "
            "Overrides benchmark defaults."
        ),
    )
    budget_group.add_argument(
        "--max-tokens-by-model",
        action="append",
        default=None,
        metavar="MODEL=TOKENS",
        help=(
            "Explicit per-model max token budget. Repeat for multiple models. "
            "Requires --allow-mixed-token-budgets when budgets differ."
        ),
    )
    parser.add_argument(
        "--allow-mixed-token-budgets",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Allow different effective max token budgets across models during generation. "
            "Overrides the selected preset fairness policy."
        ),
    )
    parser.add_argument(
        "--legacy-token-cap-threshold",
        type=int,
        default=None,
        help=(
            "Threshold used to infer token-cap hits when analyzing legacy "
            "generation rows that do not store max_tokens_requested."
        ),
    )
    return parser


def main():
    """Run the benchmark."""
    parser = create_parser()
    args = parser.parse_args()
    preset = BENCHMARK_SPEC.get_run_preset(args.preset)
    requested_models = list(args.models) if args.models else list(preset.models)
    requested_samples = args.samples if args.samples is not None else preset.samples
    requested_judge_model = args.judge_model or preset.judge_model
    requested_output_dir = args.output_dir or preset.output_dir
    requested_legacy_threshold = (
        args.legacy_token_cap_threshold
        if args.legacy_token_cap_threshold is not None
        else preset.legacy_token_cap_threshold
    )

    logger.info("=" * 100)
    logger.info("CONSTITUTIONAL COMPLIANCE BENCHMARK")
    logger.info("=" * 100)
    has_local = any(not m.startswith(("gemini", "gpt", "claude")) for m in requested_models)
    has_cloud = any(m.startswith(("gemini", "gpt", "claude")) for m in requested_models)
    if has_local and not has_cloud:
        gen_source = "LOCAL (Ollama)"
    elif has_cloud and not has_local:
        gen_source = "CLOUD (OpenAI/Anthropic/Google)"
    else:
        gen_source = "MIXED (Ollama + Cloud)"
    logger.info("Run preset: %s", args.preset)
    logger.info("Generation: %s (%s)", ", ".join(requested_models), gen_source)
    if any(m.startswith("gemini") for m in requested_models):
        logger.info("  (Gemini free tier - may be slow due to rate limits)")
    judge_source = (
        "LOCAL (free)"
        if requested_judge_model
        in ("llama3.1", "mistral", "phi", "neural-chat", "dolphin-mixtral")
        else (
            "CLOUD (free tier)"
            if requested_judge_model.startswith("gemini")
            else "CLOUD (paid)"
        )
    )
    logger.info("Judge: %s (%s)", requested_judge_model, judge_source)
    logger.info("Rubrics: %s", ", ".join(BENCHMARK_SPEC.rubric_names))
    logger.info("Samples per prompt: %d", requested_samples)
    logger.info("Output directory: %s", requested_output_dir)
    logger.info("Default token budget config: %s", dict(preset.policy.token_budgets))
    logger.info("Quality filter policy: %s", preset.policy.quality_filter_mode)
    logger.info("Analysis lanes: %s", ", ".join(preset.policy.analysis_lanes))
    logger.info("Legacy token-cap threshold: %d", requested_legacy_threshold)

    total_evals = BENCHMARK_SPEC.total_evaluations(
        len(requested_models),
        requested_samples,
    )
    if requested_judge_model in (
        "llama3.1",
        "mistral",
        "phi",
        "neural-chat",
        "dolphin-mixtral",
    ):
        judge_cost = 0.0
        cost_note = "FULLY FREE (local generation + local judge)"
    else:
        judge_cost = total_evals * 0.001
        cost_note = f"${judge_cost:.2f} (judge only, generation is free)"
    logger.info("Total evaluations: %d", total_evals)
    logger.info("Cost: %s", cost_note)
    logger.info("")

    if requested_legacy_threshold <= 0:
        logger.error("--legacy-token-cap-threshold must be > 0")
        sys.exit(2)
    output_dir = setup_output_dir(requested_output_dir)

    logger.info("Checking model availability...")
    available_models = [m for m in requested_models if test_model_connection(m)]
    if not available_models:
        logger.error("No models available. Check Ollama is running or API keys are set.")
        sys.exit(1)
    logger.info("")

    if args.max_tokens is not None and args.max_tokens <= 0:
        logger.error("--max-tokens must be > 0")
        sys.exit(2)
    try:
        custom_budget_by_model = _parse_max_tokens_by_model_args(
            args.max_tokens_by_model
        )
    except ValueError as e:
        logger.error(str(e))
        sys.exit(2)

    if args.max_tokens is not None:
        token_budget_input = {model: args.max_tokens for model in available_models}
    elif custom_budget_by_model:
        token_budget_input = custom_budget_by_model
    else:
        token_budget_input = None

    try:
        run_config = BENCHMARK_SPEC.make_run_config(
            preset_name=args.preset,
            models=available_models,
            samples=requested_samples,
            judge_model=requested_judge_model,
            output_dir=str(output_dir),
            max_tokens_by_model=token_budget_input,
            allow_mixed_token_budgets=args.allow_mixed_token_budgets,
            legacy_token_cap_threshold=requested_legacy_threshold,
            skip_generation=args.skip_generation,
            skip_ranking=args.skip_ranking,
        )
        BENCHMARK_RUNNER.validate_run_config(run_config)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(2)

    if not args.skip_generation:
        token_budget_by_model = validate_token_budget_policy(
            list(run_config.models),
            allow_mixed=run_config.allow_mixed_token_budgets,
            max_tokens_by_model=(
                dict(run_config.max_tokens_by_model)
                if run_config.max_tokens_by_model is not None
                else None
            ),
            token_budget_defaults=run_config.token_budget_defaults,
        )
        distinct_budgets = sorted(set(token_budget_by_model.values()))
        if len(distinct_budgets) == 1:
            logger.info(
                "Token budget policy: uniform max_tokens=%d across %d models.",
                distinct_budgets[0],
                len(token_budget_by_model),
            )
        else:
            logger.warning(
                "Token budget policy: mixed budgets enabled: %s",
                token_budget_by_model,
            )
        logger.info("")
        logger.info("PHASE 1: Generating completions...")
        generations_path = BENCHMARK_RUNNER.generate(run_config)
    else:
        generations_path = output_dir / "generations.jsonl"
        if not generations_path.exists():
            logger.error("Generation checkpoint not found: %s", generations_path)
            sys.exit(1)

    logger.info("")
    logger.info("PHASE 2: Evaluating completions...")
    evaluations_path = BENCHMARK_RUNNER.evaluate(
        generations_path,
        run_config,
    )

    logger.info("")
    logger.info("PHASE 3: Analyzing results...")
    stats = BENCHMARK_RUNNER.analyze(evaluations_path, run_config)
    print_summary(stats, evaluations_path)

    if "pairwise" in run_config.effective_analysis_lanes:
        logger.info("PHASE 4: Pairwise model comparison...")
        BENCHMARK_RUNNER.rank(evaluations_path, run_config)

    logger.info("✓ Benchmark complete. Results in %s/", output_dir)


if __name__ == "__main__":
    main()
