#!/usr/bin/env python3
"""Safety benchmark suite CLI.

Runs any of the four safety benchmarks:
  self_harm_refusal            — model refuses and redirects self-harm requests
  refusal_quality              — model does not over-refuse benign requests
  adversarial_instruction_following — model resists jailbreaks and prompt injection
  demographic_consistency      — model treats demographic groups consistently
"""

import argparse
import logging
from pathlib import Path

from compass.benchmark import (
    get_benchmark_runner,
    get_benchmark_spec,
    log_and_exit,
    log_benchmark_run_summary,
    log_token_budget_policy,
    print_summary,
    require_available_models,
    require_or_exit,
    resolve_token_budget_overrides,
    run_or_exit,
    setup_output_dir,
    test_model_connection,
    validate_token_budget_policy,
    parse_max_tokens_by_model_args,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SAFETY_BENCHMARKS = (
    "self_harm_refusal",
    "refusal_quality",
    "adversarial_instruction_following",
    "demographic_consistency",
)


def create_parser():
    parser = argparse.ArgumentParser(
        description="Run safety benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Benchmarks:
  self_harm_refusal              5 prompts — explicit requests, ideation, validation-seeking,
                                 indirect framing, benign control
  refusal_quality                5 prompts — fiction, history, security education, creative
                                 dark themes, medical info (tests over-refusal)
  adversarial_instruction_following
                                 5 prompts — DAN jailbreak, prompt injection, fiction bypass,
                                 authority impersonation, benign control
  demographic_consistency        8 prompts — 4 paired axes: gender, race, religion, nationality

Examples:
  # Run self-harm refusal benchmark with default models
  python examples/safety_benchmarks.py self_harm_refusal

  # Run all four benchmarks sequentially
  for b in self_harm_refusal refusal_quality adversarial_instruction_following demographic_consistency; do
    python examples/safety_benchmarks.py $b
  done

  # Cloud models with a capable judge
  python examples/safety_benchmarks.py self_harm_refusal \\
    --models claude-haiku-4-5 gpt-4o-mini \\
    --judge-model gpt-4o-mini \\
    --samples 5
        """,
    )
    parser.add_argument(
        "benchmark",
        choices=SAFETY_BENCHMARKS,
        help="Which safety benchmark to run",
    )
    parser.add_argument(
        "--preset",
        default=None,
        help="Run preset to start from (default: benchmark default)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to evaluate (default: preset models)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Samples per prompt (default: preset samples)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Judge model (default: preset judge)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: preset output_dir)",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip generation, evaluate existing checkpoint",
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
        help="Uniform max token budget for all models (overrides preset default of 1000)",
    )
    budget_group.add_argument(
        "--max-tokens-by-model",
        action="append",
        default=None,
        metavar="MODEL=TOKENS",
        help="Per-model max token budget. Repeat for multiple models.",
    )
    parser.add_argument(
        "--allow-mixed-token-budgets",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--legacy-token-cap-threshold",
        type=int,
        default=None,
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    spec = get_benchmark_spec(args.benchmark)
    runner = get_benchmark_runner(args.benchmark)
    preset = spec.get_run_preset(args.preset)

    requested_models = list(args.models) if args.models else list(preset.models)
    requested_samples = args.samples if args.samples is not None else preset.samples
    requested_judge_model = args.judge_model or preset.judge_model
    requested_output_dir = args.output_dir or preset.output_dir
    requested_legacy_threshold = (
        args.legacy_token_cap_threshold
        if args.legacy_token_cap_threshold is not None
        else preset.legacy_token_cap_threshold
    )

    total_evals = spec.total_evaluations(len(requested_models), requested_samples)
    log_benchmark_run_summary(
        logger,
        benchmark_title=args.benchmark.upper().replace("_", " ") + " BENCHMARK",
        preset_name=args.preset or spec.default_preset,
        models=requested_models,
        judge_model=requested_judge_model,
        rubric_names=spec.rubric_names,
        samples=requested_samples,
        output_dir=requested_output_dir,
        token_budget_defaults=preset.policy.token_budgets,
        legacy_token_cap_threshold=requested_legacy_threshold,
        total_evaluations=total_evals,
    )

    require_or_exit(
        requested_legacy_threshold > 0,
        logger,
        "--legacy-token-cap-threshold must be > 0",
        exit_code=2,
    )
    output_dir = setup_output_dir(requested_output_dir)

    logger.info("Checking model availability...")
    available_models = require_available_models(
        requested_models,
        test_model_connection,
        logger,
        exit_code=1,
        unavailable_message="No models available. Check Ollama is running or API keys are set.",
    )
    logger.info("")

    token_budget_input = resolve_token_budget_overrides(
        available_models,
        args.max_tokens,
        args.max_tokens_by_model,
        logger,
        exit_code=2,
    )

    def _build_run_config():
        run_config = spec.make_run_config(
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
        return runner.validate_run_config(run_config)

    run_config = run_or_exit(_build_run_config, logger, exit_code=2)
    logger.info("Quality filter policy: %s", run_config.quality_filter_mode)
    logger.info("Analysis lanes: %s", ", ".join(run_config.effective_analysis_lanes))

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
        log_token_budget_policy(logger, token_budget_by_model)
        logger.info("")
        logger.info("PHASE 1: Generating completions...")
        generations_path = runner.generate(run_config)
    else:
        generations_path = output_dir / "generations.jsonl"
        if not generations_path.exists():
            log_and_exit(
                logger,
                f"Generation checkpoint not found: {generations_path}",
                exit_code=1,
            )

    logger.info("")
    logger.info("PHASE 2: Evaluating completions...")
    evaluations_path = run_or_exit(
        lambda: runner.evaluate(generations_path, run_config),
        logger,
        exit_code=1,
    )

    if "summary" in run_config.effective_analysis_lanes:
        logger.info("")
        logger.info("PHASE 3: Analyzing results...")
        stats = run_or_exit(
            lambda: runner.analyze(evaluations_path, run_config),
            logger,
            exit_code=1,
        )
        print_summary(stats, evaluations_path)

    if "pairwise" in run_config.effective_analysis_lanes:
        logger.info("PHASE 4: Pairwise model comparison...")
        run_or_exit(
            lambda: runner.rank(evaluations_path, run_config),
            logger,
            exit_code=1,
        )

    logger.info("✓ Benchmark complete. Results in %s/", output_dir)


if __name__ == "__main__":
    main()
