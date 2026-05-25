#!/usr/bin/env python3
"""Constitutional compliance benchmark CLI."""

import argparse
import logging
import sys
from pathlib import Path

from compass.benchmark import (
    DEFAULT_TOKEN_BUDGETS,
    LEGACY_TOKEN_CAP_FALLBACK,
    _compute_generation_quality,
    _default_max_tokens_for_model,
    _generation_quality_from_record,
    analyze_results,
    compute_token_budget_by_model,
    default_max_tokens_for_model,
    evaluate_completions as _evaluate_completions,
    generate_completions as _generate_completions,
    load_evaluation_records,
    load_generation_records,
    print_summary,
    rank_models as _rank_models,
    setup_output_dir,
    test_model_connection,
    validate_token_budget_policy,
)
from compass.benchmark.registry import CONSTITUTIONAL_COMPLIANCE_BENCHMARK
from compass.benchmark.specs import build_benchmark_spec
from compass.rubrics.library import RubricLibrary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BENCHMARK_SPEC = CONSTITUTIONAL_COMPLIANCE_BENCHMARK
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
    allow_mixed_token_budgets: bool = False,
) -> Path:
    """Backward-compatible wrapper around shared benchmark generation."""
    return _generate_completions(
        models=models,
        benchmark_spec=_coerce_spec(prompts_by_rubric),
        samples=samples,
        output_dir=output_dir,
        max_tokens_by_model=max_tokens_by_model,
        allow_mixed_token_budgets=allow_mixed_token_budgets,
    )


def evaluate_completions(
    generations_path: Path,
    judge_model: str,
    output_dir: Path,
    legacy_token_cap_threshold: int = LEGACY_TOKEN_CAP_FALLBACK,
) -> Path:
    """Backward-compatible wrapper around shared benchmark evaluation."""
    return _evaluate_completions(
        generations_path=generations_path,
        benchmark_spec=BENCHMARK_SPEC,
        judge_model=judge_model,
        output_dir=output_dir,
        legacy_token_cap_threshold=legacy_token_cap_threshold,
    )


def rank_models(evaluations_path: Path, output_dir: Path) -> None:
    """Backward-compatible wrapper around shared pairwise ranking."""
    _rank_models(evaluations_path, BENCHMARK_SPEC, output_dir)


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
        "--models",
        nargs="+",
        default=["llama3.1", "mistral", "phi"],
        help="Models to evaluate. Local: llama3.1, mistral, phi (Ollama). Cloud: gemini-1.5-flash, gemini-1.5-pro (Google AI free tier)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of samples per prompt (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/constitutional_compliance_benchmark",
        help="Output directory for results (default: results/constitutional_compliance_benchmark)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="llama3.1",
        help="Judge model for evaluation (default: llama3.1). Use 'llama3.1', 'mistral', 'phi' for local, or 'gpt-4o-mini', 'claude-haiku-4-5' for cloud",
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
        action="store_true",
        help=(
            "Allow different effective max token budgets across models during generation. "
            "By default, mixed budgets fail preflight to preserve fairness."
        ),
    )
    parser.add_argument(
        "--legacy-token-cap-threshold",
        type=int,
        default=LEGACY_TOKEN_CAP_FALLBACK,
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

    logger.info("=" * 100)
    logger.info("CONSTITUTIONAL COMPLIANCE BENCHMARK")
    logger.info("=" * 100)
    has_local = any(not m.startswith(("gemini", "gpt", "claude")) for m in args.models)
    has_cloud = any(m.startswith(("gemini", "gpt", "claude")) for m in args.models)
    if has_local and not has_cloud:
        gen_source = "LOCAL (Ollama)"
    elif has_cloud and not has_local:
        gen_source = "CLOUD (OpenAI/Anthropic/Google)"
    else:
        gen_source = "MIXED (Ollama + Cloud)"
    logger.info("Generation: %s (%s)", ", ".join(args.models), gen_source)
    if any(m.startswith("gemini") for m in args.models):
        logger.info("  (Gemini free tier - may be slow due to rate limits)")
    judge_source = (
        "LOCAL (free)"
        if args.judge_model in ("llama3.1", "mistral", "phi", "neural-chat", "dolphin-mixtral")
        else ("CLOUD (free tier)" if args.judge_model.startswith("gemini") else "CLOUD (paid)")
    )
    logger.info("Judge: %s (%s)", args.judge_model, judge_source)
    logger.info("Rubrics: %s", ", ".join(BENCHMARK_SPEC.rubric_names))
    logger.info("Samples per prompt: %d", args.samples)
    logger.info("Output directory: %s", args.output_dir)
    logger.info("Default token budget config: %s", dict(DEFAULT_TOKEN_BUDGETS))
    logger.info("Legacy token-cap threshold: %d", args.legacy_token_cap_threshold)

    total_evals = BENCHMARK_SPEC.total_evaluations(len(args.models), args.samples)
    if args.judge_model in ("llama3.1", "mistral", "phi", "neural-chat", "dolphin-mixtral"):
        judge_cost = 0.0
        cost_note = "FULLY FREE (local generation + local judge)"
    else:
        judge_cost = total_evals * 0.001
        cost_note = f"${judge_cost:.2f} (judge only, generation is free)"
    logger.info("Total evaluations: %d", total_evals)
    logger.info("Cost: %s", cost_note)
    logger.info("")

    output_dir = setup_output_dir(args.output_dir)
    if args.legacy_token_cap_threshold <= 0:
        logger.error("--legacy-token-cap-threshold must be > 0")
        sys.exit(2)

    logger.info("Checking model availability...")
    available_models = [m for m in args.models if test_model_connection(m)]
    if not available_models:
        logger.error("No models available. Check Ollama is running or API keys are set.")
        sys.exit(1)
    logger.info("")

    if not args.skip_generation:
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
            token_budget_by_model = validate_token_budget_policy(
                available_models,
                allow_mixed=args.allow_mixed_token_budgets,
                max_tokens_by_model=token_budget_input,
            )
        except ValueError as e:
            logger.error(str(e))
            sys.exit(2)

        distinct_budgets = sorted(set(token_budget_by_model.values()))
        if len(distinct_budgets) == 1:
            logger.info(
                "Token budget policy: uniform max_tokens=%d across %d models.",
                distinct_budgets[0],
                len(token_budget_by_model),
            )
        else:
            logger.warning(
                "Token budget policy: mixed budgets enabled via override: %s",
                token_budget_by_model,
            )
        logger.info("")

        logger.info("PHASE 1: Generating completions...")
        generations_path = generate_completions(
            available_models,
            PROMPTS,
            args.samples,
            output_dir,
            max_tokens_by_model=token_budget_by_model,
            allow_mixed_token_budgets=args.allow_mixed_token_budgets,
        )
    else:
        generations_path = output_dir / "generations.jsonl"
        if not generations_path.exists():
            logger.error("Generation checkpoint not found: %s", generations_path)
            sys.exit(1)

    logger.info("")
    logger.info("PHASE 2: Evaluating completions...")
    evaluations_path = evaluate_completions(
        generations_path,
        args.judge_model,
        output_dir,
        legacy_token_cap_threshold=args.legacy_token_cap_threshold,
    )

    logger.info("")
    logger.info("PHASE 3: Analyzing results...")
    stats = analyze_results(evaluations_path, output_dir)
    print_summary(stats, evaluations_path)

    if not args.skip_ranking:
        logger.info("PHASE 4: Pairwise model comparison...")
        rank_models(evaluations_path, output_dir)

    logger.info("✓ Benchmark complete. Results in %s/", output_dir)


if __name__ == "__main__":
    main()
