"""Helpers for benchmark CLI error handling."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import NoReturn, TypeVar

T = TypeVar("T")
_LOCAL_MODEL_PREFIXES = ("llama", "mistral", "phi", "neural-chat", "dolphin")
_CLOUD_MODEL_PREFIXES = ("gemini", "gpt", "claude", "o4")


def log_and_exit(
    logger: logging.Logger,
    message: str,
    *,
    exit_code: int,
) -> NoReturn:
    """Log a user-facing benchmark error and terminate."""
    logger.error(message)
    raise SystemExit(exit_code)


def log_errors_and_exit(
    logger: logging.Logger,
    messages: list[str],
    *,
    exit_code: int,
) -> NoReturn:
    """Log a sequence of user-facing benchmark errors and terminate."""
    for message in messages:
        logger.error(message)
    raise SystemExit(exit_code)


def require_or_exit(
    condition: bool,
    logger: logging.Logger,
    message: str,
    *,
    exit_code: int,
) -> None:
    """Exit with a logged error when a benchmark precondition is not met."""
    if not condition:
        log_and_exit(logger, message, exit_code=exit_code)


def run_or_exit(
    callback: Callable[[], T],
    logger: logging.Logger,
    *,
    exit_code: int,
    exception_types: tuple[type[Exception], ...] = (ValueError,),
) -> T:
    """Run a benchmark CLI step and convert expected failures into clean exits."""
    try:
        return callback()
    except exception_types as error:
        log_and_exit(logger, str(error), exit_code=exit_code)


def parse_max_tokens_by_model_args(
    items: Sequence[str] | None,
) -> dict[str, int]:
    """Parse CLI entries like model=1000 into a token-budget map."""
    budgets: dict[str, int] = {}
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


def resolve_token_budget_overrides(
    models: Sequence[str],
    max_tokens: int | None,
    max_tokens_by_model_items: Sequence[str] | None,
    logger: logging.Logger,
    *,
    exit_code: int,
) -> dict[str, int] | None:
    """Resolve benchmark token-budget CLI overrides into a shared config payload."""
    require_or_exit(
        max_tokens is None or max_tokens > 0,
        logger,
        "--max-tokens must be > 0",
        exit_code=exit_code,
    )
    custom_budget_by_model = run_or_exit(
        lambda: parse_max_tokens_by_model_args(max_tokens_by_model_items),
        logger,
        exit_code=exit_code,
    )
    if max_tokens is not None:
        return {model: max_tokens for model in models}
    return custom_budget_by_model or None


def require_available_models(
    models: Sequence[str],
    is_available: Callable[[str], bool],
    logger: logging.Logger,
    *,
    exit_code: int,
    unavailable_message: str,
) -> list[str]:
    """Filter requested models through an availability probe or exit cleanly."""
    available_models = [model for model in models if is_available(model)]
    require_or_exit(
        bool(available_models),
        logger,
        unavailable_message,
        exit_code=exit_code,
    )
    return available_models


def log_token_budget_policy(
    logger: logging.Logger,
    budgets: Mapping[str, int],
) -> None:
    """Log the resolved benchmark token-budget policy in a consistent format."""
    distinct_budgets = sorted(set(budgets.values()))
    if len(distinct_budgets) == 1:
        logger.info(
            "Token budget policy: uniform max_tokens=%d across %d models.",
            distinct_budgets[0],
            len(budgets),
        )
        return
    logger.warning(
        "Token budget policy: mixed budgets enabled: %s",
        dict(budgets),
    )


def classify_generation_source(models: Sequence[str]) -> str:
    """Return a user-facing source label for the requested generation models."""
    has_local = any(not model.startswith(_CLOUD_MODEL_PREFIXES) for model in models)
    has_cloud = any(model.startswith(_CLOUD_MODEL_PREFIXES) for model in models)
    if has_local and not has_cloud:
        return "LOCAL (Ollama)"
    if has_cloud and not has_local:
        return "CLOUD (OpenAI/Anthropic/Google)"
    return "MIXED (Ollama + Cloud)"


def classify_judge_source(judge_model: str) -> str:
    """Return a user-facing source label for the benchmark judge model."""
    if judge_model.startswith(_LOCAL_MODEL_PREFIXES):
        return "LOCAL (free)"
    if judge_model.startswith("gemini"):
        return "CLOUD (free tier)"
    return "CLOUD (paid)"


def estimate_judge_cost_note(
    total_evaluations: int,
    judge_model: str,
) -> str:
    """Return the benchmark's rough judge-cost presentation note."""
    if judge_model.startswith(_LOCAL_MODEL_PREFIXES):
        return "FULLY FREE (local generation + local judge)"
    judge_cost = total_evaluations * 0.001
    return f"${judge_cost:.2f} (judge only, generation is free)"


def log_benchmark_run_summary(
    logger: logging.Logger,
    *,
    benchmark_title: str,
    preset_name: str,
    models: Sequence[str],
    judge_model: str,
    rubric_names: Sequence[str],
    samples: int,
    output_dir: str,
    token_budget_defaults: Mapping[str, int],
    legacy_token_cap_threshold: int,
    total_evaluations: int,
) -> None:
    """Log a shared benchmark run banner and resolved high-level config."""
    logger.info("=" * 100)
    logger.info("%s", benchmark_title)
    logger.info("=" * 100)
    logger.info("Run preset: %s", preset_name)
    logger.info(
        "Generation: %s (%s)",
        ", ".join(models),
        classify_generation_source(models),
    )
    if any(model.startswith("gemini") for model in models):
        logger.info("  (Gemini free tier - may be slow due to rate limits)")
    logger.info(
        "Judge: %s (%s)",
        judge_model,
        classify_judge_source(judge_model),
    )
    logger.info("Rubrics: %s", ", ".join(rubric_names))
    logger.info("Samples per prompt: %d", samples)
    logger.info("Output directory: %s", output_dir)
    logger.info("Default token budget config: %s", dict(token_budget_defaults))
    logger.info("Legacy token-cap threshold: %d", legacy_token_cap_threshold)
    logger.info("Total evaluations: %d", total_evaluations)
    logger.info("Cost: %s", estimate_judge_cost_note(total_evaluations, judge_model))
    logger.info("")
