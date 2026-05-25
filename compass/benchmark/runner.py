"""Core benchmark execution helpers."""

import logging
from pathlib import Path

from compass import (
    CheckpointManager,
    EvaluationCache,
    JudgeConfig,
    LLMJudge,
    OllamaClient,
)
from compass.benchmark.config import (
    LEGACY_TOKEN_CAP_FALLBACK,
    default_max_tokens_for_model,
)
from compass.benchmark.io import load_generation_records
from compass.benchmark.schemas import (
    migrate_evaluation_record,
    migrate_generation_record,
)
from compass.benchmark.specs import BenchmarkSpec
from compass.clients import AnthropicClient, GoogleAIClient, OpenAIClient

logger = logging.getLogger(__name__)

MIN_VISIBLE_CHARS = 80
_SENTENCE_ENDINGS = (".", "!", "?", "\"", "'")
_TOKEN_CAP_FINISH_REASONS = {"length", "max_tokens", "max_output_tokens", "token_limit"}


def _compute_generation_quality(
    completion: str,
    output_tokens: int,
    max_tokens_requested: int,
    finish_reason: str = "",
    token_cap_inferred_legacy: bool = False,
) -> dict:
    text = (completion or "").strip()
    visible_chars = len(text)
    visible_word_count = len(text.split()) if text else 0
    finish_reason_norm = (finish_reason or "").strip().lower()
    hit_token_cap = bool(
        (max_tokens_requested > 0 and output_tokens >= max_tokens_requested)
        or (finish_reason_norm in _TOKEN_CAP_FINISH_REASONS)
    )
    sentence_complete = bool(text and text.endswith(_SENTENCE_ENDINGS))
    is_fragment = (
        not text
        or (visible_chars < MIN_VISIBLE_CHARS and not sentence_complete)
        or (hit_token_cap and not sentence_complete)
    )
    quality_flagged = bool(hit_token_cap or is_fragment)
    return {
        "visible_chars": visible_chars,
        "visible_word_count": visible_word_count,
        "hit_token_cap": hit_token_cap,
        "is_fragment": is_fragment,
        "quality_flagged": quality_flagged,
        "finish_reason": finish_reason,
        "token_cap_inferred_legacy": token_cap_inferred_legacy,
    }


def _generation_quality_from_record(record: dict) -> dict:
    output_tokens = 0
    tokens_used = record.get("tokens_used")
    if isinstance(tokens_used, dict):
        output_tokens = int(tokens_used.get("output", 0) or 0)
    raw_max_tokens = record.get("max_tokens_requested")
    max_tokens_requested = int(raw_max_tokens or 0)
    token_cap_inferred_legacy = False
    if not max_tokens_requested and output_tokens >= LEGACY_TOKEN_CAP_FALLBACK:
        max_tokens_requested = LEGACY_TOKEN_CAP_FALLBACK
        token_cap_inferred_legacy = True
    finish_reason = str(record.get("finish_reason") or "")
    return _compute_generation_quality(
        completion=record.get("completion", ""),
        output_tokens=output_tokens,
        max_tokens_requested=max_tokens_requested,
        finish_reason=finish_reason,
        token_cap_inferred_legacy=token_cap_inferred_legacy,
    )


def _default_max_tokens_for_model(model: str) -> int:
    """Backward-compatible wrapper around benchmark token budget config."""
    return default_max_tokens_for_model(model)


def compute_token_budget_by_model(models: list) -> dict:
    """Return effective max token budget by model."""
    return {model: _default_max_tokens_for_model(model) for model in models}


def validate_token_budget_policy(
    models: list, allow_mixed: bool, max_tokens_by_model: dict = None
) -> dict:
    """Validate token-budget fairness policy and return effective budgets."""
    if max_tokens_by_model is None:
        budgets = compute_token_budget_by_model(models)
    else:
        budgets = {}
        for model in models:
            if model not in max_tokens_by_model:
                raise ValueError(
                    f"Missing max token budget for model {model!r} in custom map."
                )
            budget = int(max_tokens_by_model[model])
            if budget <= 0:
                raise ValueError(
                    f"Invalid max token budget for model {model!r}: {budget}"
                )
            budgets[model] = budget

    distinct = sorted(set(budgets.values()))
    if len(distinct) > 1 and not allow_mixed:
        raise ValueError(
            "Mixed max token budgets detected across models: "
            f"{budgets}. Re-run with --allow-mixed-token-budgets to override."
        )
    return budgets


def setup_output_dir(output_dir: str) -> Path:
    """Create output directory."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _create_client(model: str):
    if model.startswith("gemini"):
        return GoogleAIClient(model=model)
    if model.startswith("gpt"):
        return OpenAIClient(model=model)
    if model.startswith("claude"):
        return AnthropicClient(model=model)
    return OllamaClient(model=model)


def test_model_connection(model: str) -> bool:
    """Test if model is available (Ollama or cloud)."""
    try:
        client = _create_client(model)
        response = client.complete("test", max_tokens=10)
        logger.info("✓ %s available (test: %s)", model, response.tokens_used)
        return True
    except Exception as e:
        logger.error("✗ %s unavailable: %s", model, e)
        return False


def generate_completions(
    models: list,
    benchmark_spec: BenchmarkSpec,
    samples: int,
    output_dir: Path,
    max_tokens_by_model: dict = None,
    allow_mixed_token_budgets: bool = False,
) -> Path:
    """Generate benchmark completions."""
    checkpoint_path = output_dir / "generations.jsonl"
    checkpoint = CheckpointManager(str(checkpoint_path))

    completed = checkpoint.load()
    logger.info("Resuming: %d prior generations", len(completed))

    total_to_generate = 0
    for rubric, prompts in benchmark_spec.prompts_by_rubric.items():
        for prompt in prompts:
            for model in models:
                for sample_idx in range(samples):
                    identity = (model, rubric, prompt.id, sample_idx)
                    if identity not in completed:
                        total_to_generate += 1

    if total_to_generate == 0:
        logger.info("All generations already complete")
        return checkpoint_path

    logger.info("Generating %d completions...", total_to_generate)
    max_tokens_by_model = validate_token_budget_policy(
        models,
        allow_mixed=allow_mixed_token_budgets,
        max_tokens_by_model=max_tokens_by_model,
    )
    clients_by_model = {model: _create_client(model) for model in models}

    count = 0
    for rubric, prompts in benchmark_spec.prompts_by_rubric.items():
        for prompt in prompts:
            for model in models:
                client = clients_by_model[model]

                for sample_idx in range(samples):
                    identity = (model, rubric, prompt.id, sample_idx)
                    if identity in completed:
                        continue

                    try:
                        max_tokens = int(max_tokens_by_model[model])
                        response = client.complete(
                            prompt=prompt.text,
                            max_tokens=max_tokens,
                            temperature=0.7,
                        )
                        quality = _compute_generation_quality(
                            completion=response.completion,
                            output_tokens=int(response.tokens_used.get("output", 0)),
                            max_tokens_requested=max_tokens,
                            finish_reason=str(
                                getattr(response, "finish_reason", "") or ""
                            ),
                        )

                        checkpoint.save(
                            migrate_generation_record(
                                {
                                    "model": model,
                                    "rubric": rubric,
                                    "prompt_id": prompt.id,
                                    "prompt_text": prompt.text,
                                    "task_type": prompt.task_type,
                                    "sample_idx": sample_idx,
                                    "completion": response.completion,
                                    "tokens_used": response.tokens_used,
                                    "cost_usd": response.cost_usd,
                                    "max_tokens_requested": max_tokens,
                                    "finish_reason": str(
                                        getattr(response, "finish_reason", "") or ""
                                    ),
                                    "visible_chars": quality["visible_chars"],
                                    "visible_word_count": quality["visible_word_count"],
                                    "hit_token_cap": quality["hit_token_cap"],
                                    "is_fragment": quality["is_fragment"],
                                    "quality_flagged": quality["quality_flagged"],
                                }
                            )
                        )

                        count += 1
                        if count % 10 == 0:
                            logger.info("  Generated %d/%d", count, total_to_generate)

                    except Exception as e:
                        logger.error(
                            "  Failed to generate %s/%s/%s: %s",
                            model,
                            rubric,
                            prompt.id,
                            e,
                        )

    logger.info("✓ Generated %d completions", count)
    return checkpoint_path


def evaluate_completions(
    generations_path: Path,
    benchmark_spec: BenchmarkSpec,
    judge_model: str,
    output_dir: Path,
) -> Path:
    """Evaluate benchmark completions with a judge model."""
    checkpoint_path = output_dir / f"evaluations_{judge_model}.jsonl"
    checkpoint = CheckpointManager(str(checkpoint_path))

    completed = checkpoint.load()
    logger.info("Resuming: %d prior evaluations", len(completed))

    generations_by_key = load_generation_records(generations_path)
    cache = EvaluationCache(cache_dir=str(output_dir / ".cache"))

    total_to_evaluate = 0
    for (model, rubric, prompt_id, sample_idx), _gen in generations_by_key.items():
        identity = (model, rubric, prompt_id, sample_idx)
        if identity not in completed:
            total_to_evaluate += 1

    if total_to_evaluate == 0:
        logger.info("All evaluations already complete")
        return checkpoint_path

    logger.info("Evaluating %d completions with %s...", total_to_evaluate, judge_model)
    judge_client = _create_client(judge_model)
    judges = {
        rubric_name: LLMJudge(
            JudgeConfig(rubric=rubric, judge_model=judge_model),
            client=judge_client,
            cache=cache,
        )
        for rubric_name, rubric in benchmark_spec.rubrics_by_name.items()
    }

    count = 0
    for (model, rubric, prompt_id, sample_idx), gen in generations_by_key.items():
        identity = (model, rubric, prompt_id, sample_idx)
        if identity in completed:
            continue

        try:
            judge = judges[rubric]
            result = judge.evaluate(gen["completion"])
            quality = _generation_quality_from_record(gen)

            checkpoint.save(
                migrate_evaluation_record(
                    {
                        "model": model,
                        "rubric": rubric,
                        "prompt_id": prompt_id,
                        "task_type": gen["task_type"],
                        "sample_idx": sample_idx,
                        "completion": gen["completion"][:200],
                        "score": result.score,
                        "hit": result.hit,
                        "confidence": result.confidence,
                        "rationale": result.rationale[:100] if result.rationale else "",
                        "judge_model": judge_model,
                        "generation_visible_chars": quality["visible_chars"],
                        "generation_visible_word_count": quality["visible_word_count"],
                        "generation_hit_token_cap": quality["hit_token_cap"],
                        "generation_is_fragment": quality["is_fragment"],
                        "generation_quality_flagged": quality["quality_flagged"],
                        "generation_finish_reason": quality["finish_reason"],
                        "generation_token_cap_inferred_legacy": quality[
                            "token_cap_inferred_legacy"
                        ],
                    }
                )
            )

            count += 1
            if count % 10 == 0:
                logger.info("  Evaluated %d/%d", count, total_to_evaluate)

        except Exception as e:
            logger.error(
                "  Failed to evaluate %s/%s/%s: %s",
                model,
                rubric,
                prompt_id,
                e,
            )

    logger.info("✓ Evaluated %d completions", count)
    return checkpoint_path
