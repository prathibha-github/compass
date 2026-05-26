"""Core benchmark execution helpers."""

import logging
import os
from pathlib import Path
from typing import Mapping, Optional

from compass import (
    CheckpointManager,
    EvaluationCache,
    JudgeConfig,
    LLMJudge,
    OllamaClient,
)
from compass.benchmark.config import (
    DEFAULT_TOKEN_BUDGETS,
    LEGACY_TOKEN_CAP_FALLBACK,
    default_max_tokens_for_model,
)
from compass.benchmark.io import load_generation_records
from compass.benchmark.reporting import analyze_results, rank_models
from compass.benchmark.schemas import (
    migrate_evaluation_record,
    migrate_generation_record,
)
from compass.benchmark.specs import BenchmarkRunConfig, BenchmarkSpec
from compass.benchmark.validation import validate_benchmark_report
from compass.clients import AnthropicClient, GoogleAIClient, OpenAIClient

logger = logging.getLogger(__name__)

MIN_VISIBLE_CHARS = 80
_SENTENCE_ENDINGS = (".", "!", "?", "\"", "'")
_TOKEN_CAP_FINISH_REASONS = {"length", "max_tokens", "max_output_tokens", "token_limit"}
_warned_legacy_token_cap_thresholds = set()


def _reset_warned_legacy_token_cap_thresholds() -> None:
    """Test helper for resetting once-per-process legacy warning state."""
    _warned_legacy_token_cap_thresholds.clear()


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


def _generation_quality_from_record(
    record: dict,
    legacy_token_cap_threshold: int = LEGACY_TOKEN_CAP_FALLBACK,
) -> dict:
    output_tokens = 0
    tokens_used = record.get("tokens_used")
    if isinstance(tokens_used, dict):
        output_tokens = int(tokens_used.get("output", 0) or 0)
    raw_max_tokens = record.get("max_tokens_requested")
    max_tokens_requested = int(raw_max_tokens or 0)
    token_cap_inferred_legacy = False
    if not max_tokens_requested and output_tokens >= legacy_token_cap_threshold:
        max_tokens_requested = legacy_token_cap_threshold
        token_cap_inferred_legacy = True
        if legacy_token_cap_threshold not in _warned_legacy_token_cap_thresholds:
            logger.warning(
                "Inferring token cap from output_tokens >= %d because "
                "max_tokens_requested is missing in legacy generation rows. "
                "Override with --legacy-token-cap-threshold if the original run "
                "used a different budget.",
                legacy_token_cap_threshold,
            )
            _warned_legacy_token_cap_thresholds.add(legacy_token_cap_threshold)
    finish_reason = str(record.get("finish_reason") or "")
    return _compute_generation_quality(
        completion=record.get("completion", ""),
        output_tokens=output_tokens,
        max_tokens_requested=max_tokens_requested,
        finish_reason=finish_reason,
        token_cap_inferred_legacy=token_cap_inferred_legacy,
    )


def _default_max_tokens_for_model(
    model: str,
    token_budgets: Mapping[str, int] = DEFAULT_TOKEN_BUDGETS,
) -> int:
    """Backward-compatible wrapper around benchmark token budget config."""
    return default_max_tokens_for_model(model, token_budgets=token_budgets)


def compute_token_budget_by_model(
    models: list,
    token_budgets: Mapping[str, int] = DEFAULT_TOKEN_BUDGETS,
) -> dict:
    """Return effective max token budget by model."""
    return {
        model: _default_max_tokens_for_model(model, token_budgets=token_budgets)
        for model in models
    }


def validate_token_budget_policy(
    models: list,
    allow_mixed: bool,
    max_tokens_by_model: Optional[dict] = None,
    token_budget_defaults: Mapping[str, int] = DEFAULT_TOKEN_BUDGETS,
) -> dict:
    """Validate token-budget fairness policy and return effective budgets."""
    if max_tokens_by_model is None:
        budgets = compute_token_budget_by_model(
            models,
            token_budgets=token_budget_defaults,
        )
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
    if model.startswith("gpt") or model.startswith("o4"):
        required_temperature = None
        if model.startswith("gpt-5") or model.startswith("o4"):
            required_temperature = 1.0
        return OpenAIClient(
            model=model,
            required_temperature=required_temperature,
        )
    if model.startswith("claude"):
        return AnthropicClient(model=model)
    return OllamaClient(model=model)


def _require_api_key(env_var: str, provider: str) -> None:
    if not os.environ.get(env_var):
        raise RuntimeError(f"{provider} probe requires {env_var} to be set")


def _probe_ollama_model(client: OllamaClient, model: str) -> None:
    listing = client.api_client.list()
    models = listing.get("models", []) if isinstance(listing, dict) else []
    available_names = {
        entry.get("model") or entry.get("name")
        for entry in models
        if isinstance(entry, dict)
    }
    available_names.discard(None)
    requested_base = model.split(":", 1)[0]
    for candidate in available_names:
        candidate_base = candidate.split(":", 1)[0]
        if candidate == model or candidate_base == requested_base:
            return
    raise RuntimeError(f"Ollama model not installed: {model}")


def test_model_connection(model: str) -> bool:
    """Run a lightweight model readiness probe without spending generation budget."""
    try:
        client = _create_client(model)
        if model.startswith("gemini"):
            _require_api_key("GOOGLE_API_KEY", "Google AI")
            logger.info("✓ %s available (lightweight Google AI readiness check)", model)
            return True
        if model.startswith("gpt") or model.startswith("o4"):
            _require_api_key("OPENAI_API_KEY", "OpenAI")
            logger.info("✓ %s available (lightweight OpenAI readiness check)", model)
            return True
        if model.startswith("claude"):
            _require_api_key("ANTHROPIC_API_KEY", "Anthropic")
            logger.info("✓ %s available (lightweight Anthropic readiness check)", model)
            return True
        _probe_ollama_model(client, model)
        logger.info("✓ %s available (Ollama model probe)", model)
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
    token_budget_defaults: Mapping[str, int] = DEFAULT_TOKEN_BUDGETS,
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
        token_budget_defaults=token_budget_defaults,
    )
    clients_by_model = {model: _create_client(model) for model in models}

    count = 0
    failure_count = 0
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
                                "benchmark_name": benchmark_spec.name,
                                "benchmark_version": benchmark_spec.version,
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
                        failure_count += 1
                        logger.error(
                            "  Failed to generate %s/%s/%s: %s",
                            model,
                            rubric,
                            prompt.id,
                            e,
                        )

    logger.info("✓ Generated %d completions", count)
    if failure_count:
        logger.warning(
            "Generation completed with %d runtime failures; see logged errors above.",
            failure_count,
        )
    return checkpoint_path


def evaluate_completions(
    generations_path: Path,
    benchmark_spec: BenchmarkSpec,
    judge_model: str,
    output_dir: Path,
    legacy_token_cap_threshold: int = LEGACY_TOKEN_CAP_FALLBACK,
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
    failure_count = 0
    for (model, rubric, prompt_id, sample_idx), gen in generations_by_key.items():
        identity = (model, rubric, prompt_id, sample_idx)
        if identity in completed:
            continue

        try:
            judge = judges[rubric]
            result = judge.evaluate(gen["completion"])
            quality = _generation_quality_from_record(
                gen,
                legacy_token_cap_threshold=legacy_token_cap_threshold,
            )

            checkpoint.save(
                migrate_evaluation_record(
                    {
                        "benchmark_name": benchmark_spec.name,
                        "benchmark_version": benchmark_spec.version,
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
            failure_count += 1
            logger.error(
                "  Failed to evaluate %s/%s/%s: %s",
                model,
                rubric,
                prompt_id,
                e,
            )

    logger.info("✓ Evaluated %d completions", count)
    if failure_count:
        logger.warning(
            "Evaluation completed with %d runtime failures; see logged errors above.",
            failure_count,
        )
    return checkpoint_path


class SharedBenchmarkRunner:
    """Shared benchmark runner backed by the core benchmark helpers."""

    def __init__(self, spec: BenchmarkSpec):
        self.spec = spec
        self._validated_run_config_ids: set[int] = set()

    def validate_run_config(self, run_config: BenchmarkRunConfig) -> BenchmarkRunConfig:
        """Validate a resolved benchmark run config for this benchmark."""
        if run_config.benchmark_name != self.spec.name:
            raise ValueError(
                "benchmark run config name does not match runner: "
                f"{run_config.benchmark_name!r} != {self.spec.name!r}"
            )
        if run_config.benchmark_version != self.spec.version:
            raise ValueError(
                "benchmark run config version does not match runner: "
                f"{run_config.benchmark_version!r} != {self.spec.version!r}"
            )
        validate_token_budget_policy(
            list(run_config.models),
            allow_mixed=run_config.allow_mixed_token_budgets,
            max_tokens_by_model=(
                dict(run_config.max_tokens_by_model)
                if run_config.max_tokens_by_model is not None
                else None
            ),
            token_budget_defaults=run_config.token_budget_defaults,
        )
        self._validated_run_config_ids.add(id(run_config))
        return run_config

    def _require_validated_run_config(
        self,
        run_config: BenchmarkRunConfig,
    ) -> BenchmarkRunConfig:
        if id(run_config) not in self._validated_run_config_ids:
            raise ValueError(
                "benchmark run config must be validated with "
                "runner.validate_run_config(...) before execution"
            )
        return run_config

    def generate(self, run_config: BenchmarkRunConfig) -> Path:
        """Generate completions for a benchmark run config."""
        config = self._require_validated_run_config(run_config)
        return generate_completions(
            models=list(config.models),
            benchmark_spec=self.spec,
            samples=config.samples,
            output_dir=setup_output_dir(config.output_dir),
            max_tokens_by_model=(
                dict(config.max_tokens_by_model)
                if config.max_tokens_by_model is not None
                else None
            ),
            allow_mixed_token_budgets=config.allow_mixed_token_budgets,
            token_budget_defaults=config.token_budget_defaults,
        )

    def evaluate(
        self,
        generations_path: Path,
        run_config: BenchmarkRunConfig,
    ) -> Path:
        """Evaluate completions for a benchmark run config."""
        config = self._require_validated_run_config(run_config)
        evaluations_path = evaluate_completions(
            generations_path=generations_path,
            benchmark_spec=self.spec,
            judge_model=config.judge_model,
            output_dir=setup_output_dir(config.output_dir),
            legacy_token_cap_threshold=config.legacy_token_cap_threshold,
        )
        errors = self._validate_report_artifacts(evaluations_path)
        if errors:
            raise ValueError(
                "Benchmark report validation failed: " + "; ".join(errors)
            )
        return evaluations_path

    def analyze(self, evaluations_path: Path, run_config: BenchmarkRunConfig) -> dict:
        """Analyze benchmark results for a run config."""
        config = self._require_validated_run_config(run_config)
        if "summary" not in config.effective_analysis_lanes:
            logger.info("Skipping summary analysis for preset %s", config.preset_name)
            return {}
        return analyze_results(
            evaluations_path,
            Path(config.output_dir),
            quality_filter_mode=config.quality_filter_mode,
        )

    def rank(self, evaluations_path: Path, run_config: BenchmarkRunConfig) -> None:
        """Run pairwise ranking for a benchmark run config."""
        config = self._require_validated_run_config(run_config)
        if "pairwise" not in config.effective_analysis_lanes:
            logger.info("Skipping pairwise ranking for preset %s", config.preset_name)
            return
        rank_models(
            evaluations_path,
            self.spec,
            Path(config.output_dir),
            quality_filter_mode=config.quality_filter_mode,
        )

    def validate_report(
        self,
        evaluations_path: Path,
        run_config: BenchmarkRunConfig,
    ) -> list:
        """Validate report artifacts for a benchmark run config."""
        self._require_validated_run_config(run_config)
        return self._validate_report_artifacts(evaluations_path)

    def _validate_report_artifacts(
        self,
        evaluations_path: Path,
    ) -> list:
        """Validate report artifacts for a benchmark evaluation file."""
        return validate_benchmark_report(evaluations_path)
