"""Two-phase, condition-aware runner for tic/style suites.

Phase one (:func:`generate_suite_completions`) calls each model once per
(prompt, condition, sample) cell and persists the completion. Phase two
(:func:`evaluate_suite_completions`) scores those saved completions with the
suite's detectors, so heuristic and LLM-judge detectors run over the same
generations and a judge can be added or changed later without paying to
regenerate. Judge calls are cached by :class:`EvaluationCache`.

This is the suite analogue of ``compass.benchmark.runner``. It reuses
``CheckpointManager`` for the per-detector evaluation rows and a dedicated
generation JSONL (see ``compass.evaluation.suite_io``) for the completions.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from compass.cache import EvaluationCache
from compass.clients import (
    AnthropicClient,
    GoogleAIClient,
    OllamaClient,
    OpenAIClient,
    OpenAIResponsesClient,
)
from compass.detectors.base import (
    TicSuite,
    summarize_detector_records,
    suite_uses_llm_judge,
)
from compass.evaluation.checkpoint import CheckpointManager
from compass.evaluation.record_schema import checkpoint_identity, suite_sample_identity
from compass.evaluation.contracts import detector_definition, fingerprint, validate_suite_identities, manifest_contracts
from compass.evaluation.suite_io import (
    append_suite_generation,
    load_suite_generations,
    reset_suite_generations,
)
from compass.clients.settings import default_request_settings
from compass.judges import JudgeConfig, LLMJudge
from compass.judges.context import saved_judge_context
from compass.judges.parsing import SCORING_VERSION
from compass.rubrics.base import Rubric

logger = logging.getLogger(__name__)

_PathLike = Union[str, Path]
_DEFAULT_SUITE_TOKEN_BUDGETS = {
    "default": 150,
    "gemini": 2000,
}
_SENTENCE_ENDINGS = (".", "!", "?", "\"", "'")
_TOKEN_CAP_FINISH_REASONS = {"length", "max_tokens", "max_output_tokens", "token_limit"}


def _create_client(model: str):
    """Mirror benchmark client routing so suites share provider behavior."""
    if model.startswith("gemini"):
        return GoogleAIClient(model=model, allow_estimated_usage=True)
    if model.startswith("gpt-5"):
        return OpenAIResponsesClient(
            model=model,
            max_output_token_multiplier=10,
            required_temperature=1.0,
        )
    if model.startswith("gpt") or model.startswith("o4"):
        required_temperature = None
        if model.startswith("o4"):
            required_temperature = 1.0
        return OpenAIClient(model=model, required_temperature=required_temperature)
    if model.startswith("claude"):
        return AnthropicClient(model=model)
    return OllamaClient(model=model)


def _is_judge_detector(detector) -> bool:
    return hasattr(detector, "detect_with_judge")


def _compute_suite_token_budget_by_model(
    models: List[str],
    *,
    max_tokens_by_model: Optional[Mapping[str, int]] = None,
    allow_mixed_token_budgets: bool = True,
) -> Dict[str, int]:
    if max_tokens_by_model is None:
        budgets = {
            model: _default_suite_max_tokens_for_model(model)
            for model in models
        }
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
    if len(distinct) > 1 and not allow_mixed_token_budgets:
        raise ValueError(
            "Mixed max token budgets detected across models: "
            f"{budgets}. Re-run with allow_mixed_token_budgets=True to override."
        )
    return budgets


def _default_suite_max_tokens_for_model(model: str) -> int:
    prefixes = sorted(
        (prefix for prefix in _DEFAULT_SUITE_TOKEN_BUDGETS if prefix != "default"),
        key=len,
        reverse=True,
    )
    for prefix in prefixes:
        if model.startswith(prefix):
            return int(_DEFAULT_SUITE_TOKEN_BUDGETS[prefix])
    return int(_DEFAULT_SUITE_TOKEN_BUDGETS["default"])


def _compute_suite_generation_quality(
    completion: str,
    output_tokens: int,
    max_tokens_requested: int,
    finish_reason: str = "",
) -> Dict[str, Any]:
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
        or (visible_chars < 80 and not sentence_complete)
        or (hit_token_cap and not sentence_complete)
    )
    quality_flagged = bool(hit_token_cap or is_fragment)
    return {
        "visible_chars": visible_chars,
        "visible_word_count": visible_word_count,
        "hit_token_cap": hit_token_cap,
        "is_fragment": is_fragment,
        "quality_flagged": quality_flagged,
    }


def _validate_resumed_suite_token_budgets(
    rows,
    effective_max_tokens_by_model: Mapping[str, int],
) -> None:
    for row in rows:
        model = row.get("model")
        if model not in effective_max_tokens_by_model:
            continue
        recorded = row.get("max_tokens_requested")
        if recorded is None:
            continue
        expected = int(effective_max_tokens_by_model[model])
        if int(recorded) != expected:
            raise ValueError(
                "Existing suite generations use a different max token budget "
                f"for model {model!r}: found {recorded}, expected {expected}. "
                "Use a fresh output directory or reset generations before rerunning."
            )


def _detector_rubric(detector) -> Rubric:
    """Build a stable Rubric from an LLM-judge detector's rubric text.

    The rubric hash (and therefore the judge cache key) is derived from the
    detector name, version, text, and threshold, so it is stable across runs
    and across processes.
    """
    return Rubric(
        name=detector.name,
        category="suite",
        version="1.0",
        created_at="2026-05-30",
        text=detector.rubric,
        hit_threshold=getattr(detector, "hit_threshold", 0.5),
        max_tokens=getattr(detector, "max_tokens", 180),
    )


def generate_suite_completions(
    suite: TicSuite,
    models: List[str],
    samples: int,
    output_dir: _PathLike,
    *,
    temperature: float = 0.0,
    resume: bool = False,
    max_tokens_by_model: Optional[Mapping[str, int]] = None,
    allow_mixed_token_budgets: bool = True,
    client_factory=_create_client,
) -> Path:
    """Generate and persist one completion per (model, prompt, condition, sample).

    Args:
        suite: The suite whose prompts and conditions drive generation.
        models: Models under test.
        samples: Completions per (prompt, condition) cell. Use a temperature
            above 0.0 for these to be independent draws.
        output_dir: Directory for ``suite_generations.jsonl``.
        temperature: Sampling temperature for the model under test.
        resume: Continue from an existing generations file instead of resetting.
        max_tokens_by_model: Optional explicit generation token budget per model.
            If omitted, Compass benchmark token-budget defaults are used.
        allow_mixed_token_budgets: Allow different effective token budgets across
            models. Disable for strict apples-to-apples budget comparisons.
        client_factory: Override for client construction (tests inject fakes).

    Returns the path to the generations JSONL.
    """
    validate_suite_identities(suite)
    if len(models) != len(set(models)):
        raise ValueError("Model names must be unique")
    if type(samples) is not int or samples <= 0:
        raise ValueError("samples must be a positive integer")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    generations_path = out / "suite_generations.jsonl"

    effective_max_tokens_by_model = _compute_suite_token_budget_by_model(
        models,
        max_tokens_by_model=(
            dict(max_tokens_by_model) if max_tokens_by_model is not None else None
        ),
        allow_mixed_token_budgets=allow_mixed_token_budgets,
    )
    contracts = {
        (model, suite.name, prompt.id, condition.name): fingerprint({
            "version": 1, "model": model, "suite": suite.name,
            "prompt": prompt.text, "task_type": prompt.task_type,
            "system": condition.system_prompt, "temperature": temperature,
            "max_tokens": effective_max_tokens_by_model[model],
            "provider_settings": default_request_settings(model, effective_max_tokens_by_model[model], temperature, responses_api=True) if client_factory is _create_client else {"verified": False},
        })
        for model in models for prompt in suite.prompts for condition in suite.conditions
    }
    planned = {key + (sample,): value for key, value in contracts.items() for sample in range(samples)}
    prior_manifest = generations_path.with_suffix(".manifest.json")
    if resume and prior_manifest.exists():
        previous = manifest_contracts(json.loads(prior_manifest.read_text()))
        if any(planned.get(key) != value for key, value in previous.items()):
            raise ValueError("Cannot resume generation: Planned inputs changed; use a fresh output directory to change the sample population")
    if not resume:
        reset_suite_generations(generations_path)
    existing_generations = load_suite_generations(generations_path, strict=True)
    if resume and not prior_manifest.exists() and any(row.get("generation_contract") for row in existing_generations.values()):
        raise ValueError("Generation manifest is missing for modern artifacts; restore it or use a fresh output directory")
    _validate_resumed_suite_token_budgets(
        existing_generations.values(),
        effective_max_tokens_by_model,
    )
    for identity, row in existing_generations.items():
        expected = contracts.get(identity[:4])
        if expected is None or row.get("generation_contract") != expected or identity[4] >= samples:
            raise ValueError(
                f"Cannot resume generation {identity}: inputs changed or the artifact "
                "has no verified generation contract. Use a fresh output directory "
                "or explicitly regenerate with resume=False."
            )
    completed = set(existing_generations.keys())
    logger.info("Suite generation: %d cells already complete", len(completed))

    generation_manifest = {
        "expected": [
            {"identity": list(key) + [sample_idx], "contract": value}
            for key, value in contracts.items() for sample_idx in range(samples)
        ],
    }
    manifest_path = generations_path.with_suffix(".manifest.json")
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(generation_manifest, sort_keys=True) + "\n")
    temporary.replace(manifest_path)

    clients_by_model = {model: client_factory(model) for model in models}

    written = 0
    failures = 0
    for prompt in suite.prompts:
        for condition in suite.conditions:
            for model in models:
                client = clients_by_model[model]
                for sample_idx in range(samples):
                    identity = (
                        model,
                        suite.name,
                        prompt.id,
                        condition.name,
                        sample_idx,
                    )
                    if identity in completed:
                        continue
                    try:
                        max_tokens = int(effective_max_tokens_by_model[model])
                        response = client.complete(
                            prompt=prompt.text,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            system=condition.system_prompt,
                        )
                        tokens_used = getattr(response, "tokens_used", {}) or {}
                        output_tokens = int(tokens_used.get("output", 0) or 0)
                        finish_reason = str(getattr(response, "finish_reason", "") or "")
                        settings = getattr(response, "request_settings", {}) or {}
                        effective = settings.get("effective") or {}
                        quality = _compute_suite_generation_quality(
                            completion=response.completion,
                            output_tokens=output_tokens,
                            max_tokens_requested=effective.get("max_tokens", max_tokens),
                            finish_reason=finish_reason,
                        )
                        append_suite_generation(
                            generations_path,
                            {
                                "model": model,
                                "suite": suite.name,
                                "prompt_id": prompt.id,
                                "prompt_text": prompt.text,
                                "task_type": prompt.task_type,
                                "condition": condition.name,
                                "sample_idx": sample_idx,
                                "generation_contract": contracts[identity[:4]],
                                "temperature": temperature,
                                "system_prompt": condition.system_prompt,
                                "completion": response.completion,
                                "tokens_used": tokens_used,
                                "cost_usd": getattr(response, "cost_usd", 0.0),
                                "max_tokens_requested": max_tokens,
                                "request_settings": settings,
                                "max_tokens_effective": effective.get("max_tokens"),
                                "finish_reason": finish_reason,
                                "visible_chars": quality["visible_chars"],
                                "visible_word_count": quality["visible_word_count"],
                                "hit_token_cap": quality["hit_token_cap"],
                                "is_fragment": quality["is_fragment"],
                                "quality_flagged": quality["quality_flagged"],
                            },
                        )
                        written += 1
                        if written % 10 == 0:
                            logger.info("  Generated %d completions", written)
                    except Exception as e:  # noqa: BLE001 - log and continue per cell
                        failures += 1
                        logger.error(
                            "  Failed to generate %s/%s[%s] sample %d: %s",
                            model,
                            prompt.id,
                            condition.name,
                            sample_idx,
                            e,
                        )

    logger.info("✓ Suite generation wrote %d completions", written)
    if failures:
        raise RuntimeError(
            f"Suite generation had {failures} runtime failures; successful completions were saved. "
            "Retry with resume=True before evaluating."
        )
    return generations_path


def evaluate_suite_completions(
    suite: TicSuite,
    generations_path: _PathLike,
    output_dir: _PathLike,
    *,
    judge_model: Optional[str] = None,
    resume: bool = True,
    judge_context_mode: str = "user",
    client_factory=_create_client,
) -> Path:
    """Score saved completions with the suite's detectors.

    Heuristic detectors run directly; LLM-judge detectors run through
    :class:`LLMJudge` with the user prompt supplied as context, and judge calls
    are cached. No model generation happens here, so re-running with a different
    ``judge_model`` re-scores the same completions without regenerating.

    Returns the path to ``suite_evaluations_<judge_model>.jsonl``.
    """
    validate_suite_identities(suite)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    judge_tag = judge_model or "heuristic"
    evaluations_path = out / f"suite_evaluations_{judge_tag}.jsonl"

    generations = load_suite_generations(Path(generations_path), strict=True)
    generation_manifest_path = Path(generations_path).with_suffix(".manifest.json")
    if generation_manifest_path.exists():
        manifest = json.loads(generation_manifest_path.read_text())
        expected = manifest_contracts(manifest)
        if set(generations) != set(expected) or any(
            row.get("generation_contract") != expected[key] for key, row in generations.items()
        ):
            raise ValueError("Incomplete or incompatible generations; resume generation before scoring")
    else:
        if any("generation_contract" in row for row in generations.values()):
            raise ValueError("Modern generation manifest is missing; restore it before scoring")
        logger.warning("Legacy generations have no manifest; intended generation coverage cannot be verified")

    contexts = {
        key: saved_judge_context(gen, judge_context_mode)
        for key, gen in generations.items()
    } if suite_uses_llm_judge(suite) else {}
    needs_judge = suite_uses_llm_judge(suite)
    if needs_judge and not judge_model:
        raise ValueError("suite has LLM-judge detectors; judge_model is required to evaluate")
    definitions = {d.name: detector_definition(d) for d in suite.detectors}
    contracts = {}
    for (model, source_suite, prompt_id, condition, sample_idx), gen in generations.items():
        if source_suite != suite.name:
            raise ValueError(f"Generation belongs to suite {source_suite!r}, expected {suite.name!r}")
        for detector in suite.detectors:
            identity = suite_sample_identity(model, suite.name, detector.name, prompt_id, condition, sample_idx)
            contracts[identity] = fingerprint({
                "generation": gen,
                "detector": definitions[detector.name],
                "judge_model": judge_model if _is_judge_detector(detector) else None,
                "judge_context": contexts.get((model, source_suite, prompt_id, condition, sample_idx)),
                "judge_provider_settings": default_request_settings(judge_model, getattr(detector, "max_tokens", 180), 0.0, responses_api=True) if _is_judge_detector(detector) and client_factory is _create_client else {"verified": False},
                "judge_prompt_version": LLMJudge.PROMPT_VERSION,
                "scoring_version": SCORING_VERSION,
            })

    previous_manifest = evaluations_path.with_suffix(".manifest.json")
    if resume and previous_manifest.exists():
        previous = manifest_contracts(json.loads(previous_manifest.read_text()))
        if any(contracts.get(key) != value for key, value in previous.items()):
            raise ValueError("Cannot resume evaluation: Planned inputs changed; re-score with resume=False")
    completed = set()
    if resume and evaluations_path.exists():
        with evaluations_path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("evaluation_contract") and not previous_manifest.exists():
                    raise ValueError("Evaluation manifest is missing for modern artifacts; restore it or re-score with resume=False")
                identity = checkpoint_identity(row)
                expected = contracts.get(identity)
                if expected is None or row.get("evaluation_contract") != expected:
                    raise ValueError(
                        f"Cannot resume evaluation {identity}: inputs changed or the artifact "
                        "has no verified evaluation contract. Re-score saved generations "
                        "with resume=False or use a fresh output directory."
                    )
                if identity in completed:
                    raise ValueError(f"Duplicate evaluation identity: {identity}")
                completed.add(identity)

    checkpoint = CheckpointManager(str(evaluations_path))
    if not resume:
        checkpoint.reset()
    # Retain the expected population even when an entire sample fails judging.
    manifest_path = evaluations_path.with_suffix(".manifest.json")
    manifest = {
        "suite": suite.name,
        "generation_coverage_verified": generation_manifest_path.exists(),
        "detectors": definitions,
        "expected": [{"identity": list(key), "contract": value} for key, value in contracts.items()],
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    judges: Dict[str, LLMJudge] = {}
    if needs_judge:
        judge_client = client_factory(judge_model)
        cache = EvaluationCache(cache_dir=str(out / ".cache"))
        for detector in suite.detectors:
            if _is_judge_detector(detector):
                judges[detector.name] = LLMJudge(
                    JudgeConfig(
                        rubric=_detector_rubric(detector),
                        judge_model=judge_model,
                        max_tokens=getattr(detector, "max_tokens", 180),
                    ),
                    client=judge_client,
                    cache=cache,
                )

    scored = 0
    failures = 0
    for (model, _suite, prompt_id, condition, sample_idx), gen in generations.items():
        for detector in suite.detectors:
            identity = suite_sample_identity(
                model=model,
                suite=suite.name,
                detector=detector.name,
                prompt_id=prompt_id,
                condition=condition,
                sample_idx=sample_idx,
            )
            if identity in completed:
                continue
            try:
                completion = gen["completion"]
                if _is_judge_detector(detector):
                    selected_context = contexts[(model, _suite, prompt_id, condition, sample_idx)]
                    result = judges[detector.name].evaluate(
                        completion, context=selected_context["user_request"],
                        system_context=selected_context["system_prompt"],
                    )
                    score, hit = result.score, result.hit
                    count = 1 if hit else 0
                    rationale = result.rationale
                else:
                    dr = detector.detect(completion)
                    score, hit, count, rationale = dr.score, dr.hit, dr.count, ""

                checkpoint.save(
                    {
                        "model": model,
                        "suite": suite.name,
                        "detector": detector.name,
                        "prompt_id": prompt_id,
                        "condition": condition,
                        "task_type": gen.get("task_type", "general"),
                        "sample_idx": sample_idx,
                        "evaluation_contract": contracts[identity],
                        "judge_context": contexts.get((model, _suite, prompt_id, condition, sample_idx)) if _is_judge_detector(detector) else None,
                        "generation_fingerprint": fingerprint(gen),
                        "generation_coverage_verified": generation_manifest_path.exists(),
                        "completion": completion[:200],
                        "generation_visible_chars": gen.get("visible_chars"),
                        "generation_visible_word_count": gen.get("visible_word_count"),
                        "generation_hit_token_cap": gen.get("hit_token_cap"),
                        "generation_is_fragment": gen.get("is_fragment"),
                        "generation_quality_flagged": gen.get("quality_flagged"),
                        "generation_finish_reason": gen.get("finish_reason", ""),
                        "generation_max_tokens_requested": gen.get(
                            "max_tokens_requested"
                        ),
                        "score": score,
                        "hit": hit,
                        "count": count,
                        "rationale": (rationale or "")[:200],
                        "judge_model": judge_model if _is_judge_detector(detector) else None,
                    }
                )
                scored += 1
            except Exception as e:  # noqa: BLE001 - log and continue per detector
                failures += 1
                logger.error(
                    "  Failed to score %s/%s[%s] sample %d with %s: %s",
                    model,
                    prompt_id,
                    condition,
                    sample_idx,
                    detector.name,
                    e,
                )

    logger.info("✓ Suite evaluation scored %d detector results", scored)
    if failures:
        raise RuntimeError(
            f"Suite evaluation had {failures} runtime failures; successful scores were saved. "
            "Retry with resume=True before reporting results."
        )
    return evaluations_path


def summarize_suite_evaluations(
    evaluations_path: _PathLike,
    suite: TicSuite,
    *,
    pool_models: bool = False,
) -> Dict[str, dict]:
    """Return per-detector summaries indexed by model, then condition.

    Set ``pool_models=True`` to explicitly combine models and return the older
    condition-indexed shape. Pooling weights each response equally, so models
    with more responses contribute more to the result.
    """
    # (model, prompt_id, condition, sample_idx) -> {detector_name: {count, hit}}
    cells: Dict[Any, Dict[str, dict]] = {}
    manifest_path = Path(evaluations_path).with_suffix(".manifest.json")
    if not manifest_path.exists():
        raise ValueError("Cannot verify evaluation coverage without its manifest; re-score saved generations")
    manifest = json.loads(manifest_path.read_text())
    definitions = {d.name: detector_definition(d) for d in suite.detectors}
    if manifest["suite"] != suite.name or manifest["detectors"] != json.loads(json.dumps(definitions)):
        raise ValueError("Evaluation definitions changed; re-score before summarizing")
    expected = manifest_contracts(manifest)
    seen = set()
    with open(evaluations_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            identity = checkpoint_identity(row)
            if identity in seen or identity not in expected or row.get("evaluation_contract") != expected[identity]:
                raise ValueError("Duplicate, unexpected, or stale evaluation row; re-score before summarizing")
            seen.add(identity)
            cell_key = (
                row["model"],
                row["prompt_id"],
                row["condition"],
                row.get("sample_idx", 0),
            )
            cells.setdefault(cell_key, {})[row["detector"]] = {
                "count": row.get("count", 1 if row.get("hit") else 0),
                "hit": row.get("hit", False),
            }

    if seen != set(expected):
        raise ValueError("Incomplete evaluation coverage; retry failed evaluations before summarizing")

    model_records: Dict[str, Dict[str, list]] = {}
    for (model, _prompt_id, condition, _sample_idx), detector_map in cells.items():
        group = "pooled" if pool_models else model
        model_records.setdefault(group, {}).setdefault(condition, []).append(detector_map)

    summaries = {model: summarize_detector_records(records, suite) for model, records in model_records.items()}
    for summary in summaries.values():
        for condition in summary.values():
            condition["generation_coverage_verified"] = manifest.get("generation_coverage_verified", False)
    return summaries.get("pooled", {}) if pool_models else summaries
