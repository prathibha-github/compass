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
from typing import Any, Dict, List, Optional, Union

from compass.cache import EvaluationCache
from compass.clients import (
    AnthropicClient,
    GoogleAIClient,
    OllamaClient,
    OpenAIClient,
)
from compass.detectors.base import (
    TicSuite,
    summarize_detector_records,
    suite_uses_llm_judge,
)
from compass.evaluation.checkpoint import CheckpointManager
from compass.evaluation.record_schema import suite_sample_identity
from compass.evaluation.suite_io import (
    append_suite_generation,
    load_suite_generations,
    reset_suite_generations,
)
from compass.judges import JudgeConfig, LLMJudge
from compass.rubrics.base import Rubric

logger = logging.getLogger(__name__)

_PathLike = Union[str, Path]


def _create_client(model: str):
    """Mirror benchmark client routing so suites share provider behavior."""
    if model.startswith("gemini"):
        return GoogleAIClient(model=model, allow_estimated_usage=True)
    if model.startswith("gpt") or model.startswith("o4"):
        required_temperature = None
        if model.startswith("gpt-5") or model.startswith("o4"):
            required_temperature = 1.0
        return OpenAIClient(model=model, required_temperature=required_temperature)
    if model.startswith("claude"):
        return AnthropicClient(model=model)
    return OllamaClient(model=model)


def _is_judge_detector(detector) -> bool:
    return hasattr(detector, "detect_with_judge")


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
        client_factory: Override for client construction (tests inject fakes).

    Returns the path to the generations JSONL.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    generations_path = out / "suite_generations.jsonl"

    if not resume:
        reset_suite_generations(generations_path)

    completed = set(load_suite_generations(generations_path).keys())
    logger.info("Suite generation: %d cells already complete", len(completed))

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
                        response = client.complete(
                            prompt=prompt.text,
                            max_tokens=suite.max_tokens,
                            temperature=temperature,
                            system=condition.system_prompt,
                        )
                        tokens_used = getattr(response, "tokens_used", {}) or {}
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
                                "completion": response.completion,
                                "tokens_used": tokens_used,
                                "cost_usd": getattr(response, "cost_usd", 0.0),
                                "finish_reason": str(
                                    getattr(response, "finish_reason", "") or ""
                                ),
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
        logger.warning("Suite generation had %d runtime failures", failures)
    return generations_path


def evaluate_suite_completions(
    suite: TicSuite,
    generations_path: _PathLike,
    output_dir: _PathLike,
    *,
    judge_model: Optional[str] = None,
    resume: bool = True,
    client_factory=_create_client,
) -> Path:
    """Score saved completions with the suite's detectors.

    Heuristic detectors run directly; LLM-judge detectors run through
    :class:`LLMJudge` with the user prompt supplied as context, and judge calls
    are cached. No model generation happens here, so re-running with a different
    ``judge_model`` re-scores the same completions without regenerating.

    Returns the path to ``suite_evaluations_<judge_model>.jsonl``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    judge_tag = judge_model or "heuristic"
    evaluations_path = out / f"suite_evaluations_{judge_tag}.jsonl"

    checkpoint = CheckpointManager(str(evaluations_path))
    if not resume:
        checkpoint.reset()
    completed = checkpoint.load()

    generations = load_suite_generations(Path(generations_path), strict=True)

    needs_judge = suite_uses_llm_judge(suite)
    judges: Dict[str, LLMJudge] = {}
    if needs_judge:
        if not judge_model:
            raise ValueError(
                "suite has LLM-judge detectors; judge_model is required to evaluate"
            )
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
                    result = judges[detector.name].evaluate(
                        completion, context=gen.get("prompt_text")
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
                        "completion": completion[:200],
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
        logger.warning("Suite evaluation had %d runtime failures", failures)
    return evaluations_path


def summarize_suite_evaluations(
    evaluations_path: _PathLike,
    suite: TicSuite,
) -> Dict[str, dict]:
    """Aggregate per-detector hit rates and Wilson CIs by condition.

    Regroups the per-detector evaluation rows back into per-sample records and
    delegates to :func:`summarize_detector_records`.
    """
    # (model, prompt_id, condition, sample_idx) -> {detector_name: {count, hit}}
    cells: Dict[Any, Dict[str, dict]] = {}
    with open(evaluations_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
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

    condition_records: Dict[str, List[Dict[str, dict]]] = {}
    for (_model, _prompt_id, condition, _sample_idx), detector_map in cells.items():
        condition_records.setdefault(condition, []).append(detector_map)

    return summarize_detector_records(condition_records, suite)
