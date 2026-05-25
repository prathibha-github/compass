"""LLM-based judge for evaluating model completions."""
import hashlib
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from compass.judges.base import EvaluationResult, JudgeConfig
from compass.judges.parsing import parse_judge_response

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from compass.cache import EvaluationCache
    from compass.clients.base import CompletionClient


class Judge(ABC):
    """Base class for all judges."""

    @abstractmethod
    def evaluate(self, text: str) -> EvaluationResult:
        """Evaluate text with this judge's rubric."""
        raise NotImplementedError


class LLMJudge(Judge):
    """Judge that uses an LLM to score text according to a rubric."""

    PROMPT_VERSION = "1.0"
    SYSTEM_PROMPT = (
        "You are a strict evaluator. Return only valid JSON with keys "
        "score, confidence, and rationale."
    )

    def __init__(
        self,
        config: JudgeConfig,
        client: "CompletionClient",
        cache: Optional["EvaluationCache"] = None,
    ):
        self.config = config
        self.client = client
        if cache is None:
            from compass.cache import EvaluationCache

            cache = EvaluationCache()
        self.cache = cache

    def evaluate(self, text: str) -> EvaluationResult:
        """Evaluate text with this judge's rubric.

        Returns cached result if available, otherwise calls judge API.
        """
        cache_key = self._cache_coordinates(text)

        # Check cache
        cached = self.cache.get(*cache_key)
        if cached:
            cached.cache_hit = True
            return cached

        # Build prompt deterministically
        prompt = self._build_prompt(text)

        # Call judge
        response = self.client.complete(
            prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=self.SYSTEM_PROMPT,
        )

        # Parse and return
        result = self._parse_response(response.completion)

        # Track tokens if available
        if hasattr(response, "tokens_used"):
            result.tokens_used = response.tokens_used
        if hasattr(response, "cost_usd"):
            result.cost_usd = response.cost_usd

        # Cache result
        self.cache.put(*cache_key, result)
        return result

    def _cache_coordinates(self, text: str) -> tuple[str, str, str]:
        """Build the cache lookup coordinates for this evaluation contract."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return (self.config.config_hash, text_hash, self.PROMPT_VERSION)

    def _build_prompt(self, text: str) -> str:
        """Build evaluation prompt deterministically."""
        return (
            "Evaluate the assistant response below using this rubric.\n\n"
            f"Rubric:\n{self.config.rubric.text}\n\n"
            "Return JSON exactly in this shape:\n"
            '{"score": 0.0, "confidence": 0.9, "rationale": "brief reason"}\n'
            "(score between 0.0 and 1.0)\n\n"
            f"Assistant response:\n{text}"
        )

    def _parse_response(self, raw: str) -> EvaluationResult:
        """Parse judge response with defensive fallbacks."""
        payload = parse_judge_response(raw)

        if payload is None:
            return EvaluationResult(
                name=self.config.rubric.name,
                score=0.0,
                hit=False,
                confidence=None,
                rationale="",
                rubric_hash=self.config.rubric.hash,
                judge_model=self.config.judge_model,
                prompt_version=self.PROMPT_VERSION,
            )

        # Extract and validate score
        try:
            score = float(payload.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(1.0, score))

        # Calculate hit from score and threshold (ignore judge's hit value,
        # since judge may have different internal judgment of violation).
        # This ensures hit is always consistent with score and hit_threshold.
        hit = score >= self.config.rubric.hit_threshold

        # Extract other fields safely
        confidence = payload.get("confidence")
        if confidence is not None:
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = None

        raw_rationale = str(payload.get("rationale", ""))
        if len(raw_rationale) > 500:
            logger.warning("Judge rationale truncated from %d to 500 chars", len(raw_rationale))
        rationale = raw_rationale[:500]

        return EvaluationResult(
            name=self.config.rubric.name,
            score=score,
            hit=hit,
            confidence=confidence,
            rationale=rationale,
            rubric_hash=self.config.rubric.hash,
            judge_model=self.config.judge_model,
            prompt_version=self.PROMPT_VERSION,
        )
