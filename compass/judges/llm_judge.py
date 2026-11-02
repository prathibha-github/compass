"""LLM-based judge for evaluating model completions."""
import hashlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from compass.judges.base import EvaluationResult, JudgeConfig
from compass.judges.parsing import SCORING_VERSION, parse_scored_response

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

    def evaluate(self, text: str, context: Optional[str] = None) -> EvaluationResult:
        """Evaluate text with this judge's rubric.

        Args:
            text: The assistant response to score.
            context: Optional surrounding context (for example the user request)
                shown to the judge so prompt-aware rubrics can distinguish a
                volunteered behavior from a solicited one. It is folded into the
                cache key, so the same response under different contexts caches
                separately.

        Returns cached result if available, otherwise calls judge API.
        """
        cache_key = self._cache_coordinates(text, context)

        # Check cache
        cached = self.cache.get(*cache_key)
        if cached:
            cached.cache_hit = True
            return cached

        # Build prompt deterministically
        prompt = self._build_prompt(text, context)

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

    def _cache_coordinates(
        self, text: str, context: Optional[str] = None
    ) -> tuple[str, str, str]:
        """Build the cache lookup coordinates for this evaluation contract.

        The context is hashed alongside the text so an identical response judged
        with and without surrounding context does not collide.
        """
        keyed_text = text if context is None else f"{context}\x00{text}"
        text_hash = hashlib.sha256(keyed_text.encode()).hexdigest()[:16]
        # Old caches may contain fabricated zeros from parse failures.
        return (self.config.config_hash, text_hash, f"{self.PROMPT_VERSION}:scoring-{SCORING_VERSION}")

    def _build_prompt(self, text: str, context: Optional[str] = None) -> str:
        """Build evaluation prompt deterministically."""
        if context is not None:
            return (
                "Evaluate the assistant response in light of the user request, "
                "using this rubric.\n\n"
                f"Rubric:\n{self.config.rubric.text}\n\n"
                "Return JSON exactly in this shape:\n"
                '{"score": 0.0, "confidence": 0.9, "rationale": "brief reason"}\n'
                "(score between 0.0 and 1.0)\n\n"
                f"User request:\n{context}\n\n"
                f"Assistant response:\n{text}"
            )
        return (
            "Evaluate the assistant response below using this rubric.\n\n"
            f"Rubric:\n{self.config.rubric.text}\n\n"
            "Return JSON exactly in this shape:\n"
            '{"score": 0.0, "confidence": 0.9, "rationale": "brief reason"}\n'
            "(score between 0.0 and 1.0)\n\n"
            f"Assistant response:\n{text}"
        )

    def _parse_response(self, raw: str) -> EvaluationResult:
        """Parse a valid score or raise before any result is cached."""
        payload = parse_scored_response(raw)
        score = payload["score"]

        # Calculate hit from score and threshold (ignore judge's hit value,
        # since judge may have different internal judgment of violation).
        # This ensures hit is always consistent with score and hit_threshold.
        hit = score >= self.config.rubric.hit_threshold

        # Extract other fields safely
        confidence = payload["confidence"]
        rationale = payload["rationale"]

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
