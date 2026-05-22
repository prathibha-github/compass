"""Base abstractions for LLM judges."""
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

from compass.rubrics.base import Rubric


@dataclass
class EvaluationResult:
    """Result from a single evaluation by a judge.

    All fields are deterministic and reproducible given the same inputs.
    This enables auditing, reproducibility, and cost tracking across evaluations.

    Attributes:
        name: Rubric name (e.g., "sycophancy").
        score: Numerical score [0.0, 1.0], clamped. Higher is worse for negative traits.
        hit: Boolean classification; True if score >= rubric.hit_threshold.
        confidence: Judge's self-reported confidence in the score [0.0, 1.0] (optional).
        rationale: Brief explanation of why this score was assigned.
        rubric_hash: Hash of the immutable rubric used. Changes if rubric definition changes.
        judge_model: Name of the LLM that performed the evaluation (e.g., "gpt-4o").
        prompt_version: Version of the prompt template used (default "1.0").
        timestamp: ISO 8601 timestamp of when evaluation occurred.
        cache_hit: True if result came from persistent cache, False if freshly computed.
        tokens_used: Dict with "input" and "output" token counts (for cost tracking).
        cost_usd: USD cost of this evaluation (useful for budgeting large runs).

    Note:
        To reproduce identical results, ensure: same rubric (by hash), same judge_model,
        and same text. The cache key is (rubric_hash, text_hash, judge_model).
    """

    # Core evaluation results
    name: str  # Rubric name (e.g., "sycophancy")
    score: float  # [0.0, 1.0], clamped
    hit: bool  # score >= hit_threshold
    confidence: Optional[float] = None  # Judge's confidence in the score
    rationale: str = ""  # Explanation of the score

    # Reproducibility metadata
    rubric_hash: str = ""  # Hash of the rubric that was used
    judge_model: str = ""  # Model that did the judging
    prompt_version: str = "1.0"  # Version of the prompt template
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )  # ISO timestamp of evaluation
    cache_hit: bool = False  # Whether this came from cache

    # Resource tracking
    tokens_used: Dict[str, int] = field(
        default_factory=dict
    )  # {"input": X, "output": Y}
    cost_usd: float = 0.0

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return asdict(self)

    def __repr__(self) -> str:
        return f"EvaluationResult({self.name}={self.score:.2f}, hit={self.hit})"


@dataclass
class JudgeConfig:
    """Configuration for judge evaluation.

    Immutable configuration that uniquely identifies a judge setup.
    All evaluations with the same config on the same text will produce
    identical results (pulled from cache or recomputed).

    Attributes:
        rubric: The rubric to evaluate against.
        judge_model: LLM model name (e.g., "gpt-4o", "claude-opus-4-7").
        max_tokens: Maximum tokens in judge response (default 180).
        temperature: LLM temperature (default 0.0 for deterministic).
        seed: Random seed (default 42, for any stochastic components).

    The config_hash property provides a stable identifier for this configuration,
    used in cache keys and reproducibility tracking.
    """

    rubric: Rubric
    judge_model: str  # e.g., "gpt-4o", "claude-opus-4-7"
    max_tokens: int = 180
    temperature: float = 0.0  # Deterministic
    seed: int = 42

    @property
    def config_hash(self) -> str:
        """Hash of this configuration for caching.

        Truncates to 12 characters for readability. With 12 chars (~2^48 space),
        collision risk is negligible for typical use cases.
        """
        import hashlib

        content = f"{self.rubric.hash}_{self.judge_model}_{self.max_tokens}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def __repr__(self) -> str:
        return f"JudgeConfig({self.rubric.name}, {self.judge_model})"
