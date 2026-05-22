"""Base abstractions for LLM judges."""
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, Optional

from compass.rubrics.base import Rubric


@dataclass
class EvaluationResult:
    """Result from a single evaluation by a judge.

    All fields are deterministic and reproducible given the same inputs.
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
    timestamp: str = ""  # ISO timestamp of evaluation
    cache_hit: bool = False  # Whether this came from cache

    # Resource tracking
    tokens_used: Dict[str, int] = field(default_factory=dict)  # {"input": X, "output": Y}
    cost_usd: float = 0.0

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return asdict(self)

    def __repr__(self) -> str:
        return f"EvaluationResult({self.name}={self.score:.2f}, hit={self.hit})"


@dataclass
class JudgeConfig:
    """Configuration for judge evaluation."""

    rubric: Rubric
    judge_model: str  # e.g., "gpt-4o", "claude-opus-4-7"
    max_tokens: int = 180
    temperature: float = 0.0  # Deterministic
    seed: int = 42

    @property
    def config_hash(self) -> str:
        """Hash of this configuration for caching."""
        import hashlib

        content = f"{self.rubric.hash}_{self.judge_model}_{self.max_tokens}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def __repr__(self) -> str:
        return f"JudgeConfig({self.rubric.name}, {self.judge_model})"
