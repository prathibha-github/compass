"""Rubric definitions for evaluations."""
import hashlib
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Rubric:
    """Immutable, versioned rubric for evaluation.

    The hash is deterministic and changes if any aspect of the rubric changes,
    making it easy to track reproducibility.
    """

    name: str  # e.g., "sycophancy"
    category: str  # e.g., "constitutional", "alignment", "safety"
    version: str  # e.g., "1.0", "1.1" (semantic)
    created_at: str  # ISO timestamp
    text: str  # The actual rubric prompt given to the judge
    hit_threshold: float = 0.5  # Score >= threshold means hit
    max_tokens: int = 180  # Default max tokens for judge response

    @property
    def hash(self) -> str:
        """Immutable hash of this rubric. Changes if content changes.

        Used as cache key and reproducibility identifier.
        """
        content = f"{self.name}_{self.version}_{self.text}_{self.hit_threshold}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @property
    def full_id(self) -> str:
        """Human-readable ID combining name, version, and hash."""
        return f"{self.name}_v{self.version}_{self.hash}"

    def __repr__(self) -> str:
        return f"Rubric({self.name}_v{self.version})"
