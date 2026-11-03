"""Base client abstraction for LLM API calls."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CompletionResponse:
    """Response from a completion call."""

    completion: str
    tokens_used: Optional[dict] = None  # {"input": X, "output": Y}
    cost_usd: float = 0.0
    logprobs: Optional[list] = None
    finish_reason: str = ""
    request_settings: dict = field(default_factory=dict)


class CompletionClient(ABC):
    """Base class for LLM completion clients."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        max_tokens: int = 180,
        temperature: float = 0.0,
        system: Optional[str] = None,
        logprobs: bool = False,
        top_logprobs: int = 0,
    ) -> CompletionResponse:
        """Get a completion from the LLM."""
        raise NotImplementedError
