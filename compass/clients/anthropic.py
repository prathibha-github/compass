"""Anthropic client for LLM inference."""
import logging
from typing import Optional

from compass.clients.base import CompletionClient, CompletionResponse

logger = logging.getLogger(__name__)

# Approximate prices per 1M tokens (as of May 2026). Matched on model prefix.
_PRICING = {
    "claude-haiku": (0.80, 4.00),      # input, output $/1M tokens
    "claude-sonnet": (3.00, 15.00),
    "claude-opus": (15.00, 75.00),
}
_PRICING_DEFAULT = (3.00, 15.00)  # fall back to Sonnet pricing if unknown


def _price_per_token(model: str) -> tuple:
    for prefix, prices in _PRICING.items():
        if prefix in model:
            return prices
    return _PRICING_DEFAULT


class AnthropicClient(CompletionClient):
    """Client for Anthropic API (Claude models).

    Usage:
        client = AnthropicClient(model="claude-haiku-4-5")
        response = client.complete("What is 2+2?")
        print(response.completion)
    """

    def __init__(self, model: str, api_key: Optional[str] = None):
        """
        Initialize Anthropic client.

        Args:
            model: Model name (e.g., 'claude-haiku-4-5', 'claude-sonnet-4-6')
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.

        Raises:
            ImportError: If anthropic package is not installed
        """
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            ) from e

        self.model = model
        self.client = Anthropic(api_key=api_key)
        self._input_tokens = 0
        self._output_tokens = 0

    @property
    def total_tokens(self) -> dict:
        """Total tokens used across all requests."""
        return {"input": self._input_tokens, "output": self._output_tokens}

    @property
    def total_cost_usd(self) -> float:
        """Total cost in USD (estimated based on token counts and model pricing)."""
        input_price, output_price = _price_per_token(self.model)
        return (
            self._input_tokens * input_price / 1_000_000
            + self._output_tokens * output_price / 1_000_000
        )

    def complete(
        self,
        prompt: str,
        max_tokens: int = 180,
        temperature: float = 0.0,
        system: Optional[str] = None,
    ) -> CompletionResponse:
        """
        Generate completion via Anthropic API.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            system: Optional system prompt

        Returns:
            CompletionResponse with completion text and token counts

        Raises:
            RuntimeError: If Anthropic API call fails
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system or "",
                messages=[{"role": "user", "content": prompt}],
            )

            completion = response.content[0].text
            if not completion:
                raise RuntimeError(f"Empty response from {self.model}")

            # Track tokens
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens

            input_price, output_price = _price_per_token(self.model)
            return CompletionResponse(
                completion=completion,
                tokens_used={
                    "input": input_tokens,
                    "output": output_tokens,
                },
                cost_usd=(
                    input_tokens * input_price / 1_000_000
                    + output_tokens * output_price / 1_000_000
                ),
            )

        except Exception as e:
            logger.error(f"Anthropic error for model {self.model}: {e}")
            raise RuntimeError(f"Anthropic inference failed for {self.model}: {e}") from e
