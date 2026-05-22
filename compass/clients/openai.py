"""OpenAI client for LLM inference."""
import logging
from typing import Optional

from compass.clients.base import CompletionClient, CompletionResponse

logger = logging.getLogger(__name__)


class OpenAIClient(CompletionClient):
    """Client for OpenAI API (GPT-4, GPT-4o-mini, etc.).

    Usage:
        client = OpenAIClient(model="gpt-4o-mini")
        response = client.complete("What is 2+2?")
        print(response.completion)
    """

    def __init__(self, model: str, api_key: Optional[str] = None):
        """
        Initialize OpenAI client.

        Args:
            model: Model name (e.g., 'gpt-4o-mini', 'gpt-4o')
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.

        Raises:
            ImportError: If openai package is not installed
        """
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            ) from e

        self.model = model
        if api_key:
            openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
        self._input_tokens = 0
        self._output_tokens = 0

    @property
    def total_tokens(self) -> dict:
        """Total tokens used across all requests."""
        return {"input": self._input_tokens, "output": self._output_tokens}

    @property
    def total_cost_usd(self) -> float:
        """Total cost in USD (estimated based on token counts)."""
        # Approximate costs for gpt-4o-mini (as of May 2026)
        input_cost = self._input_tokens * 0.00015 / 1000  # $0.15 per 1M tokens
        output_cost = self._output_tokens * 0.0006 / 1000  # $0.60 per 1M tokens
        return input_cost + output_cost

    def complete(
        self,
        prompt: str,
        max_tokens: int = 180,
        temperature: float = 0.0,
        system: Optional[str] = None,
    ) -> CompletionResponse:
        """
        Generate completion via OpenAI API.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            system: Optional system prompt

        Returns:
            CompletionResponse with completion text and token counts

        Raises:
            RuntimeError: If OpenAI API call fails
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            completion = response.choices[0].message.content
            if not completion:
                raise RuntimeError(f"Empty response from {self.model}")

            # Track tokens
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens

            return CompletionResponse(
                completion=completion,
                tokens_used={
                    "input": input_tokens,
                    "output": output_tokens,
                },
                cost_usd=float(response.usage.prompt_tokens) * 0.00015 / 1000
                + float(response.usage.completion_tokens) * 0.0006 / 1000,
            )

        except Exception as e:
            logger.error(f"OpenAI error for model {self.model}: {e}")
            raise RuntimeError(f"OpenAI inference failed for {self.model}: {e}") from e
