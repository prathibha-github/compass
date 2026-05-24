"""Ollama client for local LLM inference."""
import logging
import time
from typing import Optional

from compass.clients.base import CompletionClient, CompletionResponse

logger = logging.getLogger(__name__)


class OllamaClient(CompletionClient):
    """Client for running models locally via Ollama.

    Ollama must be running before use:
        ollama serve  # Default: http://localhost:11434

    Usage:
        client = OllamaClient(model="llama3.1:latest")
        response = client.complete("What is 2+2?")
        print(response.completion)
    """

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        request_interval: float = 0.1,
    ):
        """
        Initialize Ollama client.

        Args:
            model: Model name (e.g., 'llama3.1:latest', 'mistral:latest')
            host: Ollama API endpoint
            request_interval: Minimum seconds between requests

        Raises:
            ImportError: If ollama package is not installed
        """
        try:
            from ollama import Client as OllamaAPIClient
        except ImportError as e:
            raise ImportError(
                "ollama package required for local model inference. "
                "Install with: pip install ollama"
            ) from e

        self.api_client = OllamaAPIClient(host=host)
        self.model = model
        self.host = host
        self._input_tokens = 0
        self._output_tokens = 0
        self._request_interval = request_interval
        self._last_call_at: float = 0.0

    @property
    def total_tokens(self) -> dict:
        """Total tokens used across all requests."""
        return {"input": self._input_tokens, "output": self._output_tokens}

    @property
    def total_cost_usd(self) -> float:
        """Local models are free."""
        return 0.0

    def _throttle(self) -> None:
        """Enforce the configured minimum interval between local requests."""
        elapsed = time.monotonic() - self._last_call_at
        gap = self._request_interval - elapsed
        if gap > 0:
            time.sleep(gap)

    def complete(
        self,
        prompt: str,
        max_tokens: int = 180,
        temperature: float = 0.0,
        system: Optional[str] = None,
        logprobs: bool = False,
        top_logprobs: int = 0,
    ) -> CompletionResponse:
        """
        Generate completion via Ollama.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            system: Optional system prompt

        Returns:
            CompletionResponse with completion text and token estimates

        Raises:
            RuntimeError: If Ollama API call fails (e.g., model not loaded)
        """
        if logprobs:
            raise ValueError(
                f"{self.__class__.__name__} does not support logprobs"
            )
        del top_logprobs

        self._throttle()
        self._last_call_at = time.monotonic()

        # Build full prompt with system message if provided
        full_prompt = prompt
        if system:
            full_prompt = f"<system>\n{system}\n</system>\n\n{prompt}"

        try:
            # Call Ollama API
            response = self.api_client.generate(
                model=self.model,
                prompt=full_prompt,
                stream=False,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            )

            completion = response.get("response", "")
            if not completion:
                raise RuntimeError(f"Empty response from {self.model}")

            # Ollama doesn't expose exact token counts,
            # but we can estimate from response length
            estimated_input_tokens = len(full_prompt.split())
            estimated_output_tokens = len(completion.split())

            self._input_tokens += estimated_input_tokens
            self._output_tokens += estimated_output_tokens

            return CompletionResponse(
                completion=completion,
                tokens_used={
                    "input": estimated_input_tokens,
                    "output": estimated_output_tokens,
                },
                cost_usd=0.0,
            )

        except Exception as e:
            logger.error(f"Ollama error for model {self.model}: {e}")
            raise RuntimeError(f"Ollama inference failed for {self.model}: {e}") from e
