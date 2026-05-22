"""Google Generative AI (Gemini) client for LLM inference."""
import logging
from typing import Optional

from compass.clients.base import CompletionClient, CompletionResponse

logger = logging.getLogger(__name__)


class GoogleAIClient(CompletionClient):
    """Client for Google Generative AI (Gemini) free tier.

    Usage:
        client = GoogleAIClient(model="gemini-1.5-flash")
        response = client.complete("What is 2+2?")
        print(response.completion)
    """

    def __init__(self, model: str, api_key: Optional[str] = None):
        """
        Initialize Google AI client.

        Args:
            model: Model name (e.g., 'gemini-1.5-flash', 'gemini-1.5-pro')
            api_key: Google API key. If None, reads from GOOGLE_API_KEY env var.

        Raises:
            ImportError: If google-generativeai package is not installed
        """
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError(
                "google-generativeai package required. Install with: pip install google-generativeai"
            ) from e

        self.model_name = model
        if api_key:
            genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(model)
        self._input_tokens = 0
        self._output_tokens = 0

    @property
    def total_tokens(self) -> dict:
        """Total tokens used across all requests."""
        return {"input": self._input_tokens, "output": self._output_tokens}

    @property
    def total_cost_usd(self) -> float:
        """Gemini free tier is free."""
        return 0.0

    def complete(
        self,
        prompt: str,
        max_tokens: int = 180,
        temperature: float = 0.0,
        system: Optional[str] = None,
    ) -> CompletionResponse:
        """
        Generate completion via Google Gemini API.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            system: Optional system prompt

        Returns:
            CompletionResponse with completion text and token counts

        Raises:
            RuntimeError: If Gemini API call fails
        """
        try:
            # Build full prompt with system message if provided
            full_prompt = prompt
            if system:
                full_prompt = f"{system}\n\n{prompt}"

            response = self.model.generate_content(
                full_prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                },
                stream=False,
            )

            if not response.text:
                raise RuntimeError(f"Empty response from {self.model_name}")

            completion = response.text

            # Gemini doesn't always provide token counts in free tier
            # Estimate from response length
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
                cost_usd=0.0,  # Free tier
            )

        except Exception as e:
            logger.error(f"Gemini error for model {self.model_name}: {e}")
            raise RuntimeError(f"Gemini inference failed for {self.model_name}: {e}") from e
