"""Google Generative AI (Gemini) client using google-genai."""
import logging
import time
from typing import Optional

from compass.clients.base import CompletionClient, CompletionResponse

logger = logging.getLogger(__name__)


class GoogleAIClient(CompletionClient):
    """Client for Google Generative AI (Gemini) models using google-genai.

    Uses the official google-genai package with support for latest Gemini models.

    Usage:
        client = GoogleAIClient(api_key="...", model="gemini-1.5-flash")
        response = client.complete("What is 2+2?")
        print(response.completion)
    """

    def __init__(self, model: str, api_key: Optional[str] = None, request_interval: float = 0.1):
        """
        Initialize Google AI (Gemini) client.

        Args:
            model: Model name (e.g., 'gemini-1.5-flash', 'gemini-1.5-pro')
            api_key: Google API key. If None, reads from GOOGLE_API_KEY env var.
            request_interval: Minimum seconds between requests

        Raises:
            ImportError: If google-genai package is not installed
        """
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "Gemini support requires the 'google-genai' package. "
                "Run: pip install google-genai"
            ) from exc

        # Initialize client
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client()  # Uses GOOGLE_API_KEY env var

        self.model = model
        self._input_tokens = 0
        self._output_tokens = 0
        self._request_interval = request_interval
        self._last_call_at: float = 0.0

        logger.info(f"Using {model} (google-genai)")

    @property
    def total_tokens(self) -> dict:
        """Total tokens used across all requests."""
        return {"input": self._input_tokens, "output": self._output_tokens}

    @property
    def total_cost_usd(self) -> float:
        """Gemini free tier is free, paid tier billed separately."""
        return 0.0

    def _throttle(self) -> None:
        """Enforce minimum request interval."""
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
        self._throttle()
        self._last_call_at = time.monotonic()

        try:
            # Build request with system prompt if provided
            if system:
                full_prompt = f"{system}\n\n{prompt}"
            else:
                full_prompt = prompt

            # Call Gemini API
            response = self.client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config={
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                },
            )

            completion = response.text if response.text else ""

            # Estimate tokens (google-genai may not always provide exact counts)
            input_tokens = len(full_prompt.split())
            output_tokens = len(completion.split())

            self._input_tokens += input_tokens
            self._output_tokens += output_tokens

            return CompletionResponse(
                completion=completion.strip(),
                tokens_used={"input": input_tokens, "output": output_tokens},
                cost_usd=0.0,
            )

        except Exception as exc:
            logger.error(f"Gemini error: {exc}")
            raise RuntimeError(f"Gemini inference failed: {exc}") from exc
