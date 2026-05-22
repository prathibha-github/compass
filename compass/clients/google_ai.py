"""Google Generative AI (Gemini) client for LLM inference."""
import logging
import time
from typing import Optional

from compass.clients.base import CompletionClient, CompletionResponse

logger = logging.getLogger(__name__)


class GoogleAIClient(CompletionClient):
    """Client for Google Generative AI (Gemini) models.

    WARNING: Free tier has aggressive rate limits (~1 request per minute).
    For benchmarking, consider using local Ollama models or paid API tier.

    Usage:
        client = GoogleAIClient(api_key="...", model="gemini-2.0-flash")
        response = client.complete("What is 2+2?")
        print(response.completion)
    """

    def __init__(self, model: str, api_key: Optional[str] = None, request_interval: float = 60.0):
        """
        Initialize Google AI (Gemini) client.

        Args:
            model: Model name (e.g., 'gemini-2.0-flash', 'gemini-1.5-pro')
            api_key: Google API key. If None, reads from GOOGLE_API_KEY env var.
            request_interval: Minimum seconds between requests (default 60s for free tier)

        Raises:
            ImportError: If google-generativeai package is not installed
        """
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "Gemini support requires the 'google-generativeai' package. "
                "Run: pip install google-generativeai"
            ) from exc

        self._genai = genai
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)
        self.model = model
        self._input_tokens = 0
        self._output_tokens = 0
        self._request_interval = request_interval
        self._last_call_at: float = 0.0

        if "2.5" in model:
            logger.warning(
                f"Using {model} on free tier. This model has reasoning mode enabled by default. "
                "Consider using gemini-2.0-flash for benchmarking instead."
            )
        else:
            logger.warning(
                f"Using {model} on free tier. Rate limits are ~1 request/min. "
                "For faster benchmarking, use local Ollama models instead."
            )

    @property
    def total_tokens(self) -> dict:
        """Total tokens used across all requests."""
        return {"input": self._input_tokens, "output": self._output_tokens}

    @property
    def total_cost_usd(self) -> float:
        """Gemini free tier is free."""
        return 0.0

    def _throttle(self) -> None:
        """Enforce minimum request interval for free tier."""
        elapsed = time.monotonic() - self._last_call_at
        gap = self._request_interval - elapsed
        if gap > 0:
            logger.info(f"Rate limit: waiting {gap:.1f}s before next request")
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
            system: Optional system prompt (ignored for Gemini)

        Returns:
            CompletionResponse with completion text and token counts

        Raises:
            RuntimeError: If Gemini API call fails
        """
        if system:
            logger.debug("Gemini fixed system instruction. Per-call system prompt ignored.")

        self._throttle()
        self._last_call_at = time.monotonic()

        try:
            response = self.client.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                },
                safety_settings=[
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ],
            )

            # Handle empty response (finish_reason=2)
            if not response.text:
                logger.warning(f"Gemini returned empty response (finish_reason={getattr(response, 'finish_reason', 'unknown')})")
                completion = ""
            else:
                completion = response.text

            # Estimate tokens
            input_tokens = len(prompt.split())
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
