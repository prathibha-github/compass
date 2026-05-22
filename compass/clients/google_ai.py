"""Google Generative AI (Gemini) client for LLM inference."""
import logging
import random
import time
from typing import Optional

from compass.clients.base import CompletionClient, CompletionResponse

logger = logging.getLogger(__name__)


class GoogleAIClient(CompletionClient):
    """Client for Google Generative AI (Gemini) models.

    Supports rate limiting, safety settings, and robust error handling.

    Usage:
        client = GoogleAIClient(api_key="...", model="gemini-2.0-flash")
        response = client.complete("What is 2+2?")
        print(response.completion)
    """

    def __init__(self, model: str, api_key: Optional[str] = None, request_interval: float = 2.0):
        """
        Initialize Google AI (Gemini) client.

        Args:
            model: Model name (e.g., 'gemini-2.0-flash', 'gemini-1.5-pro')
            api_key: Google API key. If None, reads from GOOGLE_API_KEY env var.
            request_interval: Minimum seconds between requests to avoid rate limits

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
        # genai.configure() sets the API key globally
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(
            model,
            system_instruction="You are a helpful assistant.",
        )
        self.model = model
        self._input_tokens = 0
        self._output_tokens = 0
        self._request_count = 0
        self._request_interval = request_interval
        self._last_call_at: float = 0.0

    @property
    def total_tokens(self) -> dict:
        """Total tokens used across all requests."""
        return {"input": self._input_tokens, "output": self._output_tokens}

    @property
    def total_cost_usd(self) -> float:
        """Gemini free tier is free."""
        return 0.0

    def _throttle(self) -> None:
        """Enforce request interval to avoid rate limits."""
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
            system: Optional system prompt (note: Gemini uses fixed system instruction,
                   this parameter is ignored)

        Returns:
            CompletionResponse with completion text and token counts

        Raises:
            RuntimeError: If Gemini API call fails after retries
        """
        if system and system != "You are a helpful assistant.":
            logger.warning(
                "Gemini client has a fixed system instruction set at construction time. "
                "Per-call system prompts are not supported and will be ignored."
            )

        max_attempts = 10
        for attempt in range(max_attempts):
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
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "threshold": "BLOCK_NONE",
                        },
                    ],
                )

                # Extract token counts from response
                usage = getattr(response, "usage_metadata", None)
                input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
                output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
                self._input_tokens += input_tokens
                self._output_tokens += output_tokens
                self._request_count += 1

                # Check if response has valid candidates before accessing text
                if response.candidates and len(response.candidates) > 0:
                    completion = response.text
                else:
                    completion = ""

                return CompletionResponse(
                    completion=completion.strip(),
                    tokens_used={
                        "input": input_tokens,
                        "output": output_tokens,
                    },
                    cost_usd=0.0,  # Free tier
                )

            except self._genai.types.BlockedPromptException as exc:
                logger.error("Gemini API blocked prompt: %s", exc)
                raise RuntimeError(f"Gemini blocked prompt: {exc}") from exc

            except Exception as exc:
                error_name = type(exc).__name__
                # Detect rate limit errors: google.api_core.exceptions.ResourceExhausted (429)
                is_rate_limit = "ResourceExhausted" in error_name or "429" in str(exc)

                if is_rate_limit:
                    wait = min(15 * (2 ** attempt), 60) + random.uniform(0, 5)
                    logger.warning(
                        "Gemini rate limit (attempt %d/%d) — waiting %.1fs",
                        attempt + 1,
                        max_attempts,
                        wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error("Gemini API error on attempt %d: %s", attempt + 1, exc)
                    if attempt == max_attempts - 1:
                        raise RuntimeError(f"Gemini inference failed after {max_attempts} attempts: {exc}") from exc
                    time.sleep(min(4 * (2 ** attempt), 30))

        raise RuntimeError(f"Gemini call failed after {max_attempts} attempts")
