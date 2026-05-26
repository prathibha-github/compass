"""Google Generative AI (Gemini) client using google-genai."""
import logging
import time
from typing import Optional

from compass.clients.base import CompletionClient, CompletionResponse
from compass.clients.pricing import get_pricing

logger = logging.getLogger(__name__)

_SAFETY_OFF = [
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
]


class GoogleAIClient(CompletionClient):
    """Client for Google Generative AI (Gemini) models using google-genai.

    Uses the official google-genai package with support for latest Gemini models.

    Usage:
        client = GoogleAIClient(api_key="...", model="gemini-1.5-flash")
        response = client.complete("What is 2+2?")
        print(response.completion)
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        request_interval: float = 0.1,
        disable_safety_filters: bool = False,
        allow_estimated_usage: bool = False,
    ):
        """
        Initialize Google AI (Gemini) client.

        Args:
            model: Model name (e.g., 'gemini-1.5-flash', 'gemini-1.5-pro')
            api_key: Google API key. If None, reads from GOOGLE_API_KEY env var.
            request_interval: Minimum seconds between requests
            disable_safety_filters: Whether to disable Gemini safety settings.
            allow_estimated_usage: Whether to estimate token usage when Gemini
                omits usage metadata.

        Raises:
            ImportError: If google-genai package is not installed
        """
        try:
            from google import genai
            from google.genai import types as genai_types
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
        self._types = genai_types
        self._pricing = get_pricing(model)
        self._input_tokens = 0
        self._output_tokens = 0
        self._request_interval = request_interval
        self._last_call_at: float = 0.0
        self._request_count = 0
        self._disable_safety_filters = disable_safety_filters
        self._allow_estimated_usage = allow_estimated_usage

        logger.info(f"Using {model} (google-genai)")

    @property
    def total_tokens(self) -> dict:
        """Total tokens used across all requests."""
        return {"input": self._input_tokens, "output": self._output_tokens}

    @property
    def total_cost_usd(self) -> float:
        return (
            self._input_tokens * self._pricing.input_cost_per_million / 1_000_000
            + self._output_tokens * self._pricing.output_cost_per_million / 1_000_000
        )

    def _throttle(self) -> None:
        """Enforce minimum request interval."""
        elapsed = time.monotonic() - self._last_call_at
        gap = self._request_interval - elapsed
        if gap > 0:
            time.sleep(gap)

    def complete(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        system: Optional[str] = None,
        logprobs: bool = False,
        top_logprobs: int = 0,
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
        if logprobs:
            raise ValueError(
                f"{self.__class__.__name__} does not support logprobs"
            )
        del top_logprobs

        if self._pricing.max_requests and self._request_count >= self._pricing.max_requests:
            raise RuntimeError(
                f"Reached max_requests={self._pricing.max_requests} for {self.model}. "
                "Use a paid-tier key or lower request volume."
            )
        self._request_count += 1

        self._throttle()
        self._last_call_at = time.monotonic()

        try:
            # Build request with system prompt if provided
            if system:
                full_prompt = f"{system}\n\n{prompt}"
            else:
                full_prompt = prompt

            # Call Gemini API
            config_kwargs = {
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            }
            if self._disable_safety_filters:
                config_kwargs["safety_settings"] = [
                    self._types.SafetySetting(**setting) for setting in _SAFETY_OFF
                ]

            response = self.client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=self._types.GenerateContentConfig(**config_kwargs),
            )

            completion = response.text if response.text else ""

            usage = getattr(response, "usage_metadata", None)
            if usage is not None:
                input_tokens = getattr(usage, "prompt_token_count", None) or 0
                output_tokens = getattr(usage, "candidates_token_count", None) or 0
            else:
                if not self._allow_estimated_usage:
                    raise RuntimeError(
                        "Gemini response did not return usage metadata. "
                        "Re-run with allow_estimated_usage=True to accept "
                        "estimated token accounting."
                    )
                logger.warning(
                    "Gemini response omitted usage metadata; estimating token counts."
                )
                input_tokens = len(full_prompt.split())
                output_tokens = len(completion.split())

            self._input_tokens += input_tokens
            self._output_tokens += output_tokens
            cost_usd = (
                input_tokens * self._pricing.input_cost_per_million / 1_000_000
                + output_tokens * self._pricing.output_cost_per_million / 1_000_000
            )

            return CompletionResponse(
                completion=completion.strip(),
                tokens_used={"input": input_tokens, "output": output_tokens},
                cost_usd=cost_usd,
            )

        except Exception as exc:
            logger.error(f"Gemini error: {exc}")
            raise RuntimeError(f"Gemini inference failed: {exc}") from exc
