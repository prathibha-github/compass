"""OpenAI Responses API client for GPT-5 and compatible models."""
import logging
import random
import re
import time
from typing import Optional

from compass.clients.base import CompletionClient, CompletionResponse
from compass.clients.pricing import get_pricing

logger = logging.getLogger(__name__)


def _parse_reset_seconds(s: str) -> Optional[float]:
    """Parse OpenAI's rate limit reset values: '20s', '1m0s', '1m30.5s'."""
    s = s.strip()
    m = re.fullmatch(r"([\d.]+)s", s)
    if m:
        return float(m.group(1))
    m = re.fullmatch(r"(\d+)m([\d.]+)s", s)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    m = re.fullmatch(r"(\d+)m", s)
    if m:
        return int(m.group(1)) * 60
    try:
        return float(s)
    except ValueError:
        return None


class OpenAIResponsesClient(CompletionClient):
    """OpenAI Responses API client for GPT-5 and compatible models.

    The Responses API does not support logprobs. Callers may opt into a larger
    output-token budget with ``max_output_token_multiplier`` when a model needs
    extra reasoning headroom.

    Usage:
        client = OpenAIResponsesClient(
            model="gpt-5-mini",
            max_output_token_multiplier=10,
        )
        response = client.complete("What is 2+2?")
        print(response.completion)
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        request_interval: float = 0.0,
        max_output_token_multiplier: int = 1,
        allow_estimated_usage: bool = False,
    ):
        """
        Args:
            model: Model name (e.g., 'gpt-5-mini', 'gpt-5')
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            request_interval: Minimum seconds between API calls (0 = no throttle).
            max_output_token_multiplier: Explicit multiplier applied to requested
                ``max_tokens`` before calling the Responses API.
            allow_estimated_usage: Whether to estimate token usage when the
                Responses API omits usage metadata.
        """
        try:
            import openai as _openai
        except ImportError as e:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            ) from e

        self._openai = _openai
        self.model = model
        self.client = _openai.OpenAI(api_key=api_key, max_retries=0)
        self._pricing = get_pricing(model)
        self._input_tokens = 0
        self._output_tokens = 0
        self._request_interval = request_interval
        self._last_call_at: float = 0.0
        if max_output_token_multiplier < 1:
            raise ValueError("max_output_token_multiplier must be at least 1")
        self._max_output_token_multiplier = max_output_token_multiplier
        self._allow_estimated_usage = allow_estimated_usage

    @property
    def total_tokens(self) -> dict:
        return {"input": self._input_tokens, "output": self._output_tokens}

    @property
    def total_cost_usd(self) -> float:
        return (
            self._input_tokens * self._pricing.input_cost_per_million / 1_000_000
            + self._output_tokens * self._pricing.output_cost_per_million / 1_000_000
        )

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call_at
        gap = self._request_interval - elapsed
        if gap > 0:
            time.sleep(gap)

    def _wait_from_429_headers(self, headers: dict, attempt: int) -> float:
        for key in ("retry-after", "x-ratelimit-reset-requests"):
            val = headers.get(key)
            if val:
                secs = _parse_reset_seconds(val)
                if secs is not None:
                    return secs + random.uniform(0.5, 2.0)
        return min(15 * (2 ** attempt), 60) + random.uniform(0, 5)

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
        Generate completion via OpenAI Responses API with retry/backoff on 429.

        The configured ``max_output_token_multiplier`` is applied to ``max_tokens``
        before making the request. Non-zero temperatures are rejected because
        the Responses API path does not forward temperature. Usage metadata is
        required unless the caller explicitly allows estimated usage.

        Args:
            prompt: Input prompt
            max_tokens: Maximum output tokens before applying the configured
                output-token multiplier
            temperature: Must remain 0.0 for this adapter path
            system: Optional system/instructions prompt

        Returns:
            CompletionResponse with completion text and token counts

        Raises:
            RuntimeError: If the API call fails after all retries
        """
        if logprobs:
            raise ValueError(
                f"{self.__class__.__name__} does not support logprobs"
            )
        del top_logprobs
        if temperature != 0.0:
            raise ValueError(
                f"{self.__class__.__name__} does not support temperature overrides"
            )

        actual_max_tokens = max_tokens * self._max_output_token_multiplier
        request_kwargs = dict(
            model=self.model,
            input=prompt,
            max_output_tokens=actual_max_tokens,
        )
        if system is not None:
            request_kwargs["instructions"] = system

        max_attempts = 10
        for attempt in range(max_attempts):
            self._throttle()
            self._last_call_at = time.monotonic()

            try:
                resp = self.client.responses.create(**request_kwargs)

                completion = resp.output_text or ""
                usage = getattr(resp, "usage", None)
                input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
                output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

                if not completion.strip():
                    logger.warning(
                        "Responses API returned empty output. status=%s, "
                        "incomplete_details=%s, usage=%s, attempt=%d/%d",
                        getattr(resp, "status", "unknown"),
                        getattr(resp, "incomplete_details", None),
                        usage,
                        attempt + 1, max_attempts,
                    )
                    if attempt == max_attempts - 1:
                        raise RuntimeError(
                            f"Responses API returned empty output after {max_attempts} attempts"
                        )
                    time.sleep(min(4 * (2 ** attempt), 30))
                    continue

                if not usage:
                    if not self._allow_estimated_usage:
                        raise RuntimeError(
                            "Responses API did not return usage metadata. "
                            "Re-run with allow_estimated_usage=True to accept "
                            "estimated token accounting."
                        )
                    logger.warning(
                        "Responses API omitted usage metadata; estimating token counts."
                    )
                    output_tokens = len(completion) // 4

                self._input_tokens += input_tokens
                self._output_tokens += output_tokens

                return CompletionResponse(
                    completion=completion,
                    tokens_used={"input": input_tokens, "output": output_tokens},
                    cost_usd=(
                        input_tokens * self._pricing.input_cost_per_million / 1_000_000
                        + output_tokens * self._pricing.output_cost_per_million / 1_000_000
                    ),
                )

            except self._openai.RateLimitError as exc:
                headers: dict = {}
                if hasattr(exc, "response") and exc.response is not None:
                    headers = dict(exc.response.headers)
                wait = self._wait_from_429_headers(headers, attempt)
                logger.warning(
                    "Rate limit (attempt %d/%d) — waiting %.0fs",
                    attempt + 1, max_attempts, wait,
                )
                time.sleep(wait)

            except self._openai.APIError as exc:
                logger.error("API error on attempt %d: %s", attempt + 1, exc)
                if attempt == max_attempts - 1:
                    raise RuntimeError(
                        f"OpenAI Responses API failed for {self.model}: {exc}"
                    ) from exc
                time.sleep(min(4 * (2 ** attempt), 30))

        raise RuntimeError(f"Responses API call failed after {max_attempts} attempts")
