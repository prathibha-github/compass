"""Anthropic client for LLM inference."""
import logging
import random
import re
import time
from typing import Optional

from compass.clients.base import CompletionClient, CompletionResponse
from compass.clients.pricing import get_pricing

logger = logging.getLogger(__name__)


def _parse_reset_seconds(s: str) -> Optional[float]:
    """Parse Anthropic rate-limit reset values: '20s' or bare seconds."""
    s = s.strip()
    m = re.fullmatch(r"([\d.]+)s", s)
    if m:
        return float(m.group(1))
    try:
        return float(s)
    except ValueError:
        return None


class AnthropicClient(CompletionClient):
    """Client for Anthropic API (Claude models).

    Usage:
        client = AnthropicClient(model="claude-haiku-4-5")
        response = client.complete("What is 2+2?")
        print(response.completion)
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        request_interval: float = 0.0,
    ):
        """
        Args:
            model: Model name (e.g., 'claude-haiku-4-5', 'claude-sonnet-4-6')
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
            request_interval: Minimum seconds between API calls (0 = no throttle).
        """
        try:
            import anthropic as _anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            ) from e

        self._anthropic = _anthropic
        self.model = model
        self.client = _anthropic.Anthropic(api_key=api_key, max_retries=0)
        self._pricing = get_pricing(model)
        self._input_tokens = 0
        self._output_tokens = 0
        self._request_interval = request_interval
        self._last_call_at: float = 0.0

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
        for key in ("retry-after", "anthropic-ratelimit-requests-reset"):
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
        Generate completion via Anthropic API with retry/backoff on 429.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            system: Optional system prompt

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

        max_attempts = 10
        for attempt in range(max_attempts):
            self._throttle()
            self._last_call_at = time.monotonic()

            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system or "",
                    messages=[{"role": "user", "content": prompt}],
                )

                usage = getattr(resp, "usage", None)
                input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
                output_tokens = getattr(usage, "output_tokens", 0) if usage else 0
                self._input_tokens += input_tokens
                self._output_tokens += output_tokens

                chunks = [
                    getattr(block, "text", "")
                    for block in (getattr(resp, "content", []) or [])
                    if getattr(block, "type", None) == "text"
                ]
                completion = "\n".join(chunks).strip()
                if not completion:
                    raise RuntimeError(f"Empty response from {self.model}")

                return CompletionResponse(
                    completion=completion,
                    tokens_used={"input": input_tokens, "output": output_tokens},
                    cost_usd=(
                        input_tokens * self._pricing.input_cost_per_million / 1_000_000
                        + output_tokens * self._pricing.output_cost_per_million / 1_000_000
                    ),
                )

            except self._anthropic.RateLimitError as exc:
                headers: dict = {}
                if hasattr(exc, "response") and exc.response is not None:
                    headers = dict(exc.response.headers)
                wait = self._wait_from_429_headers(headers, attempt)
                logger.warning(
                    "Anthropic rate limit (attempt %d/%d) — waiting %.0fs",
                    attempt + 1, max_attempts, wait,
                )
                time.sleep(wait)

            except self._anthropic.APIError as exc:
                status = getattr(exc, "status_code", None)
                if status is not None and 400 <= status < 500:
                    raise RuntimeError(
                        f"Anthropic inference failed for {self.model}: {exc}"
                    ) from exc
                logger.error("Anthropic API error on attempt %d: %s", attempt + 1, exc)
                if attempt == max_attempts - 1:
                    raise RuntimeError(
                        f"Anthropic inference failed for {self.model}: {exc}"
                    ) from exc
                time.sleep(min(4 * (2 ** attempt), 30))

        raise RuntimeError(f"Anthropic call failed after {max_attempts} attempts")
