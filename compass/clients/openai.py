"""OpenAI Chat Completions client for LLM inference."""
import logging
import random
import re
import time
from typing import Optional

from compass.clients.base import CompletionClient, CompletionResponse
from compass.clients.pricing import get_pricing

logger = logging.getLogger(__name__)


def _parse_reset_seconds(s: str) -> Optional[float]:
    """Parse OpenAI's x-ratelimit-reset-* values: '20s', '1m0s', '1m30.5s'."""
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


class OpenAIClient(CompletionClient):
    """Client for OpenAI Chat Completions API (GPT-4, GPT-4o-mini, o4, etc.).

    Usage:
        client = OpenAIClient(model="gpt-4o-mini")
        response = client.complete("What is 2+2?")
        print(response.completion)
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        request_interval: float = 0.0,
        required_temperature: Optional[float] = None,
    ):
        """
        Args:
            model: Model name (e.g., 'gpt-4o-mini', 'gpt-4o', 'o4-mini')
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            request_interval: Minimum seconds between API calls (0 = no throttle).
            required_temperature: Explicit temperature to send on every request.
        """
        try:
            import openai as _openai
        except ImportError as e:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            ) from e

        self._openai = _openai
        self.model = model
        # max_retries=0: the SDK must not absorb 429s before we handle them.
        self.client = _openai.OpenAI(api_key=api_key, max_retries=0)
        self._pricing = get_pricing(model)
        self._input_tokens = 0
        self._output_tokens = 0
        self._request_interval = request_interval
        self._last_call_at: float = 0.0
        self._required_temperature = required_temperature

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

    def _adapt_from_success_headers(self, headers: dict) -> None:
        """Proactively pause when a quota window is nearly exhausted."""
        for quota_type in ("requests", "tokens"):
            remaining = headers.get(f"x-ratelimit-remaining-{quota_type}")
            reset_str = headers.get(f"x-ratelimit-reset-{quota_type}")
            if remaining is None or reset_str is None:
                continue
            try:
                if int(remaining) <= 1:
                    wait = (_parse_reset_seconds(reset_str) or 60) + 1.0
                    logger.info(
                        "Quota for %s nearly exhausted — pausing %.0fs for window reset",
                        quota_type, wait,
                    )
                    time.sleep(wait)
                    break
            except (ValueError, TypeError):
                continue

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
        Generate completion via OpenAI Chat Completions API with retry/backoff on 429.

        If ``required_temperature`` is configured, it overrides the caller-supplied
        ``temperature`` when building the provider request.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            system: Optional system prompt
            logprobs: Whether to request top-token log probabilities for the first token
            top_logprobs: Number of top-token log probabilities to request

        Returns:
            CompletionResponse with completion text and token counts

        Raises:
            RuntimeError: If the API call fails after all retries
        """
        actual_temperature = (
            self._required_temperature
            if self._required_temperature is not None
            else temperature
        )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = dict(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=actual_temperature,
        )
        if logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = top_logprobs

        max_attempts = 10
        for attempt in range(max_attempts):
            self._throttle()
            self._last_call_at = time.monotonic()

            try:
                # with_raw_response is the only way to read response headers from the
                # openai SDK; the parsed response object does not expose them.
                raw = self.client.chat.completions.with_raw_response.create(**kwargs)
                resp = raw.parse()
                headers = dict(raw.headers)

                completion = resp.choices[0].message.content or ""
                if not completion:
                    raise RuntimeError(f"Empty response from {self.model}")

                raw_logprobs = None
                if logprobs:
                    choice_lp = resp.choices[0].logprobs
                    if choice_lp and choice_lp.content:
                        raw_logprobs = choice_lp.content[0].top_logprobs

                usage = resp.usage
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0
                self._input_tokens += input_tokens
                self._output_tokens += output_tokens

                self._adapt_from_success_headers(headers)

                return CompletionResponse(
                    completion=completion,
                    tokens_used={"input": input_tokens, "output": output_tokens},
                    cost_usd=(
                        input_tokens * self._pricing.input_cost_per_million / 1_000_000
                        + output_tokens * self._pricing.output_cost_per_million / 1_000_000
                    ),
                    logprobs=raw_logprobs,
                    finish_reason=str(
                        getattr(resp.choices[0], "finish_reason", "") or ""
                    ),
                )

            except self._openai.RateLimitError as exc:
                headers_429: dict = {}
                if hasattr(exc, "response") and exc.response is not None:
                    headers_429 = dict(exc.response.headers)
                wait = self._wait_from_429_headers(headers_429, attempt)
                logger.warning(
                    "Rate limit (attempt %d/%d) — waiting %.0fs  "
                    "[retry-after=%s  reset-requests=%s]",
                    attempt + 1, max_attempts, wait,
                    headers_429.get("retry-after", "—"),
                    headers_429.get("x-ratelimit-reset-requests", "—"),
                )
                time.sleep(wait)

            except self._openai.APIError as exc:
                status = getattr(exc, "status_code", None)
                if status is not None and 400 <= status < 500:
                    raise RuntimeError(
                        f"OpenAI inference failed for {self.model}: {exc}"
                    ) from exc
                logger.error("OpenAI API error on attempt %d: %s", attempt + 1, exc)
                if attempt == max_attempts - 1:
                    raise RuntimeError(
                        f"OpenAI inference failed for {self.model}: {exc}"
                    ) from exc
                time.sleep(min(4 * (2 ** attempt), 30))

        raise RuntimeError(f"OpenAI call failed after {max_attempts} attempts")
