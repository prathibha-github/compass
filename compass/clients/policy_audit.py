"""Inventory of client-side policy translations and exceptions."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ClientPolicyTranslation:
    """Named client-side behavior that changes or constrains caller intent."""

    adapter: str
    category: str
    trigger: str
    behavior: str
    explicit_to_caller: bool
    tested_by: Tuple[str, ...] = ()


_CLIENT_POLICY_TRANSLATIONS = (
    ClientPolicyTranslation(
        adapter="OpenAIClient",
        category="required_temperature",
        trigger="required_temperature is configured",
        behavior="sends the configured temperature on every request instead of the caller-provided temperature",
        explicit_to_caller=True,
        tested_by=("tests/test_client_features.py", "tests/test_benchmark_runner.py"),
    ),
    ClientPolicyTranslation(
        adapter="OpenAIClient",
        category="retry_backoff",
        trigger="provider rate limits or transient API failures",
        behavior="retries with backoff and may proactively pause on quota headers",
        explicit_to_caller=False,
        tested_by=("tests/test_client_features.py",),
    ),
    ClientPolicyTranslation(
        adapter="OpenAIResponsesClient",
        category="output_token_multiplier",
        trigger="max_output_token_multiplier is configured above 1",
        behavior="multiplies requested max_tokens by the configured explicit output-token multiplier",
        explicit_to_caller=True,
        tested_by=("tests/test_client_conformance.py", "tests/test_client_features.py"),
    ),
    ClientPolicyTranslation(
        adapter="OpenAIResponsesClient",
        category="estimated_usage_fallback",
        trigger="allow_estimated_usage=True and response usage is unavailable",
        behavior="estimates output tokens from completion length and leaves input tokens at 0",
        explicit_to_caller=True,
        tested_by=("tests/test_client_features.py",),
    ),
    ClientPolicyTranslation(
        adapter="OpenAIResponsesClient",
        category="unsupported_temperature",
        trigger="temperature is not 0.0",
        behavior="raises ValueError because the adapter does not forward temperature to the Responses API",
        explicit_to_caller=True,
        tested_by=("tests/test_client_features.py",),
    ),
    ClientPolicyTranslation(
        adapter="OpenAIResponsesClient",
        category="unsupported_feature",
        trigger="logprobs requested",
        behavior="raises ValueError because the adapter does not support logprobs",
        explicit_to_caller=True,
        tested_by=("tests/test_client_conformance.py",),
    ),
    ClientPolicyTranslation(
        adapter="AnthropicClient",
        category="retry_backoff",
        trigger="provider rate limits or transient API failures",
        behavior="retries with backoff before failing the request",
        explicit_to_caller=False,
        tested_by=("tests/test_client_features.py",),
    ),
    ClientPolicyTranslation(
        adapter="AnthropicClient",
        category="unsupported_feature",
        trigger="logprobs requested",
        behavior="raises ValueError because the adapter does not support logprobs",
        explicit_to_caller=True,
        tested_by=("tests/test_client_conformance.py",),
    ),
    ClientPolicyTranslation(
        adapter="GoogleAIClient",
        category="max_request_cap",
        trigger="request_count reaches pricing.max_requests",
        behavior="fails early using a client-side free-tier request ceiling",
        explicit_to_caller=False,
        tested_by=("tests/test_client_features.py",),
    ),
    ClientPolicyTranslation(
        adapter="GoogleAIClient",
        category="usage_fallback",
        trigger="usage_metadata is unavailable",
        behavior="estimates input and output tokens from word counts",
        explicit_to_caller=False,
        tested_by=(),
    ),
    ClientPolicyTranslation(
        adapter="GoogleAIClient",
        category="optional_safety_override",
        trigger="disable_safety_filters=True",
        behavior="turns off Gemini safety settings in the generated request config",
        explicit_to_caller=True,
        tested_by=("tests/test_client_features.py",),
    ),
    ClientPolicyTranslation(
        adapter="GoogleAIClient",
        category="unsupported_feature",
        trigger="logprobs requested",
        behavior="raises ValueError because the adapter does not support logprobs",
        explicit_to_caller=True,
        tested_by=("tests/test_client_conformance.py",),
    ),
    ClientPolicyTranslation(
        adapter="OllamaClient",
        category="prompt_wrapping",
        trigger="system is provided",
        behavior="wraps system content into the generated prompt with <system> tags",
        explicit_to_caller=False,
        tested_by=("tests/test_client_conformance.py",),
    ),
    ClientPolicyTranslation(
        adapter="OllamaClient",
        category="token_estimation",
        trigger="all requests",
        behavior="estimates input and output tokens from word counts because the API does not return usage",
        explicit_to_caller=False,
        tested_by=("tests/test_client_conformance.py",),
    ),
    ClientPolicyTranslation(
        adapter="OllamaClient",
        category="unsupported_feature",
        trigger="logprobs requested",
        behavior="raises ValueError because the adapter does not support logprobs",
        explicit_to_caller=True,
        tested_by=("tests/test_client_conformance.py",),
    ),
)


def list_client_policy_translations() -> Tuple[ClientPolicyTranslation, ...]:
    """Return the current audited inventory of adapter-side policy translations."""
    return _CLIENT_POLICY_TRANSLATIONS
