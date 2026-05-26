"""Client abstractions for compass."""
from compass.clients.base import CompletionClient, CompletionResponse
from compass.clients.ollama import OllamaClient
from compass.clients.policy_audit import (
    ClientPolicyTranslation,
    list_client_policy_translations,
)
from compass.clients.pricing import ModelPricing, get_pricing


def _missing_client_class(client_name: str, package_name: str, extra_name: str, exc: Exception):
    class _MissingClient:
        def __init__(self, *_args, **_kwargs):
            raise ImportError(
                f"{client_name} requires the optional dependency '{package_name}'. "
                f"Install with: pip install compass-eval[{extra_name}]"
            ) from exc

    _MissingClient.__name__ = client_name
    _MissingClient.__qualname__ = client_name
    _MissingClient.__doc__ = (
        f"Placeholder for {client_name}. Install compass-eval[{extra_name}] to use it."
    )
    return _MissingClient


try:
    from compass.clients.openai import OpenAIClient
except ImportError as exc:
    OpenAIClient = _missing_client_class("OpenAIClient", "openai", "openai", exc)

try:
    from compass.clients.openai_responses import OpenAIResponsesClient
except ImportError as exc:
    OpenAIResponsesClient = _missing_client_class(
        "OpenAIResponsesClient", "openai", "openai", exc
    )

try:
    from compass.clients.anthropic import AnthropicClient
except ImportError as exc:
    AnthropicClient = _missing_client_class(
        "AnthropicClient", "anthropic", "anthropic", exc
    )

try:
    from compass.clients.google_ai import GoogleAIClient
except ImportError as exc:
    GoogleAIClient = _missing_client_class(
        "GoogleAIClient", "google-genai", "google", exc
    )

__all__ = [
    "CompletionClient",
    "CompletionResponse",
    "ClientPolicyTranslation",
    "ModelPricing",
    "get_pricing",
    "list_client_policy_translations",
    "OllamaClient",
    "OpenAIClient",
    "OpenAIResponsesClient",
    "AnthropicClient",
    "GoogleAIClient",
]
