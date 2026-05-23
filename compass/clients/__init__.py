"""Client abstractions for compass."""
from compass.clients.base import CompletionClient, CompletionResponse
from compass.clients.ollama import OllamaClient
from compass.clients.pricing import ModelPricing, get_pricing

try:
    from compass.clients.openai import OpenAIClient
except ImportError:
    OpenAIClient = None

try:
    from compass.clients.openai_responses import OpenAIResponsesClient
except ImportError:
    OpenAIResponsesClient = None

try:
    from compass.clients.anthropic import AnthropicClient
except ImportError:
    AnthropicClient = None

try:
    from compass.clients.google_ai import GoogleAIClient
except ImportError:
    GoogleAIClient = None

__all__ = [
    "CompletionClient",
    "CompletionResponse",
    "ModelPricing",
    "get_pricing",
    "OllamaClient",
    "OpenAIClient",
    "OpenAIResponsesClient",
    "AnthropicClient",
    "GoogleAIClient",
]
