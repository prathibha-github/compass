"""Client abstractions for compass."""
from compass.clients.base import CompletionClient, CompletionResponse
from compass.clients.ollama import OllamaClient

try:
    from compass.clients.openai import OpenAIClient
except ImportError:
    OpenAIClient = None

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
    "OllamaClient",
    "OpenAIClient",
    "AnthropicClient",
    "GoogleAIClient",
]
