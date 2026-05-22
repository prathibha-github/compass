"""Client abstractions for compass."""
from compass.clients.base import CompletionClient, CompletionResponse
from compass.clients.ollama import OllamaClient

__all__ = ["CompletionClient", "CompletionResponse", "OllamaClient"]
