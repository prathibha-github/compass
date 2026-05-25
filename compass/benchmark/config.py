"""Benchmark configuration defaults."""

from types import MappingProxyType
from typing import Mapping

DEFAULT_TOKEN_BUDGETS = MappingProxyType(
    {
        "default": 150,
        "gemini": 2000,
    }
)

LEGACY_TOKEN_CAP_FALLBACK = DEFAULT_TOKEN_BUDGETS["default"]


def default_max_tokens_for_model(
    model: str,
    token_budgets: Mapping[str, int] = DEFAULT_TOKEN_BUDGETS,
) -> int:
    """Return the configured default max token budget for a model."""
    for prefix, budget in token_budgets.items():
        if prefix == "default":
            continue
        if model.startswith(prefix):
            return int(budget)
    return int(token_budgets["default"])
