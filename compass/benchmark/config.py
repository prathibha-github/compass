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
    prefixes = sorted(
        (prefix for prefix in token_budgets if prefix != "default"),
        key=len,
        reverse=True,
    )
    for prefix in prefixes:
        budget = token_budgets[prefix]
        if model.startswith(prefix):
            return int(budget)
    return int(token_budgets["default"])
