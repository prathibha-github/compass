"""Shared per-model pricing table. Exact model names are the source of truth."""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelPricing:
    name: str
    input_cost_per_million: float   # USD per 1M input tokens
    output_cost_per_million: float  # USD per 1M output tokens
    max_requests: Optional[int] = None  # rate-limit ceiling (e.g. Gemini free tier RPM)


PRICING_TABLE: dict = {
    "gpt-4o-mini":            ModelPricing("gpt-4o-mini",            0.15,   0.60),
    "gpt-4o":                 ModelPricing("gpt-4o",                 2.50,  10.00),
    "gpt-4-turbo":            ModelPricing("gpt-4-turbo",           10.00,  30.00),
    "gpt-4.1-nano":           ModelPricing("gpt-4.1-nano",           0.10,   0.40),
    "gpt-4.1-mini":           ModelPricing("gpt-4.1-mini",           0.40,   1.60),
    "gpt-5.4-mini":           ModelPricing("gpt-5.4-mini",           0.40,   1.60),
    "o4-mini":                ModelPricing("o4-mini",                1.10,   4.40),
    "claude-haiku-4-5-20251001": ModelPricing("claude-haiku-4-5-20251001", 1.00, 5.00),
    "claude-haiku-4-5":       ModelPricing("claude-haiku-4-5",       1.00,   5.00),
    "claude-sonnet-4-6":      ModelPricing("claude-sonnet-4-6",      3.00,  15.00),
    "claude-opus-4-6":        ModelPricing("claude-opus-4-6",        15.00, 75.00),
    "claude-opus-4-7":        ModelPricing("claude-opus-4-7",        5.00,  25.00),
    "gemini-2.5-flash":       ModelPricing("gemini-2.5-flash",       0.075,  0.30, max_requests=15),
    "gemini-2.0-flash":       ModelPricing("gemini-2.0-flash",       0.075,  0.30, max_requests=15),
    "gemini-1.5-pro":         ModelPricing("gemini-1.5-pro",         1.25,   5.00, max_requests=5),
    "gemini-1.5-flash":       ModelPricing("gemini-1.5-flash",       0.075,  0.30, max_requests=15),
    "gemini-1.5-flash-8b":    ModelPricing("gemini-1.5-flash-8b",    0.0375, 0.15, max_requests=15),
}

_PROVIDER_DEFAULTS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "google":    "gemini-2.0-flash",
    "openai":    "gpt-4o-mini",
}


def _provider(model: str) -> str:
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("gemini-"):
        return "google"
    return "openai"


def get_pricing(model: str) -> ModelPricing:
    """Return pricing for a model; unknown models fall back to a conservative same-provider default."""
    if model in PRICING_TABLE:
        return PRICING_TABLE[model]
    return PRICING_TABLE[_PROVIDER_DEFAULTS[_provider(model)]]
