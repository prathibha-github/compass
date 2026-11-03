"""Requested and provider-facing settings, recorded without making API calls."""


def request_settings(provider, model, max_tokens, temperature, *, required_temperature=None,
                     token_multiplier=1, temperature_supported=True, **policy):
    return {
        "version": 1, "verified": True, "provider": provider, "model": model,
        "requested": {"max_tokens": max_tokens, "temperature": temperature},
        "effective": {
            "max_tokens": max_tokens * token_multiplier,
            "temperature": (required_temperature if required_temperature is not None else temperature)
                if temperature_supported else None,
            "seed": None,
        },
        "policy": {"token_multiplier": token_multiplier,
                   "required_temperature": required_temperature,
                   "temperature_supported": temperature_supported, **policy},
    }


def client_request_settings(client, max_tokens, temperature):
    method = getattr(type(client), "request_settings", None)
    if method is None:
        return {"version": 1, "verified": False,
                "requested": {"max_tokens": max_tokens, "temperature": temperature},
                "effective": None}
    return client.request_settings(max_tokens, temperature)


def default_request_settings(model, max_tokens, temperature, *, responses_api=False):
    """Contract for the built-in runner factories; custom clients remain unverified."""
    if model.startswith("gpt-5") and responses_api:
        return request_settings("openai_responses", model, max_tokens, temperature,
                                required_temperature=1.0, token_multiplier=10)
    if model.startswith(("gpt", "o4")):
        return request_settings("openai", model, max_tokens, temperature,
            required_temperature=1.0 if model.startswith(("gpt-5", "o4")) else None)
    if model.startswith("claude"):
        from compass.clients.anthropic import _supports_temperature
        return request_settings("anthropic", model, max_tokens, temperature,
                                temperature_supported=_supports_temperature(model))
    if model.startswith("gemini"):
        return request_settings("google_ai", model, max_tokens, temperature,
                                disable_safety_filters=False)
    return request_settings("ollama", model, max_tokens, temperature)
