# Client Adapter Contract

Compass uses provider-specific adapters behind one shared completion interface.
Benchmarks, judges, and examples should call these adapters, not provider SDKs
directly.

## Shared Interface

All adapters implement `CompletionClient.complete(...)`:

```python
from compass.clients.base import CompletionClient, CompletionResponse

response = client.complete(
    prompt="Explain the error",
    max_tokens=180,
    temperature=0.0,
    system="You are a strict evaluator.",
    logprobs=False,
    top_logprobs=0,
)
```

Every adapter must return a `CompletionResponse` with:
- `completion`: non-empty text
- `tokens_used`: `{"input": int, "output": int}` when token accounting is available
- `cost_usd`: floating-point cost estimate for that call
- `logprobs`: provider-native top-token data only when supported

The shared conformance suite lives in `tests/test_client_conformance.py`.

## Behavior Contract

The conformance tests enforce these rules:

- All adapters expose the same `complete(...)` signature as `CompletionClient`.
- `system` instructions flow through the adapter-specific request shape.
- `max_tokens` and `temperature` are translated into the provider request in a
  predictable way.
- Successful responses return normalized token accounting and per-call cost.
- Adapters maintain cumulative usage via `total_tokens` and `total_cost_usd`.
- Unsupported optional features fail explicitly. For example, adapters that do
  not support `logprobs` must raise `ValueError` rather than silently ignoring
  the request.

If a provider cannot supply an exact field, the adapter must still make the
behavior explicit. Ollama, for example, estimates token counts because its API
does not return exact usage metrics.

## Provider-Specific Notes

### OpenAI Chat Completions

- Uses Chat Completions request/response objects.
- Handles provider rate limits inside the adapter with retry and backoff.
- Forces temperature to `1.0` for model families that require it.

### OpenAI Responses API

- Uses the Responses API instead of Chat Completions.
- Preserves the shared completion contract even though the upstream request
  shape differs.
- Rejects `logprobs` because the current adapter contract does not support them
  on this path.

### Anthropic

- Maps `system` and `messages` into Anthropic's message format.
- Handles provider rate limits inside the adapter with retry and backoff.
- Rejects unsupported `logprobs` requests explicitly.

### Google AI

- Maps the shared request into `generate_content(...)`.
- Normalizes usage metadata into the shared token-accounting shape.
- Rejects unsupported `logprobs` requests explicitly.

### Ollama

- Runs against a local Ollama server rather than a hosted API.
- Uses adapter-side throttling to avoid hammering the local server.
- Estimates token usage because the API does not provide exact counts.
- Rejects unsupported `logprobs` requests explicitly.

## Benchmark and Judge Integration

The client layer is part of the benchmark contract, not a convenience wrapper.

- `LLMJudge` should always evaluate through a `CompletionClient`.
- Benchmark generation and evaluation should create provider clients through the
  shared benchmark runner helpers, not by importing provider SDKs directly.
- Provider routing is covered in `tests/test_benchmark_runner.py` so benchmark
  paths do not bypass adapter logic.

When adding a new provider:

1. Implement `CompletionClient`.
2. Add conformance coverage in `tests/test_client_conformance.py`.
3. Verify benchmark routing reaches the adapter contract.
4. Document any deliberate deviations, especially around token accounting,
   retries, and unsupported features.
