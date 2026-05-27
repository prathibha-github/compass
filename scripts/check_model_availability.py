#!/usr/bin/env python3
"""Probe each provider API to verify the benchmark models are reachable."""
import sys

ANTHROPIC_MODELS = ["claude-sonnet-4-6", "claude-opus-4-6"]
OPENAI_MODELS = ["gpt-5.4-mini"]
GOOGLE_MODELS = ["gemini-2.5-flash"]


def check_anthropic() -> dict[str, str]:
    try:
        import anthropic
        client = anthropic.Anthropic()
    except Exception as e:
        return {m: f"client error: {e}" for m in ANTHROPIC_MODELS}
    results = {}
    for model in ANTHROPIC_MODELS:
        try:
            client.messages.create(
                model=model,
                max_tokens=5,
                messages=[{"role": "user", "content": "hi"}],
            )
            results[model] = "OK"
        except Exception as e:
            results[model] = f"FAIL: {e}"
    return results


def check_openai() -> dict[str, str]:
    try:
        import openai
        client = openai.OpenAI()
    except Exception as e:
        return {m: f"client error: {e}" for m in OPENAI_MODELS}
    results = {}
    for model in OPENAI_MODELS:
        try:
            client.chat.completions.create(
                model=model,
                max_completion_tokens=5,
                messages=[{"role": "user", "content": "hi"}],
            )
            results[model] = "OK"
        except Exception as e:
            results[model] = f"FAIL: {e}"
    return results


def check_google() -> dict[str, str]:
    try:
        from google import genai
        client = genai.Client()
    except Exception as e:
        return {m: f"client error: {e}" for m in GOOGLE_MODELS}
    results = {}
    for model in GOOGLE_MODELS:
        try:
            client.models.generate_content(model=model, contents="hi")
            results[model] = "OK"
        except Exception as e:
            results[model] = f"FAIL: {e}"
    return results


def main() -> int:
    results = {}
    results.update(check_anthropic())
    results.update(check_openai())
    results.update(check_google())

    failures = 0
    for model, status in results.items():
        symbol = "✓" if status == "OK" else "✗"
        print(f"  {symbol}  {model}: {status}")
        if status != "OK":
            failures += 1

    print()
    if failures:
        print(f"{failures} model(s) unavailable — fix before running benchmarks.")
    else:
        print("All models reachable.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
