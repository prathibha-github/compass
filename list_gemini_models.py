#!/usr/bin/env python3
"""List available Gemini models from your Google API key."""
import os
import sys

try:
    from google import genai
except ImportError:
    print("Error: google-genai not installed")
    print("Install with: pip install google-genai")
    sys.exit(1)

# Get API key from environment
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY environment variable not set")
    print("\nSet your API key:")
    print("  export GOOGLE_API_KEY='your-key-from-google-ai-studio'")
    print("\nThen run this script again")
    sys.exit(1)

print("Fetching available Gemini models...")
print()

try:
    client = genai.Client(api_key=api_key)
    models = client.models.list()

    print("✓ Available models for your account:")
    print()
    for model in models:
        print(f"  {model.name}")

    print()
    print("Pick one and run:")
    print("  python examples/local_benchmark.py --models <model-name> --judge-model gpt-4o-mini --samples 5")

except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
