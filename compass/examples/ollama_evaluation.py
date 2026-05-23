#!/usr/bin/env python3
"""Example: Local model evaluation with Ollama (free, no API keys).

Demonstrates how to:
- Evaluate models running locally via Ollama
- Compare local models (llama3.1, mistral, phi) to cloud models
- Run the same evaluation across multiple models
- Keep costs to $0 for local models

Prerequisites:
    1. Install Ollama: https://ollama.ai
    2. Pull models: ollama pull llama3.1 mistral phi
    3. Start server: ollama serve
"""

from compass import (
    EvaluationCache,
    JudgeConfig,
    LLMJudge,
    OllamaClient,
    OpenAIClient,
    RubricLibrary,
)


def demo_ollama_client_basic():
    """Basic usage of OllamaClient."""

    print("Basic OllamaClient Usage")
    print("=" * 70)
    print("Prerequisite: ollama serve running at http://localhost:11434\n")

    try:
        # Create client for a local model
        client = OllamaClient(model="llama3.1:latest")

        # Generate a completion
        response = client.complete(
            prompt="What is machine learning?",
            max_tokens=100,
            temperature=0.0,
        )

        print(f"Model:      llama3.1:latest")
        print(f"Completion: {response.completion[:100]}...")
        print(f"Tokens:     {response.tokens_used}")
        print(f"Cost:       ${response.cost_usd} (local = free)")

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure Ollama is running:")
        print("  $ ollama serve")


def demo_multi_model_comparison():
    """Evaluate multiple local models on the same task."""

    print("\nMulti-Model Comparison (Local)")
    print("=" * 70)
    print("Compare three local models on task focus compliance\n")

    models = ["llama3.1:latest", "mistral:latest", "phi:latest"]
    tasks = [
        ("Fix this bug", "function returns None instead of computed value"),
        ("Explain", "why database migrations take time"),
        ("Calculate", "the square root of 144"),
    ]

    print(f"{'Model':<15} {'Task':<15} {'Tokens':<10} {'Cost':<10}")
    print("-" * 50)

    for model_name in models:
        try:
            client = OllamaClient(model=model_name)

            for task_type, prompt in tasks[:1]:  # Just first task for demo
                response = client.complete(
                    prompt=prompt,
                    max_tokens=50,
                    temperature=0.0,
                )

                tokens = response.tokens_used["input"] + response.tokens_used["output"]
                print(f"{model_name:<15} {task_type:<15} {tokens:<10} ${response.cost_usd:<9.4f}")

        except Exception as e:
            print(f"{model_name:<15} ERROR: {e}")

    print("\nTotal cost for all local evaluations: $0.00")


def demo_local_vs_cloud():
    """Compare local models to cloud models."""

    print("\nLocal vs Cloud Models")
    print("=" * 70)
    print("Trade-offs: cost, latency, privacy, model capability\n")

    print("LOCAL MODELS (Ollama)")
    print("-" * 40)
    print("Advantages:")
    print("  ✓ Free (no API costs)")
    print("  ✓ No network latency for inference")
    print("  ✓ Complete privacy (no data sent to cloud)")
    print("  ✓ Works offline")
    print("\nDisadvantages:")
    print("  ✗ Slower (depends on local hardware)")
    print("  ✗ Lower capability (7B-13B models)")
    print("  ✗ GPU/memory requirements")
    print("\nBest for: Budget-conscious evals, privacy-critical, prototyping")

    print("\n" + "=" * 70)
    print("CLOUD MODELS (OpenAI, Anthropic)")
    print("-" * 40)
    print("Advantages:")
    print("  ✓ State-of-the-art capability (GPT-4o, Claude-opus)")
    print("  ✓ Fast inference")
    print("  ✓ No hardware requirements")
    print("  ✓ Instant scaling")
    print("\nDisadvantages:")
    print("  ✗ API costs ($0.01-0.05 per eval)")
    print("  ✗ Network latency")
    print("  ✗ Data privacy (sent to cloud)")
    print("\nBest for: Production evals, high-stakes tasks, best accuracy")


def demo_hybrid_evaluation():
    """Evaluate with both local and cloud models."""

    print("\nHybrid Evaluation Strategy")
    print("=" * 70)
    print("Use local models for initial screening, cloud models for")
    print("high-confidence regions or final ranking.\n")

    print("Stage 1: Screen with local models (FAST, FREE)")
    print("-" * 40)
    print("  • Evaluate all 1000 samples with llama3.1 (~$0)")
    print("  • Filter to top 100 most interesting cases")
    print("\nStage 2: Deep-dive with cloud models (ACCURATE, PAID)")
    print("-" * 40)
    print("  • Evaluate filtered 100 samples with gpt-4o (~$0.50)")
    print("  • Fine-grained ranking and analysis")
    print("\nBenefit: 10-100x cheaper than cloud-only, 10-100x faster than local-only")


def demo_ollama_with_judge():
    """Use Ollama for inference, cloud judge for evaluation."""

    print("\nOllama Generation + Cloud Judge")
    print("=" * 70)
    print("Generate completions locally, evaluate with cloud judge\n")

    print("Workflow:")
    print("  1. Generate completion from llama3.1 (local, free)")
    print("  2. Send completion to gpt-4o-mini (judge, $0.0001)")
    print("  3. Judge returns score + confidence")
    print("\nBenefit: Cheap inference + accurate evaluation\n")

    try:
        # Generate locally
        generator = OllamaClient(model="llama3.1:latest")
        response = generator.complete(
            prompt="What is the capital of France?",
            max_tokens=20,
            temperature=0.0,
        )

        print(f"Generated (local): {response.completion}")
        print(f"Cost so far:       ${response.cost_usd}")

        # Judge evaluation
        config = JudgeConfig(
            rubric=RubricLibrary.truthfulness,
            judge_model="gpt-4o-mini",
        )
        cache = EvaluationCache(cache_dir=".cache/judges")
        judge_client = OpenAIClient(model="gpt-4o-mini")
        judge = LLMJudge(config, judge_client, cache)

        result = judge.evaluate(response.completion)
        print(f"\nJudge evaluation:")
        print(f"  Score:      {result.score:.2f}")
        print(f"  Hit:        {result.hit}")
        print(f"  Cost:       ${result.cost_usd:.4f}")
        print(f"\nTotal cost: ${response.cost_usd + result.cost_usd:.4f} (mostly judge)")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("=" * 70)
    print("Example: Local Model Evaluation with Ollama")
    print("=" * 70)

    demo_ollama_client_basic()
    demo_multi_model_comparison()
    demo_local_vs_cloud()
    demo_hybrid_evaluation()
    demo_ollama_with_judge()

    print("\n" + "=" * 70)
    print("Getting started with Ollama:")
    print("\n  1. Install Ollama")
    print("     $ brew install ollama  # macOS")
    print("\n  2. Pull models")
    print("     $ ollama pull llama3.1")
    print("     $ ollama pull mistral")
    print("\n  3. Start server")
    print("     $ ollama serve")
    print("\n  4. Use in compass")
    print("     client = OllamaClient(model='llama3.1:latest')")
