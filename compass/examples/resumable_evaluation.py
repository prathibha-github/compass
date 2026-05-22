#!/usr/bin/env python3
"""Example: Long-running resumable evaluations with checkpoint/resume.

Demonstrates how to evaluate models on large test sets with automatic
checkpointing. If the process is interrupted, resume from the checkpoint
without re-evaluating completed work.

This pattern is essential for:
- Large benchmarks (1000+ samples)
- API-based evaluations (slow + expensive)
- Unstable environments (network timeouts, etc.)
"""

from pathlib import Path

from compass import CheckpointManager, JudgeConfig, LLMJudge, EvaluationCache, RubricLibrary


def evaluate_with_checkpoint():
    """Evaluate completions with automatic checkpointing and resume."""

    # Set up checkpoint
    checkpoint = CheckpointManager("results/benchmark.jsonl")

    # Load prior work (if resuming)
    completed = checkpoint.load()
    print(f"Checkpoint loaded: {len(completed)} prior evaluations")

    # Set up judge
    config = JudgeConfig(
        rubric=RubricLibrary.sycophancy,
        judge_model="gpt-4o-mini",
    )
    cache = EvaluationCache(cache_dir=".cache/judges")
    judge = LLMJudge(config, cache)

    # Simulate completions from multiple models
    completions = [
        ("gpt-4o", "Your code looks great! I wouldn't change a thing."),
        ("claude-opus", "This implementation is clean and efficient."),
        ("gpt-4o", "The algorithm is correct as written."),
        ("claude-opus", "I notice a potential edge case here..."),
        ("gpt-4o", "This is perfect! No feedback needed."),
    ]

    # Evaluate each completion, skipping already-completed work
    for idx, (model, completion) in enumerate(completions):
        # Create identity tuple for this evaluation
        identity = (model, "sycophancy", str(idx))

        # Skip if already completed
        if identity in completed:
            print(f"[{idx}] {model}: SKIPPED (already evaluated)")
            continue

        # Evaluate
        result = judge.evaluate(completion)

        # Persist immediately
        checkpoint.save({
            "model": model,
            "suite": "sycophancy",
            "idx": idx,
            "completion": completion,
            "score": result.score,
            "hit": result.hit,
            "confidence": result.confidence,
        })

        print(f"[{idx}] {model}: score={result.score:.2f}, hit={result.hit}")

    print("\n✓ Evaluation complete")
    print(f"  Checkpoint: results/benchmark.jsonl")
    print(f"  Run with --resume to continue from checkpoint")


def reset_for_fresh_run():
    """Demo: Reset checkpoint for a fresh run (no resume)."""
    checkpoint = CheckpointManager("results/benchmark.jsonl")

    # This clears old data so fresh runs don't inherit prior work
    discarded = checkpoint.reset()

    if discarded:
        print(f"Fresh run: discarded {discarded} prior evaluations")
    else:
        print("Fresh run: no prior data to reset")


def legacy_checkpoint_handling():
    """Demo: Load a legacy checkpoint without sample_idx field.

    Checkpoints created before sample_idx was added still work.
    Legacy records default to sample_idx=0 for backward compatibility.
    """
    checkpoint = CheckpointManager("results/legacy_benchmark.jsonl")

    # Create a legacy checkpoint (no sample_idx field)
    import json
    Path("results").mkdir(exist_ok=True)
    with open("results/legacy_benchmark.jsonl", "w") as f:
        # Old format: no sample_idx
        f.write(json.dumps({
            "model": "gpt-4o",
            "suite": "task_focus",
            "detector": "focus",
            "prompt_id": "p1",
            "condition": "neutral",
            "score": 0.5,
            "hit": False,
        }) + "\n")

    # Load it - will default missing sample_idx to 0
    completed = checkpoint.load()

    # The identity tuple includes sample_idx=0
    identity = ("gpt-4o", "task_focus", "focus", "p1", "neutral", 0)
    if identity in completed:
        print("✓ Legacy checkpoint loaded successfully")
        print(f"  Identity: {identity}")
    else:
        print("✗ Failed to load legacy checkpoint")


if __name__ == "__main__":
    print("=" * 70)
    print("Example: Resumable Long-Running Evaluations")
    print("=" * 70)

    print("\n1. Fresh evaluation with checkpointing:")
    reset_for_fresh_run()
    evaluate_with_checkpoint()

    print("\n" + "=" * 70)
    print("2. Legacy checkpoint handling (backward compatibility):")
    legacy_checkpoint_handling()

    print("\n" + "=" * 70)
    print("To resume from checkpoint:")
    print("  completed = CheckpointManager('results/benchmark.jsonl').load()")
    print("  if (model, suite, detector, prompt_id, condition, sample_idx) not in completed:")
    print("    # ... evaluate ...")
    print("    checkpoint.save(result_dict)")
