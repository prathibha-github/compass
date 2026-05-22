# Checkpoint System: Resumable Long-Running Evaluations

## Overview

The CheckpointManager enables evaluations to be interrupted and resumed without re-evaluating completed work. This is essential for:

- **Large benchmarks** (1000+ samples × multiple models)
- **API-based evaluations** (slow + expensive per sample)
- **Unreliable environments** (network timeouts, server restarts)
- **Incremental evaluation** (add new models/suites without re-running old ones)

## How It Works

### Basic Concept

Each evaluation has an **identity tuple**:
```
(model, suite, detector, prompt_id, condition, sample_idx)
```

This tuple uniquely identifies one evaluation. When interrupted and resumed:

1. **Load** all completed identities from checkpoint file
2. **Skip** evaluations whose identities are already in completed set
3. **Evaluate** remaining work
4. **Save** results immediately (not buffered)

### Checkpoint Format

The checkpoint is a **JSONL file** (one JSON object per line):

```json
{
  "model": "gpt-4o",
  "suite": "task_focus",
  "detector": "task_focus",
  "prompt_id": "bug_fix",
  "condition": "default",
  "sample_idx": 0,
  "score": 0.75,
  "hit": true,
  "confidence": 0.95,
  "rationale": "Response stayed focused on the bug fix.",
  "input_tokens": 100,
  "output_tokens": 45
}
```

Each line is **appended** immediately after evaluation (not buffered), so data persists even if the process crashes.

## Usage

### Fresh Run (Start from Scratch)

```python
from compass import CheckpointManager

checkpoint = CheckpointManager("results/benchmark.jsonl")

# Reset old data (if any)
checkpoint.reset()

# Load completed (empty on fresh run)
completed = checkpoint.load()

# Evaluate and save
for model in models:
    for prompt in prompts:
        # Skip already-completed work
        identity = (model, "task_focus", "detector1", prompt.id, "neutral", 0)
        if identity in completed:
            continue
        
        # Evaluate
        result = judge.evaluate(completion)
        
        # Save immediately
        checkpoint.save({
            "model": model,
            "suite": "task_focus",
            "detector": "detector1",
            "prompt_id": prompt.id,
            "condition": "neutral",
            "sample_idx": 0,
            "score": result.score,
            "hit": result.hit,
        })
```

### Resume from Checkpoint

```python
checkpoint = CheckpointManager("results/benchmark.jsonl")

# Load prior work
completed = checkpoint.load()
print(f"Resuming: {len(completed)} prior evaluations")

# Continue evaluation
# (skip logic identical to fresh run above)
```

## Backward Compatibility

### Legacy Checkpoints

Checkpoints written **before sample_idx existed** are still readable. Legacy records default to `sample_idx=0`:

```python
# Old checkpoint (no sample_idx field)
{"model": "gpt-4o", "suite": "task_focus", ...}

# When loaded, treated as:
("gpt-4o", "task_focus", ..., 0)  # sample_idx defaults to 0
```

**Warning:** If you resume an old checkpoint, sample indices will collapse:
- Sample 0, 1, 2 all treated as sample 0
- Subsequent samples won't resume correctly

**Best practice:** When upgrading to a new evaluation system with sample-level tracking, start a fresh checkpoint.

### Type Coercion

String sample_idx values are automatically converted to int:

```python
# Manually-created checkpoint (sample_idx as string)
{"model": "gpt-4o", "sample_idx": "0", ...}

# Automatically converted to int(0)
# Logs: "Loaded 5 legacy checkpoint records without sample_idx"
```

## Sample-Level Granularity

The identity tuple includes **sample_idx** (0-indexed sample number within a condition):

```
(model, suite, detector, prompt_id, condition, sample_idx=0)  # Sample 0
(model, suite, detector, prompt_id, condition, sample_idx=1)  # Sample 1
(model, suite, detector, prompt_id, condition, sample_idx=2)  # Sample 2
```

This enables **detector-level partial retry**:

```python
# All detectors for sample 0 are complete?
all_done = checkpoint.is_sample_complete(
    completed,
    model="gpt-4o",
    suite="task_focus",
    detector_names=["detector1", "detector2"],
    prompt_id="p1",
    condition="neutral",
    sample_idx=0,
)

if all_done:
    skip_sample()  # All detectors done
else:
    evaluate_missing_detectors()  # Some failed, retry them
```

## Fresh Run vs Resume

### Fresh Run (Non-Resume)

```python
checkpoint = CheckpointManager("results/benchmark.jsonl")
checkpoint.reset()  # Clears old data, logs count
completed = checkpoint.load()  # Empty set
# ... evaluate ...
```

Behavior:
- If file exists, count its lines and log: `"Resetting existing checkpoint for fresh run; discarding 150 prior evaluations"`
- Truncate file to empty
- Load returns empty set
- New evaluations start fresh

### Resume

```python
checkpoint = CheckpointManager("results/benchmark.jsonl")
# NO reset() call
completed = checkpoint.load()  # Loads prior work
# ... evaluate ...
```

Behavior:
- Load returns identities of all prior evaluations
- Skip matching identities
- Append new results to same file

## Error Handling

### Malformed JSON

If a line in the checkpoint is invalid JSON:
```
{"model": "gpt-4o", "suite": ...  ← Truncated, invalid

Logs: "Skipping malformed JSON at checkpoint line 42"
Result: Line is skipped, evaluation continues
```

### Missing Required Fields

If a record is missing required fields (`model`, `suite`, etc.):
```
{"model": "gpt-4o"}  ← No suite, detector, etc.

Logs: "Skipping invalid checkpoint record at line 42"
Result: Line is skipped, evaluation continues
```

### Type Errors

If a field has wrong type:
```
{"model": "gpt-4o", ..., "sample_idx": "invalid"}

Logs: "Skipping invalid checkpoint record at line 42"
Result: Line is skipped, evaluation continues
```

## Best Practices

### 1. Always Check Completion Before Evaluating

```python
# ✓ Good
identity = (model, suite, detector, prompt_id, condition, sample_idx)
if identity not in completed:
    # evaluate and save
```

```python
# ✗ Bad (will re-evaluate)
result = judge.evaluate(completion)  # Always evaluates
checkpoint.save(result)
```

### 2. Use Fixed Checkpoint Path

```python
# ✓ Good - fixed filename
checkpoint = CheckpointManager("results/benchmark.jsonl")

# ✗ Bad - timestamp changes each run
import time
checkpoint = CheckpointManager(f"results/benchmark_{time.time()}.jsonl")
# → Different file each run, no resume!
```

### 3. Save Immediately After Evaluation

```python
# ✓ Good - saves immediately
for item in items:
    result = evaluate(item)
    checkpoint.save(result)

# ✗ Bad - buffers until end
results = []
for item in items:
    results.append(evaluate(item))
for r in results:
    checkpoint.save(r)  # Loss if interrupted before this
```

### 4. Handle Legacy Checkpoints

```python
# Log migration advice
if legacy_records > 0:
    print(f"⚠️  {legacy_records} legacy records loaded")
    print("    If resuming, samples may collapse.")
    print("    Consider starting fresh checkpoint for new runs.")
```

## Checkpoint Data Loss Scenarios

### Scenario 1: Interrupted Before Save
```
Evaluation completes → Hit Ctrl+C → Result lost (not saved)

Mitigated by: Save immediately after each evaluation
```

### Scenario 2: File System Corruption
```
Power loss / disk error → Checkpoint file corrupted

Mitigated by: Backup checkpoint before large runs
    cp results/benchmark.jsonl results/benchmark.jsonl.bak
```

### Scenario 3: Wrong Checkpoint Path (Accidentally Fresh Run)
```
Was evaluating: results/benchmark.jsonl
Accidentally ran: python script.py results/other.jsonl

Result: New checkpoint created, prior work untouched but not resumed
```

**Prevention:** Use environment variables or config files for paths

```python
import os
checkpoint_path = os.getenv("CHECKPOINT_PATH", "results/benchmark.jsonl")
checkpoint = CheckpointManager(checkpoint_path)
```

## Advanced: Custom Identity Tuples

The identity tuple is arbitrary—you can use any fields:

```python
# Default (recommended)
identity = (model, suite, detector, prompt_id, condition, sample_idx)

# Alternative: include task type
identity = (model, suite, detector, prompt_id, condition, task_type, sample_idx)

# Alternative: segmented evaluation
identity = (model, suite, detector, segment, sample_idx)
```

The important part: **identities must be unique across your evaluation set** and **consistent across runs**.

## Checkpoint vs. Database

| Aspect | Checkpoint (JSONL) | Database |
|--------|-------------------|----------|
| **Format** | Text lines (JSON) | Structured tables |
| **Read speed** | O(n) - scan all lines | O(log n) - indexed |
| **Append speed** | O(1) - fast | O(log n) - indexed |
| **Corruption recovery** | Skip bad lines | Crash/rollback |
| **Storage** | Human-readable | Opaque binary |
| **Portability** | Works anywhere | DB-specific |

**When to use checkpoint:** Streaming evaluations, simple resume capability, portable results

**When to use database:** Complex queries, statistics, large-scale production systems

## Testing

```python
import tempfile
from pathlib import Path

# Test fresh run + resume cycle
with tempfile.TemporaryDirectory() as tmpdir:
    checkpoint_path = Path(tmpdir) / "test.jsonl"
    
    # Fresh run
    cp = CheckpointManager(str(checkpoint_path))
    cp.reset()
    completed = cp.load()
    assert len(completed) == 0
    
    cp.save({"model": "gpt-4o", "suite": "task_focus", ...})
    
    # Resume
    cp2 = CheckpointManager(str(checkpoint_path))
    completed2 = cp2.load()
    assert len(completed2) == 1
```
