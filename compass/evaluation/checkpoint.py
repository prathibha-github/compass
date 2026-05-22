"""Checkpoint and resume system for long-running evaluations."""
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manage checkpoints for resumable evaluations.

    Enables long-running evaluation batches to be interrupted and resumed
    without re-evaluating completed work. Uses sample-level granularity so
    partial detector failures can be retried independently.

    Supports backward compatibility: checkpoints written without sample_idx
    are treated as sample_idx=0.

    Usage:
        checkpoint = CheckpointManager("results/benchmark.jsonl")
        completed = checkpoint.load()  # Resume: load prior work
        # ... evaluate ...
        checkpoint.save(result_dict)  # Persist incrementally

    Checkpoint format (JSONL):
        {"model": "gpt-4o", "suite": "task_focus", "detector": "focus",
         "prompt_id": "p1", "condition": "neutral", "sample_idx": 0,
         "score": 0.75, "hit": true, ...}
    """

    def __init__(self, checkpoint_path: str):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_path: Path to JSONL checkpoint file
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> Set[Tuple[str, str, str, str, str, int]]:
        """
        Load completed evaluation identities from checkpoint.

        Returns set of (model, suite, detector, prompt_id, condition, sample_idx)
        tuples representing completed work. Checkpoints written before sample_idx
        existed default to sample_idx=0 for backward compatibility.

        Returns:
            Set of completed identity tuples

        Logs:
            - Warning for each malformed JSON line
            - Warning for each invalid record (missing required fields)
            - Warning for legacy records without sample_idx
        """
        completed = set()
        if not self.checkpoint_path.exists():
            return completed

        legacy_records = 0
        with open(self.checkpoint_path) as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    result = json.loads(line)

                    # Extract sample_idx with backward compatibility
                    sample_idx = result.get("sample_idx", 0)
                    missing_sample_idx = "sample_idx" not in result

                    # Handle legacy string serialization
                    if not isinstance(sample_idx, int):
                        sample_idx = int(sample_idx)

                    if missing_sample_idx:
                        legacy_records += 1

                    # Build identity tuple
                    key = (
                        result["model"],
                        result["suite"],
                        result["detector"],
                        result["prompt_id"],
                        result["condition"],
                        sample_idx,
                    )
                    completed.add(key)

                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSON at checkpoint line {line_num}")
                    continue
                except (KeyError, TypeError, ValueError):
                    logger.warning(f"Skipping invalid checkpoint record at line {line_num}")
                    continue

        if legacy_records:
            logger.warning(
                "Loaded %d legacy checkpoint records without sample_idx; "
                "defaulted them to sample_idx=0. This may collapse multi-sample runs "
                "if you resume without explicitly tracking samples.",
                legacy_records,
            )

        return completed

    def save(self, result: Dict[str, Any]) -> None:
        """
        Append a single evaluation result to checkpoint.

        Writes immediately (not buffered) so data persists even if process
        is interrupted.

        Args:
            result: Dict with keys: model, suite, detector, prompt_id,
                   condition, sample_idx, score, hit, confidence,
                   rationale, input_tokens, output_tokens
        """
        with open(self.checkpoint_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    def reset(self) -> int:
        """
        Clear existing checkpoint (for fresh non-resume runs).

        Returns:
            Number of evaluations discarded
        """
        if not self.checkpoint_path.exists():
            return 0

        with open(self.checkpoint_path) as f:
            discarded = sum(1 for line in f if line.strip())

        self.checkpoint_path.write_text("")

        if discarded:
            logger.warning(
                "Resetting existing checkpoint for fresh run; "
                "discarding %d prior evaluations",
                discarded,
            )

        return discarded

    def is_sample_complete(
        self,
        completed: Set[Tuple[str, str, str, str, str, int]],
        model: str,
        suite: str,
        detector_names: list,
        prompt_id: str,
        condition: str,
        sample_idx: int,
    ) -> bool:
        """
        Check if all detectors for a sample are complete.

        Allows partial detector failures to be retried independently.
        A sample is only skipped if ALL detectors for that sample are done.

        Args:
            completed: Set of completed identity tuples
            model: Model name
            suite: Suite name
            detector_names: List of detector names to check
            prompt_id: Prompt ID
            condition: Condition name
            sample_idx: Sample index

        Returns:
            True if all detectors for this sample are complete
        """
        return all(
            (model, suite, detector_name, prompt_id, condition, sample_idx)
            in completed
            for detector_name in detector_names
        )
