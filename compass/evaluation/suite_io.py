"""Schema and IO for two-phase tic/style suite evaluation.

A suite run is split into a generation phase and an evaluation phase. The
generation phase persists the model completion for every
(model, suite, prompt, condition, sample) cell so the evaluation phase can score
those saved outputs with any set of detectors, heuristic or LLM-judge, without
re-calling the model. Generation rows live in their own JSONL file and are read
back with :func:`load_suite_generations`; they never pass through
``CheckpointManager`` (which handles the per-detector ``suite_eval`` rows the
evaluation phase writes).
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)

SUITE_GENERATION_SCHEMA_VERSION = 1
SUITE_GENERATION_SCHEMA_VERSION_FIELD = "suite_schema_version"
SUITE_GENERATION_RECORD_TYPE_FIELD = "suite_record_type"
SUITE_GENERATION_RECORD_TYPE = "suite_generation"

_REQUIRED_FIELDS = ("model", "suite", "prompt_id", "condition", "completion")
_SUPPORTED_INPUT_SCHEMA_VERSIONS = frozenset(
    {0, SUITE_GENERATION_SCHEMA_VERSION}
)


def _coerce_sample_idx(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    return int(value)


def migrate_suite_generation_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a suite generation record to the current schema shape."""
    if not isinstance(record, dict):
        raise TypeError("suite generation record must be a dict")

    data = dict(record)
    missing = [key for key in _REQUIRED_FIELDS if key not in data]
    if missing:
        raise ValueError(f"suite generation record missing fields: {missing}")

    data["sample_idx"] = _coerce_sample_idx(data.get("sample_idx", 0))

    schema_version = int(
        data.get(SUITE_GENERATION_SCHEMA_VERSION_FIELD, 0)
    )
    if schema_version not in _SUPPORTED_INPUT_SCHEMA_VERSIONS:
        raise ValueError(
            f"unsupported suite generation schema version: {schema_version}"
        )

    record_type = data.get(
        SUITE_GENERATION_RECORD_TYPE_FIELD, SUITE_GENERATION_RECORD_TYPE
    )
    if record_type != SUITE_GENERATION_RECORD_TYPE:
        raise ValueError(
            f"expected suite generation record type, got {record_type!r}"
        )

    data[SUITE_GENERATION_SCHEMA_VERSION_FIELD] = SUITE_GENERATION_SCHEMA_VERSION
    data[SUITE_GENERATION_RECORD_TYPE_FIELD] = SUITE_GENERATION_RECORD_TYPE
    return data


def suite_generation_identity(record: Dict[str, Any]) -> Tuple[Any, ...]:
    """Canonical identity tuple for a suite generation record."""
    row = migrate_suite_generation_record(record)
    return (
        row["model"],
        row["suite"],
        row["prompt_id"],
        row["condition"],
        row["sample_idx"],
    )


def append_suite_generation(path: Path, record: Dict[str, Any]) -> None:
    """Append one normalized generation record to the JSONL file."""
    normalized = migrate_suite_generation_record(record)
    with open(path, "a") as f:
        f.write(json.dumps(normalized) + "\n")


def load_suite_generations(
    generations_path: Path, strict: bool = False
) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    """Load generation rows keyed by identity, with schema migration.

    Returns the most recent row for each identity; invalid rows are skipped
    (or raise when ``strict``).
    """
    generations_by_key: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    if not Path(generations_path).exists():
        return generations_by_key
    with open(generations_path) as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = migrate_suite_generation_record(json.loads(line))
                generations_by_key[suite_generation_identity(row)] = row
            except (TypeError, ValueError, json.JSONDecodeError) as e:
                message = (
                    f"invalid suite generation row at {generations_path}:{line_num} ({e})"
                )
                if strict:
                    raise ValueError(message) from e
                logger.warning("Skipping %s", message)
                continue
    return generations_by_key


def reset_suite_generations(path: Path) -> int:
    """Truncate an existing generations file; returns rows discarded."""
    p = Path(path)
    if not p.exists():
        return 0
    with open(p) as f:
        discarded = sum(1 for line in f if line.strip())
    # Atomic truncate via replace to avoid a partially written file on crash.
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=p.parent, prefix=p.name + ".", suffix=".tmp", delete=False
        ) as f:
            f.flush()
            os.fsync(f.fileno())
            temp_path = Path(f.name)
        os.replace(temp_path, p)
    except OSError:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        p.write_text("")
    if discarded:
        logger.warning(
            "Resetting existing suite generations; discarding %d prior rows",
            discarded,
        )
    return discarded
