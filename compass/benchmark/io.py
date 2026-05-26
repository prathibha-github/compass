"""Benchmark record loading helpers."""

import json
import logging
from pathlib import Path

from compass.benchmark.schemas import (
    evaluation_identity,
    generation_identity,
    migrate_evaluation_record,
    migrate_generation_record,
)

logger = logging.getLogger(__name__)


def _invalid_record_error(path: Path, line_num: int, record_type: str, error: Exception) -> str:
    return f"invalid {record_type} row at {path}:{line_num} ({error})"


def load_generation_records(generations_path: Path, strict: bool = False) -> dict:
    """Load generation rows with schema migration and validation."""
    generations_by_key = {}
    with open(generations_path) as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = migrate_generation_record(json.loads(line))
                generations_by_key[generation_identity(row)] = row
            except (TypeError, ValueError, json.JSONDecodeError) as e:
                error_message = _invalid_record_error(
                    generations_path, line_num, "generation", e
                )
                if strict:
                    raise ValueError(error_message) from e
                logger.warning("Skipping %s", error_message)
                continue
    return generations_by_key


def load_evaluation_records(evaluations_path: Path, strict: bool = False) -> list:
    """Load evaluation rows with schema migration and validation."""
    rows = []
    with open(evaluations_path) as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = migrate_evaluation_record(json.loads(line))
                rows.append(row)
            except (TypeError, ValueError, json.JSONDecodeError) as e:
                error_message = _invalid_record_error(
                    evaluations_path, line_num, "evaluation", e
                )
                if strict:
                    raise ValueError(error_message) from e
                logger.warning("Skipping %s", error_message)
                continue
    return rows
