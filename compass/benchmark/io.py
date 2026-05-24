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


def load_generation_records(generations_path: Path) -> dict:
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
                logger.warning(
                    "Skipping invalid generation row at %s:%d (%s)",
                    generations_path,
                    line_num,
                    e,
                )
                continue
    return generations_by_key


def load_evaluation_records(evaluations_path: Path) -> list:
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
                logger.warning(
                    "Skipping invalid evaluation row at %s:%d (%s)",
                    evaluations_path,
                    line_num,
                    e,
                )
                continue
    return rows
