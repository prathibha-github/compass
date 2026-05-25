"""Schema normalization for benchmark generation and evaluation records."""

from typing import Any, Dict, Tuple

from compass.schema_fields import (
    BENCHMARK_RECORD_TYPE_FIELD,
    BENCHMARK_SCHEMA_VERSION_FIELD,
)

LEGACY_BENCHMARK_SCHEMA_VERSION = 0
BENCHMARK_SCHEMA_VERSION = 1
BENCHMARK_NAME_FIELD = "benchmark_name"
BENCHMARK_VERSION_FIELD = "benchmark_version"

GENERATION_RECORD_TYPE = "generation"
EVALUATION_RECORD_TYPE = "evaluation"
VALID_BENCHMARK_RECORD_TYPES = frozenset(
    {GENERATION_RECORD_TYPE, EVALUATION_RECORD_TYPE}
)

_SUPPORTED_INPUT_SCHEMA_VERSIONS = frozenset(
    {LEGACY_BENCHMARK_SCHEMA_VERSION, BENCHMARK_SCHEMA_VERSION}
)


def _coerce_sample_idx(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    return int(value)


def _validate_base_shape(record: Dict[str, Any]) -> None:
    required = ("model", "rubric", "prompt_id")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"missing required fields: {missing}")


def _validate_benchmark_identity(record: Dict[str, Any]) -> None:
    name = record.get(BENCHMARK_NAME_FIELD)
    version = record.get(BENCHMARK_VERSION_FIELD)
    if name is None and version is None:
        return
    if not name or not version:
        raise ValueError(
            "benchmark record must include both benchmark_name and "
            "benchmark_version when either is present"
        )


def migrate_generation_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a generation result record to benchmark schema v1."""
    if not isinstance(record, dict):
        raise TypeError("generation record must be a dict")

    data = dict(record)
    _validate_base_shape(data)
    _validate_benchmark_identity(data)
    data["sample_idx"] = _coerce_sample_idx(data.get("sample_idx", 0))

    schema_version = int(
        data.get(BENCHMARK_SCHEMA_VERSION_FIELD, LEGACY_BENCHMARK_SCHEMA_VERSION)
    )
    if schema_version not in _SUPPORTED_INPUT_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported benchmark schema version: {schema_version}")

    record_type = data.get(BENCHMARK_RECORD_TYPE_FIELD, GENERATION_RECORD_TYPE)
    if record_type != GENERATION_RECORD_TYPE:
        raise ValueError(
            f"expected generation record type, got {record_type!r}"
        )

    if "completion" not in data:
        raise ValueError("generation record missing completion")

    data[BENCHMARK_SCHEMA_VERSION_FIELD] = BENCHMARK_SCHEMA_VERSION
    data[BENCHMARK_RECORD_TYPE_FIELD] = GENERATION_RECORD_TYPE
    return data


def migrate_evaluation_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an evaluation result record to benchmark schema v1."""
    if not isinstance(record, dict):
        raise TypeError("evaluation record must be a dict")

    data = dict(record)
    _validate_base_shape(data)
    _validate_benchmark_identity(data)
    data["sample_idx"] = _coerce_sample_idx(data.get("sample_idx", 0))

    schema_version = int(
        data.get(BENCHMARK_SCHEMA_VERSION_FIELD, LEGACY_BENCHMARK_SCHEMA_VERSION)
    )
    if schema_version not in _SUPPORTED_INPUT_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported benchmark schema version: {schema_version}")

    record_type = data.get(BENCHMARK_RECORD_TYPE_FIELD)
    if record_type is None:
        # Legacy evaluation rows had no explicit type but always include score/hit.
        if "score" in data and "hit" in data:
            record_type = EVALUATION_RECORD_TYPE
        else:
            raise ValueError("unknown evaluation record format")

    if record_type != EVALUATION_RECORD_TYPE:
        raise ValueError(
            f"expected evaluation record type, got {record_type!r}"
        )

    if "score" not in data or "hit" not in data:
        raise ValueError("evaluation record missing score/hit")

    data[BENCHMARK_SCHEMA_VERSION_FIELD] = BENCHMARK_SCHEMA_VERSION
    data[BENCHMARK_RECORD_TYPE_FIELD] = EVALUATION_RECORD_TYPE
    return data


def generation_identity(record: Dict[str, Any]) -> Tuple[Any, ...]:
    """Canonical identity tuple for generation records."""
    row = migrate_generation_record(record)
    return (row["model"], row["rubric"], row["prompt_id"], row["sample_idx"])


def evaluation_identity(record: Dict[str, Any]) -> Tuple[Any, ...]:
    """Canonical identity tuple for evaluation records."""
    row = migrate_evaluation_record(record)
    return (row["model"], row["rubric"], row["prompt_id"], row["sample_idx"])
