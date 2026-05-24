"""Checkpoint record schema normalization and identity extraction."""

from typing import Any, Dict, Tuple

CHECKPOINT_SCHEMA_VERSION = 1
LEGACY_SCHEMA_VERSION = 0

# Persisted on-disk values: treat as frozen schema constants.
RECORD_TYPE_SUITE = "suite_eval"
RECORD_TYPE_BENCHMARK = "benchmark_eval"
VALID_RECORD_TYPES = frozenset({RECORD_TYPE_SUITE, RECORD_TYPE_BENCHMARK})
_SUPPORTED_INPUT_SCHEMA_VERSIONS = {LEGACY_SCHEMA_VERSION, CHECKPOINT_SCHEMA_VERSION}


def _coerce_sample_idx(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    return int(value)


def _is_suite_shape(record: Dict[str, Any]) -> bool:
    return all(
        key in record
        for key in ("model", "suite", "detector", "prompt_id", "condition")
    )


def _is_benchmark_shape(record: Dict[str, Any]) -> bool:
    return all(key in record for key in ("model", "rubric", "prompt_id"))


def migrate_checkpoint_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a checkpoint record to the current schema shape.

    Legacy records are supported:
    - missing ``sample_idx`` defaults to 0
    - missing ``schema_version`` defaults to 0 (legacy)
    - missing ``record_type`` is inferred from known field shapes
    """
    if not isinstance(record, dict):
        raise TypeError("checkpoint record must be a dict")

    data = dict(record)
    data["sample_idx"] = _coerce_sample_idx(data.get("sample_idx", 0))

    schema_version_raw = data.get("schema_version", LEGACY_SCHEMA_VERSION)
    schema_version = int(schema_version_raw)
    if schema_version not in _SUPPORTED_INPUT_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported checkpoint schema_version={schema_version}")

    record_type = data.get("record_type")
    if record_type is None:
        # Prefer suite shape when fields overlap (some suite records may also
        # carry a rubric field for reporting).
        if _is_suite_shape(data):
            record_type = RECORD_TYPE_SUITE
        elif _is_benchmark_shape(data):
            record_type = RECORD_TYPE_BENCHMARK
        else:
            raise ValueError("unknown checkpoint record format")

    if record_type not in VALID_RECORD_TYPES:
        raise ValueError(f"unsupported checkpoint record_type={record_type!r}")

    if record_type == RECORD_TYPE_BENCHMARK and not _is_benchmark_shape(data):
        raise ValueError("benchmark_eval record missing required fields")
    if record_type == RECORD_TYPE_SUITE and not _is_suite_shape(data):
        raise ValueError("suite_eval record missing required fields")

    data["schema_version"] = CHECKPOINT_SCHEMA_VERSION
    data["record_type"] = record_type
    return data


def checkpoint_identity(record: Dict[str, Any]) -> Tuple[Any, ...]:
    """Return canonical identity tuple for a normalized checkpoint record."""
    normalized = migrate_checkpoint_record(record)
    if normalized["record_type"] == RECORD_TYPE_BENCHMARK:
        return (
            normalized["model"],
            normalized["rubric"],
            normalized["prompt_id"],
            normalized["sample_idx"],
        )

    return suite_sample_identity(
        model=normalized["model"],
        suite=normalized["suite"],
        detector=normalized["detector"],
        prompt_id=normalized["prompt_id"],
        condition=normalized["condition"],
        sample_idx=normalized["sample_idx"],
    )


def suite_sample_identity(
    model: str,
    suite: str,
    detector: str,
    prompt_id: str,
    condition: str,
    sample_idx: int,
) -> Tuple[Any, ...]:
    """Canonical identity tuple for suite-eval checkpoint rows."""
    return (
        model,
        suite,
        detector,
        prompt_id,
        condition,
        sample_idx,
    )
