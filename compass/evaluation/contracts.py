"""Stable fingerprints for persisted evaluation inputs.

Fingerprints protect against accidental reuse, not artifact tampering. Provider
model aliases and external state used by custom clients remain caller concerns.
"""

import hashlib
import inspect
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path


def fingerprint(value) -> str:
    """Hash JSON-compatible inputs without lossy string fallbacks."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_suite_identities(suite) -> None:
    """Reject ambiguous identifiers before they collapse persisted samples."""
    for label, values in (
        ("prompt IDs", [prompt.id for prompt in suite.prompts]),
        ("condition names", [condition.name for condition in suite.conditions]),
        ("detector names", [detector.name for detector in suite.detectors]),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"Suite {label} must be unique")


def detector_definition(detector) -> dict:
    """Capture detector parameters and the module implementing its behavior.

    Custom detectors with non-JSON state or unavailable source must provide a
    JSON-compatible ``resume_contract()`` containing their configuration and an
    implementation version, including versions of external dependencies/state.
    """
    cls = type(detector)
    if hasattr(detector, "resume_contract"):
        definition = detector.resume_contract()
    else:
        source = inspect.getsourcefile(cls)
        if not source or not Path(source).is_file():
            raise ValueError(f"Detector {detector.name!r} must define resume_contract()")
        definition = {
            "parameters": asdict(detector) if is_dataclass(detector) else vars(detector),
            "implementation": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        }
    result = {"type": f"{cls.__module__}.{cls.__qualname__}", "definition": definition}
    fingerprint(result)  # Fail before writing artifacts if state is unsupported.
    return result


def manifest_contracts(manifest):
    """Read an expected population without silently collapsing duplicate IDs."""
    if not isinstance(manifest, dict) or not isinstance(manifest.get("expected"), list):
        raise ValueError("Invalid manifest: expected an expected-row list")
    expected = {}
    for item in manifest["expected"]:
        if not isinstance(item, dict):
            raise ValueError("Invalid manifest row")
        identity, contract = item.get("identity"), item.get("contract")
        if (not isinstance(identity, list) or not identity
                or any(not isinstance(part, str) for part in identity[:-1])
                or type(identity[-1]) is not int or identity[-1] < 0
                or not isinstance(contract, str) or not contract):
            raise ValueError("Invalid manifest identity or contract")
        key = tuple(identity)
        if key in expected:
            raise ValueError("Duplicate manifest identity")
        expected[key] = contract
    return expected
