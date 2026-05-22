"""JSON parsing with defensive fallbacks for judge responses."""
import json
from typing import Optional


def parse_judge_response(raw: str) -> Optional[dict]:
    """Parse JSON from judge response with fallbacks.

    Tries:
    1. Direct JSON parsing
    2. Extracting {...} from prose and parsing
    3. Returns None if both fail
    """
    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON object from prose
    extracted = _extract_json_object(raw)
    if extracted:
        return extracted

    return None


def _extract_json_object(raw: str) -> Optional[dict]:
    """Find and parse {...} within a string."""
    start = raw.find("{")
    end = raw.rfind("}")

    if start < 0 or end <= start:
        return None

    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
