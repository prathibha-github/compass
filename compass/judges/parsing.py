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
    """Find and parse the first complete {...} within a string.

    Uses brace counting to find the matching closing brace, avoiding
    the greedy issue of .rfind("}") which would grab the last brace
    even if it belongs to a different object.
    """
    start = raw.find("{")
    if start < 0:
        return None

    brace_count = 0
    in_string = False
    escape_next = False

    for i in range(start, len(raw)):
        ch = raw[i]

        if escape_next:
            escape_next = False
            continue

        if ch == "\\" and in_string:
            escape_next = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if not in_string:
            if ch == "{":
                brace_count += 1
            elif ch == "}":
                brace_count -= 1
                if brace_count == 0:
                    try:
                        return json.loads(raw[start : i + 1])
                    except json.JSONDecodeError:
                        return None

    return None
