"""JSON parsing with defensive fallbacks for judge responses."""
import json
import math
from numbers import Real
from typing import Optional


SCORING_VERSION = "2"


class JudgeResponseError(ValueError):
    """A judge response cannot be interpreted as a valid measurement."""


def validate_score(value, *, name="score"):
    """Validate a stored or supplied score without coercion or clamping."""
    if isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value <= 1 or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number between zero and one")
    return value


def parse_scored_response(raw: str) -> dict:
    """Require a finite numeric score; never turn a failed judgment into zero.

    Numeric strings and finite out-of-range scores retain the historical
    conversion/clamping behavior. A model-provided ``hit`` is ignored: callers
    derive it from their configured threshold.
    """
    payload = parse_judge_response(raw)
    if not isinstance(payload, dict):
        raise JudgeResponseError("Judge response must contain a JSON object with a score")
    value = payload.get("score")
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise JudgeResponseError("Judge score must be a finite number")
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise JudgeResponseError("Judge score must be a finite number") from exc
    if not math.isfinite(score):
        raise JudgeResponseError("Judge score must be a finite number")
    confidence = payload.get("confidence")
    try:
        confidence = float(confidence) if not isinstance(confidence, bool) else None
    except (TypeError, ValueError, OverflowError):
        confidence = None
    if confidence is not None and not (math.isfinite(confidence) and 0 <= confidence <= 1):
        confidence = None
    return {
        "score": max(0.0, min(1.0, score)),
        "confidence": confidence,
        "rationale": str(payload.get("rationale", ""))[:500],
    }


def parse_judge_response(raw: str) -> Optional[dict]:
    """Parse JSON from judge response with fallbacks.

    Tries:
    1. Direct JSON parsing
    2. Extracting {...} from prose and parsing
    3. Returns None if both fail
    """
    # Try direct parse
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
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
