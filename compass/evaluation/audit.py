"""Keep failed attempts separate from measurements used in summaries."""

import json
from datetime import datetime, timezone
from pathlib import Path


def record_failure(path, identity, error):
    row = {"timestamp": datetime.now(timezone.utc).isoformat(), "identity": list(identity),
           "error_type": type(error).__name__, "error": str(error),
           "audit": getattr(error, "audit_record", {"usage_complete": False})}
    with Path(path).with_suffix(".failures.jsonl").open("a") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")
