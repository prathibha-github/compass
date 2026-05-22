"""Deterministic caching for evaluation results."""
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional

from compass.judges.base import EvaluationResult


class EvaluationCache:
    """Persistent cache for judge evaluation results.

    Cache keys are deterministic hashes of (rubric_hash, text_hash, judge_model),
    so the same evaluation always produces the same cache hit.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or ".compass_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory: Dict[str, EvaluationResult] = {}  # In-memory cache for this session

    def _cache_key(self, rubric_hash: str, text_hash: str, judge_model: str) -> str:
        """Deterministic cache key from inputs."""
        key_str = f"{rubric_hash}_{text_hash}_{judge_model}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:12]

    def get(self, rubric_hash: str, text_hash: str, judge_model: str) -> Optional[EvaluationResult]:
        """Get cached result if available.

        Checks in-memory cache first, then disk.
        """
        key = self._cache_key(rubric_hash, text_hash, judge_model)

        # Check memory first
        if key in self.memory:
            return self.memory[key]

        # Check disk
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                    result = EvaluationResult(**data)
                    self.memory[key] = result
                    return result
            except (json.JSONDecodeError, TypeError):
                # Corrupted cache file, skip
                return None

        return None

    def put(
        self, rubric_hash: str, text_hash: str, judge_model: str, result: EvaluationResult
    ) -> None:
        """Store result in cache (memory and disk)."""
        key = self._cache_key(rubric_hash, text_hash, judge_model)
        self.memory[key] = result

        cache_file = self.cache_dir / f"{key}.json"
        with open(cache_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    def stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        cache_files = list(self.cache_dir.glob("*.json"))
        return {
            "cached_files": len(cache_files),
            "memory_items": len(self.memory),
            "cache_dir": str(self.cache_dir),
        }

    def clear(self) -> None:
        """Clear all cached results."""
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
        self.memory.clear()
