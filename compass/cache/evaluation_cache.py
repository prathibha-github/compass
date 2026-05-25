"""Deterministic caching for evaluation results."""
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional

from compass.judges.base import EvaluationResult


class EvaluationCache:
    """Persistent cache for judge evaluation results.

    Cache keys are deterministic hashes of the judge configuration, input text,
    and prompt template version. The same evaluation contract always produces
    the same cache hit.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or ".compass_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory: Dict[
            str, EvaluationResult
        ] = {}  # In-memory cache for this session

    def _cache_key(self, config_hash: str, text_hash: str, prompt_version: str) -> str:
        """Deterministic cache key from inputs.

        Truncates SHA256 to 12 characters. Balances collision resistance (12 chars = ~2^48
        theoretical space) with reasonable filename length. For ~10k evaluations, collision
        risk is negligible.
        """
        key_str = f"{config_hash}_{text_hash}_{prompt_version}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:12]

    def get(
        self, config_hash: str, text_hash: str, prompt_version: str
    ) -> Optional[EvaluationResult]:
        """Get cached result if available.

        Checks in-memory cache first, then disk.
        """
        key = self._cache_key(config_hash, text_hash, prompt_version)

        # Check memory first
        if key in self.memory:
            return self.memory[key]

        # Check disk (EAFP: attempt read directly, handle if missing)
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file) as f:
                data = json.load(f)
                result = EvaluationResult(**data)
                self.memory[key] = result
                return result
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            # Missing or corrupted cache file, skip
            return None

    def put(
        self,
        config_hash: str,
        text_hash: str,
        prompt_version: str,
        result: EvaluationResult,
    ) -> None:
        """Store result in cache (memory and disk)."""
        key = self._cache_key(config_hash, text_hash, prompt_version)
        self.memory[key] = result

        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
        except OSError as e:
            import logging
            logging.getLogger(__name__).warning("Cache write failed for %s: %s", key, e)

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
