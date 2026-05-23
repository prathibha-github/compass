"""Core tic/style evaluation primitives: types, suite definition, and analytics."""
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

from compass.judges.reliability import wilson_interval as _wilson_base


# ── Wilson interval wrapper ──────────────────────────────────────────────────

def _wilson_pct(hits: int, n: int) -> Dict[str, float]:
    """Return a Wilson 95% CI as a {low, high} dict in percentage points."""
    if n == 0:
        return {"low": 0.0, "high": 0.0}
    lo, hi = _wilson_base(hits, n)
    return {"low": round(lo * 100, 1), "high": round(hi * 100, 1)}


# ── Core data types ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StyleCondition:
    """A system-prompt condition to evaluate."""
    name: str
    system_prompt: str
    description: str = ""


@dataclass(frozen=True)
class StylePrompt:
    """A user prompt in a tic/style suite."""
    id: str
    text: str
    task_type: str = "general"


@dataclass(frozen=True)
class DetectorResult:
    """
    Normalized output for one detector on one completion.

    Use name/count/score/hit for aggregation; metadata is detector-specific
    JSON-serializable diagnostic detail (matched spans, phrase hit counts, etc.).
    """
    name: str
    count: int
    score: float
    hit: bool
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "count": self.count,
            "score": self.score,
            "hit": self.hit,
            "metadata": self.metadata,
        }


class TicDetector:
    """Base detector interface."""
    name: str

    def detect(self, text: str) -> DetectorResult:
        raise NotImplementedError


JudgeFn = Callable[[str, int], str]

RecordExtras = Callable[[str, Dict[str, DetectorResult]], Dict[str, object]]

LOW_SAMPLE_THRESHOLD = 30


@dataclass(frozen=True)
class TicSuite:
    """
    A reusable tic/style evaluation suite.

    Example:
        suite = TicSuite(
            name="punctuation_tics",
            prompts=[StylePrompt(id="update", text="Write a short update.")],
            conditions=[StyleCondition(name="neutral", system_prompt="You are helpful.")],
            detectors=[CharacterCountDetector(name="em_dash", character="—")],
            baseline_condition="neutral",
        )

    record_extras can supply backward-compatible or suite-specific checkpoint
    fields derived from detector results. It receives the completion text and
    detector results keyed by detector name.
    """
    name: str
    prompts: List[StylePrompt]
    conditions: List[StyleCondition]
    detectors: List[TicDetector]
    baseline_condition: str
    checkpoint_name: Optional[str] = None
    max_tokens: int = 200
    record_extras: Optional[RecordExtras] = None

    @property
    def checkpoint_stem(self) -> str:
        return self.checkpoint_name or self.name

    def extra_record_fields(
        self, text: str, detector_results: Dict[str, DetectorResult]
    ) -> Dict[str, object]:
        if not self.record_extras:
            return {}
        return self.record_extras(text, detector_results)


# ── Analytics ────────────────────────────────────────────────────────────────

def suite_uses_llm_judge(suite: TicSuite) -> bool:
    """Return True if any detector in the suite requires a judge function."""
    return any(hasattr(detector, "detect_with_judge") for detector in suite.detectors)


def analyze_text(
    text: str,
    suite: TicSuite,
    judge_fn: Optional[JudgeFn] = None,
) -> Dict[str, DetectorResult]:
    """Run all suite detectors on a completion."""
    results: Dict[str, DetectorResult] = {}
    for detector in suite.detectors:
        if hasattr(detector, "detect_with_judge"):
            if judge_fn is None:
                raise ValueError(f"Detector {detector.name!r} requires a judge function")
            results[detector.name] = detector.detect_with_judge(text, judge_fn)
        else:
            results[detector.name] = detector.detect(text)
    return results


def summarize_outputs(
    condition_outputs: Dict[str, List[str]],
    suite: TicSuite,
) -> Dict[str, dict]:
    """Aggregate detector prevalence and intensity by condition."""
    summary: Dict[str, dict] = {}

    for condition, outputs in condition_outputs.items():
        detector_values = {detector.name: [] for detector in suite.detectors}
        detector_hits = {detector.name: 0 for detector in suite.detectors}

        for output in outputs:
            results = analyze_text(output, suite)
            for name, result in results.items():
                detector_values[name].append(result.count)
                if result.hit:
                    detector_hits[name] += 1

        n_outputs = len(outputs)
        detectors: Dict[str, dict] = {}
        for name, values in detector_values.items():
            ci = _wilson_pct(detector_hits[name], n_outputs)
            detectors[name] = {
                "hit_count": detector_hits[name],
                "mean_count": round(sum(values) / n_outputs, 2) if n_outputs else 0.0,
                "pct_hit": round(100 * detector_hits[name] / n_outputs, 1) if n_outputs else 0.0,
                "pct_hit_ci_low": ci["low"],
                "pct_hit_ci_high": ci["high"],
                "low_sample": 0 < n_outputs < LOW_SAMPLE_THRESHOLD,
            }

        summary[condition] = {
            "n_outputs": n_outputs,
            "low_sample": 0 < n_outputs < LOW_SAMPLE_THRESHOLD,
            "detectors": detectors,
        }

    return summary


def summarize_detector_records(
    condition_records: Dict[str, List[Dict[str, dict]]],
    suite: TicSuite,
) -> Dict[str, dict]:
    """Aggregate detector results already stored in checkpoint records."""
    summary: Dict[str, dict] = {}

    for condition, records in condition_records.items():
        detector_values = {detector.name: [] for detector in suite.detectors}
        detector_hits = {detector.name: 0 for detector in suite.detectors}

        for detector_record in records:
            for detector in suite.detectors:
                payload = detector_record.get(detector.name, {})
                count = float(payload.get("count", 0.0))
                hit = bool(payload.get("hit", count > 0))
                detector_values[detector.name].append(count)
                if hit:
                    detector_hits[detector.name] += 1

        n_outputs = len(records)
        detector_summary: Dict[str, dict] = {}
        for detector in suite.detectors:
            values = detector_values[detector.name]
            hit_count = detector_hits[detector.name]
            ci = _wilson_pct(hit_count, n_outputs)
            detector_summary[detector.name] = {
                "hit_count": hit_count,
                "pct_hit": round(100 * hit_count / n_outputs, 1) if n_outputs else 0.0,
                "pct_hit_ci_low": ci["low"],
                "pct_hit_ci_high": ci["high"],
                "mean_count": round(sum(values) / n_outputs, 3) if n_outputs else 0.0,
                "low_sample": 0 < n_outputs < LOW_SAMPLE_THRESHOLD,
            }

        summary[condition] = {
            "n_outputs": n_outputs,
            "low_sample": 0 < n_outputs < LOW_SAMPLE_THRESHOLD,
            "detectors": detector_summary,
        }

    return summary


def compute_uplift(
    condition_results: Dict[str, dict],
    detector_name: str,
    metric: str = "mean_count",
    baseline: Optional[str] = None,
) -> Dict[str, float]:
    """Compute detector uplift by condition against a baseline condition."""
    if not condition_results:
        return {}

    baseline = baseline or next(iter(condition_results))
    if baseline not in condition_results:
        baseline = next(iter(condition_results))

    base_stats = condition_results[baseline]["detectors"].get(detector_name, {})
    base_value = base_stats.get(metric, 0.0)

    uplift: Dict[str, float] = {}
    for condition, stats in condition_results.items():
        value = stats["detectors"].get(detector_name, {}).get(metric, 0.0)
        if base_value > 0:
            delta = 100 * (value - base_value) / max(base_value, 0.1)
        else:
            delta = 100 if value > 0 else 0
        uplift[condition] = round(delta, 1)
    return uplift
