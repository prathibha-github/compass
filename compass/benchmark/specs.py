"""Benchmark specification contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

from compass.benchmark.config import DEFAULT_TOKEN_BUDGETS, LEGACY_TOKEN_CAP_FALLBACK
from compass.rubrics.base import Rubric

if TYPE_CHECKING:
    from compass.benchmark.reporting import BenchmarkPairwiseReport, BenchmarkSummaryRow
    from compass.benchmark.validation import BenchmarkValidationIssue

_DEFAULT_ANALYSIS_LANES = ("summary", "pairwise")
_ALLOWED_ANALYSIS_LANES = frozenset(_DEFAULT_ANALYSIS_LANES)
_ALLOWED_QUALITY_FILTER_MODES = frozenset(("annotate", "exclude_flagged"))


def _normalize_models(models: Sequence[str]) -> Tuple[str, ...]:
    normalized = tuple(str(model) for model in models)
    if not normalized:
        raise ValueError("benchmark run config must define at least one model")
    if any(not model for model in normalized):
        raise ValueError("benchmark run config models cannot be empty")
    return normalized


def _normalize_analysis_lanes(analysis_lanes: Sequence[str]) -> Tuple[str, ...]:
    normalized = tuple(str(lane) for lane in analysis_lanes)
    if not normalized:
        raise ValueError("benchmark run config must define at least one analysis lane")
    invalid = sorted(set(normalized) - _ALLOWED_ANALYSIS_LANES)
    if invalid:
        raise ValueError(f"unsupported analysis lanes: {invalid}")
    return normalized


def _normalize_token_budgets(token_budgets: Mapping[str, int]) -> Mapping[str, int]:
    normalized = {str(prefix): int(budget) for prefix, budget in token_budgets.items()}
    if "default" not in normalized:
        raise ValueError("token budget config must define a 'default' entry")
    invalid = {
        prefix: budget
        for prefix, budget in normalized.items()
        if prefix == "" or budget <= 0
    }
    if invalid:
        raise ValueError(f"invalid token budget config: {invalid}")
    return MappingProxyType(normalized)


def _normalize_budget_overrides(
    token_budgets: Mapping[str, int],
) -> Mapping[str, int]:
    normalized = {}
    for model, budget in token_budgets.items():
        model_name = str(model)
        budget_value = int(budget)
        if not model_name:
            raise ValueError("model name cannot be empty in max_tokens_by_model")
        if budget_value <= 0:
            raise ValueError(
                f"Invalid max token budget for model {model_name!r}: {budget_value}"
            )
        normalized[model_name] = budget_value
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class BenchmarkPolicyDefaults:
    """Benchmark-owned policy defaults for a preset."""

    token_budgets: Mapping[str, int] = field(
        default_factory=lambda: dict(DEFAULT_TOKEN_BUDGETS)
    )
    allow_mixed_token_budgets: bool = False
    quality_filter_mode: str = "annotate"
    analysis_lanes: Tuple[str, ...] = _DEFAULT_ANALYSIS_LANES

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "token_budgets",
            _normalize_token_budgets(self.token_budgets),
        )
        if self.quality_filter_mode not in _ALLOWED_QUALITY_FILTER_MODES:
            raise ValueError(
                "quality_filter_mode must be one of "
                f"{sorted(_ALLOWED_QUALITY_FILTER_MODES)}"
            )
        object.__setattr__(
            self,
            "analysis_lanes",
            _normalize_analysis_lanes(self.analysis_lanes),
        )


@dataclass(frozen=True)
class BenchmarkRunPreset:
    """Named default run settings owned by a benchmark."""

    models: Tuple[str, ...]
    samples: int
    judge_model: str
    output_dir: str
    legacy_token_cap_threshold: int = LEGACY_TOKEN_CAP_FALLBACK
    policy: BenchmarkPolicyDefaults = field(default_factory=BenchmarkPolicyDefaults)

    def __post_init__(self) -> None:
        object.__setattr__(self, "models", _normalize_models(self.models))
        if self.samples <= 0:
            raise ValueError("benchmark preset samples must be > 0")
        if not self.judge_model:
            raise ValueError("benchmark preset judge_model is required")
        if not self.output_dir:
            raise ValueError("benchmark preset output_dir is required")
        if self.legacy_token_cap_threshold <= 0:
            raise ValueError("benchmark preset legacy_token_cap_threshold must be > 0")
        if not isinstance(self.policy, BenchmarkPolicyDefaults):
            raise TypeError("benchmark preset policy must be a BenchmarkPolicyDefaults")


@dataclass(frozen=True)
class BenchmarkRunConfig:
    """Resolved benchmark run config after preset selection and overrides."""

    benchmark_name: str
    benchmark_version: str
    preset_name: str
    models: Tuple[str, ...]
    samples: int
    judge_model: str
    output_dir: str
    token_budget_defaults: Mapping[str, int]
    allow_mixed_token_budgets: bool
    quality_filter_mode: str
    analysis_lanes: Tuple[str, ...]
    legacy_token_cap_threshold: int
    max_tokens_by_model: Optional[Mapping[str, int]] = None
    skip_generation: bool = False
    skip_ranking: bool = False

    def __post_init__(self) -> None:
        if not self.benchmark_name:
            raise ValueError("benchmark run config benchmark_name is required")
        if not self.benchmark_version:
            raise ValueError("benchmark run config benchmark_version is required")
        if not self.preset_name:
            raise ValueError("benchmark run config preset_name is required")
        object.__setattr__(self, "models", _normalize_models(self.models))
        if self.samples <= 0:
            raise ValueError("benchmark run config samples must be > 0")
        if not self.judge_model:
            raise ValueError("benchmark run config judge_model is required")
        if not self.output_dir:
            raise ValueError("benchmark run config output_dir is required")
        if self.legacy_token_cap_threshold <= 0:
            raise ValueError("benchmark run config legacy_token_cap_threshold must be > 0")
        object.__setattr__(
            self,
            "token_budget_defaults",
            _normalize_token_budgets(self.token_budget_defaults),
        )
        if self.max_tokens_by_model is not None:
            object.__setattr__(
                self,
                "max_tokens_by_model",
                _normalize_budget_overrides(self.max_tokens_by_model),
            )
        if self.quality_filter_mode not in _ALLOWED_QUALITY_FILTER_MODES:
            raise ValueError(
                "quality_filter_mode must be one of "
                f"{sorted(_ALLOWED_QUALITY_FILTER_MODES)}"
            )
        object.__setattr__(
            self,
            "analysis_lanes",
            _normalize_analysis_lanes(self.analysis_lanes),
        )

    @property
    def effective_analysis_lanes(self) -> Tuple[str, ...]:
        if not self.skip_ranking:
            return self.analysis_lanes
        return tuple(lane for lane in self.analysis_lanes if lane != "pairwise")


def _default_run_preset(name: str) -> BenchmarkRunPreset:
    return BenchmarkRunPreset(
        models=("llama3.1",),
        samples=1,
        judge_model="llama3.1",
        output_dir=f"results/{name}",
    )


@dataclass(frozen=True)
class BenchmarkPrompt:
    """Single benchmark prompt."""

    id: str
    text: str
    task_type: str

    def as_dict(self) -> Dict[str, str]:
        return {
            "id": self.id,
            "text": self.text,
            "task_type": self.task_type,
        }


def _coerce_prompt(prompt: Any) -> BenchmarkPrompt:
    if isinstance(prompt, BenchmarkPrompt):
        return prompt
    if not isinstance(prompt, Mapping):
        raise TypeError("benchmark prompt must be a BenchmarkPrompt or mapping")

    missing = [key for key in ("id", "text", "task_type") if key not in prompt]
    if missing:
        raise ValueError(f"benchmark prompt missing required fields: {missing}")

    return BenchmarkPrompt(
        id=str(prompt["id"]),
        text=str(prompt["text"]),
        task_type=str(prompt["task_type"]),
    )


@dataclass(frozen=True)
class BenchmarkSpec:
    """Immutable benchmark definition."""

    name: str
    version: str
    prompts_by_rubric: Mapping[str, Tuple[BenchmarkPrompt, ...]]
    rubrics_by_name: Mapping[str, Rubric]
    run_presets: Mapping[str, BenchmarkRunPreset] = field(default_factory=dict)
    default_preset: str = "default"
    pairwise_segment_field: str = "task_type"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("benchmark spec name is required")
        if not self.version:
            raise ValueError("benchmark spec version is required")
        if not self.prompts_by_rubric:
            raise ValueError("benchmark spec must define at least one rubric")
        if not self.rubrics_by_name:
            raise ValueError("benchmark spec must define rubric objects")

        normalized_prompts = {
            rubric: tuple(_coerce_prompt(prompt) for prompt in prompts)
            for rubric, prompts in self.prompts_by_rubric.items()
        }
        if set(normalized_prompts) != set(self.rubrics_by_name):
            raise ValueError(
                "prompt rubrics and rubric objects must cover the same rubric names"
            )
        normalized_presets = (
            dict(self.run_presets)
            if self.run_presets
            else {self.default_preset: _default_run_preset(self.name)}
        )
        if not self.default_preset:
            raise ValueError("default_preset is required")
        if self.default_preset not in normalized_presets:
            raise ValueError(
                "default_preset must point to an entry in benchmark run_presets"
            )
        for preset_name, preset in normalized_presets.items():
            if not preset_name:
                raise ValueError("benchmark preset names cannot be empty")
            if not isinstance(preset, BenchmarkRunPreset):
                raise TypeError(
                    "benchmark run_presets must map to BenchmarkRunPreset instances"
                )
        if not self.pairwise_segment_field:
            raise ValueError("pairwise_segment_field is required")

        object.__setattr__(
            self,
            "prompts_by_rubric",
            MappingProxyType(normalized_prompts),
        )
        object.__setattr__(
            self,
            "rubrics_by_name",
            MappingProxyType(dict(self.rubrics_by_name)),
        )
        object.__setattr__(
            self,
            "run_presets",
            MappingProxyType(normalized_presets),
        )

    @property
    def rubric_names(self) -> Tuple[str, ...]:
        return tuple(self.prompts_by_rubric.keys())

    @property
    def preset_names(self) -> Tuple[str, ...]:
        return tuple(self.run_presets.keys())

    @property
    def prompt_count(self) -> int:
        return sum(len(prompts) for prompts in self.prompts_by_rubric.values())

    def total_evaluations(self, model_count: int, samples: int) -> int:
        return model_count * samples * self.prompt_count

    def as_prompt_dict(self) -> Dict[str, list]:
        return {
            rubric: [prompt.as_dict() for prompt in prompts]
            for rubric, prompts in self.prompts_by_rubric.items()
        }

    def get_run_preset(self, preset_name: Optional[str] = None) -> BenchmarkRunPreset:
        resolved_name = preset_name or self.default_preset
        try:
            return self.run_presets[resolved_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown benchmark preset: {resolved_name}. "
                f"Available: {sorted(self.run_presets)}"
            ) from exc

    def make_run_config(
        self,
        preset_name: Optional[str] = None,
        models: Optional[Sequence[str]] = None,
        samples: Optional[int] = None,
        judge_model: Optional[str] = None,
        output_dir: Optional[str] = None,
        max_tokens_by_model: Optional[Mapping[str, int]] = None,
        allow_mixed_token_budgets: Optional[bool] = None,
        legacy_token_cap_threshold: Optional[int] = None,
        quality_filter_mode: Optional[str] = None,
        analysis_lanes: Optional[Sequence[str]] = None,
        skip_generation: bool = False,
        skip_ranking: bool = False,
    ) -> BenchmarkRunConfig:
        preset_key = preset_name or self.default_preset
        preset = self.get_run_preset(preset_key)
        policy = preset.policy
        return BenchmarkRunConfig(
            benchmark_name=self.name,
            benchmark_version=self.version,
            preset_name=preset_key,
            models=tuple(models) if models is not None else preset.models,
            samples=samples if samples is not None else preset.samples,
            judge_model=judge_model or preset.judge_model,
            output_dir=output_dir or preset.output_dir,
            token_budget_defaults=policy.token_budgets,
            allow_mixed_token_budgets=(
                policy.allow_mixed_token_budgets
                if allow_mixed_token_budgets is None
                else allow_mixed_token_budgets
            ),
            quality_filter_mode=quality_filter_mode or policy.quality_filter_mode,
            analysis_lanes=analysis_lanes or policy.analysis_lanes,
            legacy_token_cap_threshold=(
                preset.legacy_token_cap_threshold
                if legacy_token_cap_threshold is None
                else legacy_token_cap_threshold
            ),
            max_tokens_by_model=max_tokens_by_model,
            skip_generation=skip_generation,
            skip_ranking=skip_ranking,
        )


@runtime_checkable
class BenchmarkRunner(Protocol):
    """Execution contract enforced at benchmark registration time."""

    spec: BenchmarkSpec

    def validate_run_config(self, run_config: BenchmarkRunConfig) -> BenchmarkRunConfig:
        """Validate a resolved benchmark run config."""

    def generate(self, run_config: BenchmarkRunConfig) -> Path:
        """Generate benchmark completions."""

    def evaluate(
        self,
        generations_path: Path,
        run_config: BenchmarkRunConfig,
    ) -> Path:
        """Evaluate benchmark completions."""

    def analyze(
        self,
        evaluations_path: Path,
        run_config: BenchmarkRunConfig,
    ) -> Dict[str, BenchmarkSummaryRow]:
        """Analyze benchmark evaluation outputs."""

    def rank(
        self,
        evaluations_path: Path,
        run_config: BenchmarkRunConfig,
    ) -> BenchmarkPairwiseReport:
        """Run pairwise ranking for a benchmark family."""

    def validate_report(
        self,
        evaluations_path: Path,
        run_config: BenchmarkRunConfig,
    ) -> Sequence[BenchmarkValidationIssue]:
        """Validate benchmark report artifacts."""


def build_benchmark_spec(
    name: str,
    version: str,
    prompts_by_rubric: Mapping[str, Sequence[Any]],
    rubrics_by_name: Mapping[str, Rubric],
    run_presets: Optional[Mapping[str, BenchmarkRunPreset]] = None,
    default_preset: str = "default",
    pairwise_segment_field: str = "task_type",
) -> BenchmarkSpec:
    """Build a benchmark spec from prompt dicts or BenchmarkPrompt objects."""
    return BenchmarkSpec(
        name=name,
        version=version,
        prompts_by_rubric={
            rubric: tuple(_coerce_prompt(prompt) for prompt in prompts)
            for rubric, prompts in prompts_by_rubric.items()
        },
        rubrics_by_name=dict(rubrics_by_name),
        run_presets=dict(run_presets or {}),
        default_preset=default_preset,
        pairwise_segment_field=pairwise_segment_field,
    )
