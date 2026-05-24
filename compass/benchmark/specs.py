"""Benchmark specification contracts."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Mapping, Sequence, Tuple

from compass.rubrics.base import Rubric


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

    @property
    def rubric_names(self) -> Tuple[str, ...]:
        return tuple(self.prompts_by_rubric.keys())

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


def build_benchmark_spec(
    name: str,
    version: str,
    prompts_by_rubric: Mapping[str, Sequence[Any]],
    rubrics_by_name: Mapping[str, Rubric],
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
        pairwise_segment_field=pairwise_segment_field,
    )
