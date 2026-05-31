"""Compass: Evaluation framework for subjective model behavior."""
from compass._version import __version__
from compass.cache import EvaluationCache
from compass.clients import (
    AnthropicClient,
    GoogleAIClient,
    OllamaClient,
    OpenAIClient,
    OpenAIResponsesClient,
)
from compass.comparison import ComparisonResult, MultiModelComparator, PairwiseRanker
from compass.evaluation import (
    CheckpointManager,
    evaluate_suite_completions,
    generate_suite_completions,
    summarize_suite_evaluations,
)
from compass.judges import EvaluationResult, JudgeConfig, JudgeReliabilityAuditor, LLMJudge
from compass.reproducibility import (
    EvaluationMetadata,
    cost_per_judge,
    cost_summary,
    reproducibility_report,
)
from compass.rubrics import Rubric, RubricLibrary

__all__ = [
    "__version__",
    "Rubric",
    "RubricLibrary",
    "EvaluationResult",
    "JudgeConfig",
    "LLMJudge",
    "JudgeReliabilityAuditor",
    "EvaluationCache",
    "CheckpointManager",
    "generate_suite_completions",
    "evaluate_suite_completions",
    "summarize_suite_evaluations",
    "AnthropicClient",
    "GoogleAIClient",
    "OllamaClient",
    "OpenAIClient",
    "OpenAIResponsesClient",
    "ComparisonResult",
    "MultiModelComparator",
    "PairwiseRanker",
    "EvaluationMetadata",
    "reproducibility_report",
    "cost_summary",
    "cost_per_judge",
]
