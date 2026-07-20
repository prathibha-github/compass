"""Detectors module: TicSuite framework and built-in evaluation suites."""
from compass.detectors.base import (
    DetectorResult,
    JudgeFn,
    LOW_SAMPLE_THRESHOLD,
    RecordExtras,
    StyleCondition,
    StylePrompt,
    TicDetector,
    TicSuite,
    analyze_text,
    compute_uplift,
    suite_uses_llm_judge,
    summarize_detector_records,
    summarize_outputs,
)
from compass.detectors.heuristic import (
    CharacterCountDetector,
    EMOJI_PATTERN,
    EmojiDetector,
    PhraseSetDetector,
    RegexDetector,
)
from compass.detectors.llm_detector import LLMJudgeDetector, LLM_JUDGE_SYSTEM_PROMPT
from compass.detectors.persona_transfer import analyze_persona_transfer
from compass.detectors.quirk_detection import find_quirks, QUIRK_PROMPTS
from compass.detectors.suites import (
    AGENTIC_SIDE_EFFECTS_SUITE,
    DEFAULT_SUITE_KEYS,
    FORMATTING_TIC_SUITE,
    GENERAL_WRITING_PROMPTS,
    LEXICAL_TIC_SUITE,
    LINKEDIN_SUITE,
    LLM_JUDGE_TONE_SUITE,
    MILD_INSULT_REACTIVITY_SUITE,
    PUNCTUATION_TIC_SUITE,
    REST_CAI_VALIDATION_SUITE,
    REST_HYPOTHESIS_SUITE,
    SELF_HARM_SAFETY_SUITE,
    STANDARD_CONDITIONS,
    SUITE_NOTES,
    SUITE_REGISTRY,
    TASK_FOCUS_SUITE,
    TONE_TIC_SUITE,
    TRUTHFULNESS_SUITE,
    expand_suite_names,
)

__all__ = [
    # base types
    "DetectorResult",
    "JudgeFn",
    "LOW_SAMPLE_THRESHOLD",
    "RecordExtras",
    "StyleCondition",
    "StylePrompt",
    "TicDetector",
    "TicSuite",
    # analytics
    "analyze_text",
    "compute_uplift",
    "suite_uses_llm_judge",
    "summarize_detector_records",
    "summarize_outputs",
    # heuristic detectors
    "CharacterCountDetector",
    "EMOJI_PATTERN",
    "EmojiDetector",
    "PhraseSetDetector",
    "RegexDetector",
    # llm detector
    "LLMJudgeDetector",
    "LLM_JUDGE_SYSTEM_PROMPT",
    # persona transfer
    "analyze_persona_transfer",
    # quirk detection
    "find_quirks",
    "QUIRK_PROMPTS",
    # suites
    "DEFAULT_SUITE_KEYS",
    "AGENTIC_SIDE_EFFECTS_SUITE",
    "FORMATTING_TIC_SUITE",
    "GENERAL_WRITING_PROMPTS",
    "LEXICAL_TIC_SUITE",
    "LINKEDIN_SUITE",
    "LLM_JUDGE_TONE_SUITE",
    "MILD_INSULT_REACTIVITY_SUITE",
    "PUNCTUATION_TIC_SUITE",
    "REST_CAI_VALIDATION_SUITE",
    "REST_HYPOTHESIS_SUITE",
    "SELF_HARM_SAFETY_SUITE",
    "STANDARD_CONDITIONS",
    "SUITE_NOTES",
    "SUITE_REGISTRY",
    "TASK_FOCUS_SUITE",
    "TONE_TIC_SUITE",
    "TRUTHFULNESS_SUITE",
    "expand_suite_names",
]
