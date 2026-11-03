"""LLM-judge-based tic detector."""
from dataclasses import dataclass
from typing import Optional

from compass.detectors.base import DetectorResult, JudgeFn, TicDetector
from compass.judges.parsing import parse_judge_response, parse_scored_response

from compass.judges.context import build_judge_prompt, JUDGE_SYSTEM_PROMPT

LLM_JUDGE_SYSTEM_PROMPT = JUDGE_SYSTEM_PROMPT

@dataclass(frozen=True)
class LLMJudgeDetector(TicDetector):
    """Score subjective style categories with a separate judge model.

    Call detect_with_judge(text, judge_fn) rather than detect(text) directly.
    """
    name: str
    rubric: str
    hit_threshold: float = 0.5
    max_tokens: int = 180

    def build_prompt(self, text: str, context: Optional[str] = None, *, system_context: Optional[str] = None) -> str:
        return build_judge_prompt(text, self.rubric, context, system_context)

    def detect(self, text: str) -> DetectorResult:
        raise RuntimeError(
            f"LLMJudgeDetector {self.name!r} requires a judge function; "
            "call detect_with_judge(text, judge_fn) instead."
        )

    def detect_with_judge(self, text: str, judge_fn: JudgeFn, *, context: Optional[str] = None, system_context: Optional[str] = None) -> DetectorResult:
        raw = judge_fn(self.build_prompt(text, context, system_context=system_context), self.max_tokens)
        result = self.parse_response(raw)
        result.metadata["judge_context"] = {"user_request": context, "system_prompt": system_context}
        return result

    @staticmethod
    def _extract_json_object(raw: str) -> Optional[dict]:
        return parse_judge_response(raw)

    def parse_response(self, raw: str) -> DetectorResult:
        payload = parse_scored_response(raw)
        score = payload["score"]
        hit = score >= self.hit_threshold
        return DetectorResult(
            name=self.name,
            count=1 if hit else 0,
            score=score,
            hit=hit,
            metadata={
                "confidence": payload.get("confidence"),
                "rationale": str(payload.get("rationale", ""))[:500],
                "raw": raw[:500],
            },
        )
