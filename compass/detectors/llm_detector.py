"""LLM-judge-based tic detector."""
from dataclasses import dataclass
from typing import Optional

from compass.detectors.base import DetectorResult, JudgeFn, TicDetector
from compass.judges.parsing import parse_judge_response

LLM_JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator. Return only valid JSON with keys "
    "score, hit, confidence, and rationale."
)


@dataclass(frozen=True)
class LLMJudgeDetector(TicDetector):
    """Score subjective style categories with a separate judge model.

    Call detect_with_judge(text, judge_fn) rather than detect(text) directly.
    """
    name: str
    rubric: str
    hit_threshold: float = 0.5
    max_tokens: int = 180

    def build_prompt(self, text: str) -> str:
        return (
            "Evaluate the assistant response below using this rubric.\n\n"
            f"Rubric:\n{self.rubric}\n\n"
            "Return JSON exactly in this shape:\n"
            '{"score": 0.0, "hit": false, "confidence": 0.0, "rationale": "short reason"}\n\n'
            "Assistant response:\n"
            f"{text}"
        )

    def detect(self, text: str) -> DetectorResult:
        raise RuntimeError(
            f"LLMJudgeDetector {self.name!r} requires a judge function; "
            "call detect_with_judge(text, judge_fn) instead."
        )

    def detect_with_judge(self, text: str, judge_fn: JudgeFn) -> DetectorResult:
        raw = judge_fn(self.build_prompt(text), self.max_tokens)
        return self.parse_response(raw)

    @staticmethod
    def _extract_json_object(raw: str) -> Optional[dict]:
        return parse_judge_response(raw)

    def parse_response(self, raw: str) -> DetectorResult:
        payload = parse_judge_response(raw)

        if payload is None:
            return DetectorResult(
                name=self.name,
                count=0,
                score=0.0,
                hit=False,
                metadata={"parse_error": True, "raw": raw[:500]},
            )

        try:
            score = float(payload.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(1.0, score))
        hit = bool(payload.get("hit", score >= self.hit_threshold))
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
