"""Heuristic (non-LLM) tic detectors: regex, phrase sets, emoji, character counts."""
import re
from dataclasses import dataclass
from typing import Dict, Iterable

from compass.detectors.base import DetectorResult, TicDetector


@dataclass(frozen=True)
class RegexDetector(TicDetector):
    """Count regex matches; useful for lexical, punctuation, and format tics."""
    name: str
    pattern: str
    flags: int = re.IGNORECASE
    hit_threshold: int = 1

    def detect(self, text: str) -> DetectorResult:
        matches = list(re.finditer(self.pattern, text, flags=self.flags))
        count = len(matches)
        return DetectorResult(
            name=self.name,
            count=count,
            score=float(count),
            hit=count >= self.hit_threshold,
            metadata={"matches": [m.group(0) for m in matches[:20]]},
        )


@dataclass(frozen=True)
class PhraseSetDetector(TicDetector):
    """Count case-insensitive whole-phrase occurrences from a fixed phrase set."""
    name: str
    phrases: Iterable[str]
    hit_threshold: int = 1

    def detect(self, text: str) -> DetectorResult:
        phrase_hits: Dict[str, int] = {}
        total = 0
        for phrase in self.phrases:
            pattern = rf"\b{re.escape(phrase)}\b"
            count = len(re.findall(pattern, text, flags=re.IGNORECASE))
            if count:
                phrase_hits[phrase] = count
                total += count
        return DetectorResult(
            name=self.name,
            count=total,
            score=float(total),
            hit=total >= self.hit_threshold,
            metadata={"phrases": phrase_hits},
        )


EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "♀-♂"
    "☀-⭕"
    "‍"
    "⏏"
    "⏩"
    "⌚"
    "️"
    "〰"
    "]",
    flags=re.UNICODE,
)


@dataclass(frozen=True)
class EmojiDetector(TicDetector):
    """Count emoji occurrences."""
    name: str = "emoji"
    hit_threshold: int = 1

    def detect(self, text: str) -> DetectorResult:
        count = len(EMOJI_PATTERN.findall(text))
        return DetectorResult(
            name=self.name,
            count=count,
            score=float(count),
            hit=count >= self.hit_threshold,
            metadata={},
        )


@dataclass(frozen=True)
class CharacterCountDetector(TicDetector):
    """Count a literal character, such as em dashes or semicolons."""
    name: str
    character: str
    hit_threshold: int = 1

    def detect(self, text: str) -> DetectorResult:
        count = text.count(self.character)
        return DetectorResult(
            name=self.name,
            count=count,
            score=float(count),
            hit=count >= self.hit_threshold,
            metadata={"character": self.character},
        )
