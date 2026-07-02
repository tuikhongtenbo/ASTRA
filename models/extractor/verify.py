"""Sanity-check extraction against the original question to detect hallucination."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from config.config import CONF_THRESHOLD_EXTRACT

try:
    from rapidfuzz.fuzz import partial_ratio
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False


@dataclass
class ExtractionResult:
    """
    Structured result from entity extraction.

    Fields:
        O1: subject object name
        O2: reference object name (None if viewer-relative / no explicit reference)
        O2_is_viewer: True if O2 refers to the observer's viewpoint
        confidence: LLM self-reported confidence (0.0–1.0)
        O1_hallucinated: True if O1 was not found in the question text
        O2_hallucinated: True if O2 was not found in the question text
        raw_json: raw model output (for debugging)
    """
    O1: str
    O2: Optional[str]
    O2_is_viewer: bool
    confidence: float
    O1_hallucinated: bool = False
    O2_hallucinated: bool = False
    raw_json: str = ""

    @property
    def is_valid(self) -> bool:
        """True if extraction passes confidence gate and no hallucination detected."""
        return (
            self.confidence >= CONF_THRESHOLD_EXTRACT
            and not self.O1_hallucinated
        )


def verify_extraction(question: str, result: ExtractionResult) -> ExtractionResult:
    """
    Check whether O1/O2 actually appear in the original question text.
    Hallucinated entities (not found in question) → force confidence to 0.0.
    O2 is skipped when O2_is_viewer=True (viewer references don't exist in image).

    Returns the same ExtractionResult with hallucination flags and adjusted confidence.
    """
    q_lower = question.lower()

    for key in ("O1", "O2"):
        val = getattr(result, key, None)
        if val is None or (key == "O2" and result.O2_is_viewer):
            continue

        # Extract core tokens: remove articles and pronouns
        tokens = [
            t.strip()
            for t in re.split(r"\s+", val.lower())
            if t.strip() not in {"the", "a", "an", "it", "you", "your", "i"}
        ]
        if not tokens:
            continue

        # 1. Literal token match in question
        found = any(len(t) >= 3 and t in q_lower for t in tokens)

        # 2. RapidFuzz fallback for fuzzy/substring matching
        if not found and HAS_RAPIDFUZZ:
            found = any(len(t) >= 3 and partial_ratio(t, q_lower) >= 75 for t in tokens)

        if not found:
            if key == "O1":
                result.O1_hallucinated = True
            else:
                result.O2_hallucinated = True
            result.confidence = 0.0

    return result
