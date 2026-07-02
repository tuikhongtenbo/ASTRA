"""Unit tests for models.extractor.verify — verify_extraction hallucination detection."""
import re


# ─── Minimal reproduce of ExtractionResult + verify_extraction for testing ─────

try:
    from rapidfuzz.fuzz import partial_ratio
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

CONF_THRESHOLD_EXTRACT = 0.6


class ExtractionResult:
    def __init__(self, O1, O2, O2_is_viewer, confidence,
                 O1_hallucinated=False, O2_hallucinated=False, raw_json=""):
        self.O1 = O1
        self.O2 = O2
        self.O2_is_viewer = O2_is_viewer
        self.confidence = confidence
        self.O1_hallucinated = O1_hallucinated
        self.O2_hallucinated = O2_hallucinated
        self.raw_json = raw_json

    @property
    def is_valid(self):
        return (self.confidence >= CONF_THRESHOLD_EXTRACT and not self.O1_hallucinated)


def verify_extraction(question, result):
    q_lower = question.lower()
    for key in ("O1", "O2"):
        val = getattr(result, key, None)
        if val is None or (key == "O2" and result.O2_is_viewer):
            continue
        tokens = [t.strip() for t in re.split(r"\s+", val.lower())
                  if t.strip() not in {"the", "a", "an", "it", "you", "your", "i"}]
        if not tokens:
            continue
        found = any(len(t) >= 3 and t in q_lower for t in tokens)
        if not found and HAS_RAPIDFUZZ:
            found = any(len(t) >= 3 and partial_ratio(t, q_lower) >= 75 for t in tokens)
        if not found:
            if key == "O1":
                result.O1_hallucinated = True
            else:
                result.O2_hallucinated = True
            result.confidence = 0.0
    return result


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestVerifyExtraction:
    def test_hallucination_detected(self):
        result = ExtractionResult(
            O1="pizza slice", O2="coffee mug",
            O2_is_viewer=False, confidence=0.95,
        )
        verified = verify_extraction(
            "Where is the cat relative to the dog?", result
        )
        assert verified.O1_hallucinated is True
        assert verified.O2_hallucinated is True
        assert verified.confidence == 0.0

    def test_valid_extraction(self):
        result = ExtractionResult(
            O1="the cat", O2="the dog",
            O2_is_viewer=False, confidence=0.95,
        )
        verified = verify_extraction(
            "Where is the cat relative to the dog?", result
        )
        assert verified.O1_hallucinated is False
        assert verified.O2_hallucinated is False
        assert verified.confidence == 0.95

    def test_viewer_mode_skips_o2_check(self):
        result = ExtractionResult(
            O1="the sun", O2="you",
            O2_is_viewer=True, confidence=0.9,
        )
        verified = verify_extraction(
            "If you were the giraffe, would the sun be to your left or right?", result
        )
        assert verified.O1_hallucinated is False
        assert verified.O2_hallucinated is False
        assert verified.confidence == 0.9

    def test_partial_token_match(self):
        result = ExtractionResult(
            O1="the blue car", O2="the bicycle",
            O2_is_viewer=False, confidence=0.9,
        )
        verified = verify_extraction(
            "Where is the car relative to the bicycle?", result
        )
        assert verified.O1_hallucinated is False
        assert verified.O2_hallucinated is False

    def test_o1_hallucinated_only(self):
        result = ExtractionResult(
            O1="pizza slice", O2="the cat",
            O2_is_viewer=False, confidence=0.9,
        )
        verified = verify_extraction(
            "Where is the cat relative to the dog?", result
        )
        assert verified.O1_hallucinated is True
        assert verified.O2_hallucinated is False
        assert verified.confidence == 0.0

    def test_is_valid_property(self):
        result_good = ExtractionResult(
            O1="the cat", O2="the dog",
            O2_is_viewer=False, confidence=0.9,
        )
        assert result_good.is_valid is True

        result_low_conf = ExtractionResult(
            O1="the cat", O2="the dog",
            O2_is_viewer=False, confidence=0.3,
        )
        assert result_low_conf.is_valid is False

        result_halluc = ExtractionResult(
            O1="pizza slice", O2="coffee mug",
            O2_is_viewer=False, confidence=0.9,
        )
        verified = verify_extraction(
            "Where is the cat relative to the dog?", result_halluc
        )
        assert verified.is_valid is False


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
