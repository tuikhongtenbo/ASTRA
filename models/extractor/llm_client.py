"""LLM client for text-only entity extraction via DashScope OpenAI-compatible API."""
from __future__ import annotations

import json
import re
import time

from config.config import DASHSCOPE_BASE_URL


# ─── Few-shot prompt (8 examples from train/dev.jsonl) ───────────────────────

_EXAMPLES = [
    (
        "If you were the giraffe in the image, would the sun be to your left or right?",
        {"O1": "the sun", "O2": "the giraffe", "O2_is_viewer": True, "confidence": 0.95},
    ),
    (
        "If you are the largest elephant in the image, where is the smallest elephant located relative to you?",
        {"O1": "the smallest elephant", "O2": "the largest elephant", "O2_is_viewer": True, "confidence": 0.95},
    ),
    (
        "Where is the black cat located relative to the computer screen?",
        {"O1": "the black cat", "O2": "the computer screen", "O2_is_viewer": False, "confidence": 0.97},
    ),
    (
        "For the clock on the wall in the image, does the hour hand point to the left or right of the 11 mark?",
        {"O1": "the hour hand", "O2": "the 11 mark", "O2_is_viewer": False, "confidence": 0.90},
    ),
    (
        "If you were the man in the image, where would your bookshelf be?",
        {"O1": "the bookshelf", "O2": "the man", "O2_is_viewer": True, "confidence": 0.93},
    ),
    (
        "Where is the pumpkin located relative to the dinosaur toy?",
        {"O1": "the pumpkin", "O2": "the dinosaur toy", "O2_is_viewer": False, "confidence": 0.96},
    ),
    (
        "For all the white letters on the green sign in the image, where is the letter k located relative to the letter R?",
        {"O1": "the letter k", "O2": "the letter R", "O2_is_viewer": False, "confidence": 0.95},
    ),
    (
        "If you were the person standing in the image, where would the podium be located relative to you?",
        {"O1": "the podium", "O2": "the person", "O2_is_viewer": True, "confidence": 0.95},
    ),
]


def build_extractor_prompt(question: str) -> str:
    """Build few-shot user prompt for entity extraction."""
    lines = [
        "You are an entity extraction system for spatial-relation questions about images.",
        "Given a question, extract:\n",
        "- O1: the subject object (the thing whose location is being asked about)\n",
        "- O2: the reference object (the thing O1's position is relative to). "
        "Set to null if there is no explicit reference object other than the viewer.\n",
        "- O2_is_viewer: true if O2 refers to 'you' / the observer's own viewpoint "
        "(perspective-taking questions), not a physical object in the image.\n",
        "- confidence: your confidence (0.0-1.0) in this extraction.\n",
        "Respond ONLY with JSON matching this schema: "
        '{"O1": str, "O2": str|null, "O2_is_viewer": bool, "confidence": float}\n',
        "Examples:\n",
    ]
    for q, a in _EXAMPLES:
        lines.append(f'Q: "{q}"')
        lines.append(f"A: {json.dumps(a)}\n")
    lines.append(f'Now extract from this question:\nQ: "{question}"\nA:')
    return "\n".join(lines)


def parse_extraction_json(raw: str):
    """
    Parse JSON from model output. Regex fallback if JSON decode fails.
    Supports both {O1, O2, O2_is_viewer, confidence} and legacy keys.
    Returns an ExtractionResult-compatible dict.
    """
    raw = raw.strip()

    # Strip markdown code blocks
    for prefix in ("```json", "```"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
        if raw.endswith("```"):
            raw = raw[:-3]
    raw = raw.strip()

    # Try direct JSON
    try:
        obj = json.loads(raw)
        return _dict_to_result(obj, raw)
    except json.JSONDecodeError:
        pass

    # Regex fallback: parse key-value pairs line by line
    obj = {}
    for line in raw.splitlines():
        m = re.search(
            r'"(O1|O2|O2_is_viewer|confidence|subject|reference)"\s*:\s*"?([^",}\]]+)"?',
            line, re.IGNORECASE,
        )
        if m:
            k, v = m.group(1), m.group(2).strip()
            kl = k.lower()
            if kl in ("o1", "subject"):
                obj["O1"] = v.strip('"').strip("'")
            elif kl in ("o2", "reference"):
                obj["O2"] = v.strip('"').strip("'")
            elif kl == "o2_is_viewer":
                obj["O2_is_viewer"] = v.lower() in ("true", "1", "yes")
            elif kl == "confidence":
                try:
                    obj["confidence"] = float(v)
                except ValueError:
                    obj["confidence"] = 0.0

    return _dict_to_result(obj, raw)


def _dict_to_result(obj: dict, raw: str):
    """Convert parsed dict to ExtractionResult-compatible fields."""
    from .verify import ExtractionResult
    o2_val = obj.get("O2", obj.get("reference"))
    if isinstance(o2_val, str) and o2_val.lower() in ("null", "none", ""):
        o2_val = None
    return ExtractionResult(
        O1=str(obj.get("O1", obj.get("subject", ""))),
        O2=o2_val,
        O2_is_viewer=bool(obj.get("O2_is_viewer", False)),
        confidence=float(obj.get("confidence", 0.0)),
        raw_json=raw,
    )


class TextExtractorClient:
    """
    Text-only LLM client for entity extraction via DashScope.

    Lazily imports openai to avoid ImportError when the package is not installed.
    """

    def __init__(self, api_key: str, base_url: str = DASHSCOPE_BASE_URL):
        self.api_key = api_key
        self.base_url = base_url
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def extract(self, question: str, max_retries: int = 2):
        """
        Extract O1/O2 from a question using the LLM.
        Returns ExtractionResult with raw_json populated.
        """
        from .verify import ExtractionResult

        messages = [
            {"role": "system", "content": "You are a precise entity extraction system. Output ONLY valid JSON."},
            {"role": "user",   "content": build_extractor_prompt(question)},
        ]

        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model="qwen3.7-max",
                    messages=messages,
                    max_tokens=64,
                    temperature=0.0,
                    extra_body={"enable_thinking": False},
                )
                raw = resp.choices[0].message.content.strip()
                result = parse_extraction_json(raw)
                result.raw_json = raw
                return result
            except Exception:
                if attempt == max_retries - 1:
                    return ExtractionResult(
                        O1="", O2=None, O2_is_viewer=False,
                        confidence=0.0, raw_json="",
                    )
                time.sleep(2)
