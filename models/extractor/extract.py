"""Entity extraction: LLM-based (primary) + regex legacy (fallback for ablation)."""
from __future__ import annotations

import json
import os
from typing import Optional

from .llm_client import TextExtractorClient, build_extractor_prompt, parse_extraction_json
from .verify import ExtractionResult, verify_extraction
from config.config import (
    DASHSCOPE_BASE_URL, EXTRACTOR_API_KEY, EXTRACTOR_MAX_TOKENS,
    EXTRACTOR_MAX_RETRIES, EXTRACTION_OUTPUT_FILE,
)
from ..module1_ogm import extract_entities_from_question as _regex_legacy


# ─── Singleton client ────────────────────────────────────────────────────────

_client: Optional[TextExtractorClient] = None


def _get_client() -> Optional[TextExtractorClient]:
    global _client
    if _client is None:
        api_key = (
            EXTRACTOR_API_KEY
            or os.getenv("EXTRACTOR_API_KEY", "").strip()
            or os.getenv("QWEN_API_KEY", "").strip()
            or os.getenv("DASHSCOPE_API_KEY", "").strip()
        )
        if not api_key:
            return None
        base_url = os.getenv("DASHSCOPE_BASE_URL", "").strip() or DASHSCOPE_BASE_URL
        _client = TextExtractorClient(api_key, base_url)
    return _client


# ─── Primary extraction ──────────────────────────────────────────────────────

def extract_entities_llm(question: str) -> ExtractionResult:
    """
    Primary entity extractor: LLM via DashScope (qwen3.7-max).

    Falls back to regex legacy if no API key is configured.
    Runs hallucination verification before returning.
    """
    client = _get_client()
    if client is None:
        O1, O2 = extract_entities_regex_legacy(question)
        return ExtractionResult(
            O1=O1 or "", O2=O2, O2_is_viewer=False,
            confidence=0.3, raw_json="",
        )

    result = client.extract(question, max_retries=EXTRACTOR_MAX_RETRIES)
    result = verify_extraction(question, result)
    return result


def extract_entities_regex_legacy(question: str) -> tuple[Optional[str], Optional[str]]:
    """
    Legacy regex-based extractor.
    Kept for ablation comparison and as fallback when API key is missing.
    """
    return _regex_legacy(question)


# ─── Resume support for batch extraction ────────────────────────────────────

def load_existing_extractions(output_file: str) -> dict:
    """
    Load already-extracted samples from JSON file.
    Returns {id: extraction_dict}. Supports both list-of-dicts and dict-of-dicts formats.
    """
    if not os.path.exists(output_file):
        return {}
    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {str(item.get("id", item.get("question_id", i))): item for i, item in enumerate(data)}
    return {str(k): v for k, v in data.items()}


def save_extractions(all_results: list, output_file: str):
    """
    Save batch extraction results to JSON file (merge with existing).
    Uses dict keyed by id so duplicates are overwritten.
    """
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    existing = load_existing_extractions(output_file)
    for r in all_results:
        eid = str(r.get("id", r.get("question_id", 0)))
        existing[eid] = r
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
