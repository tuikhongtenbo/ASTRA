"""
Regex-based parser: extract O1/O2 from raw model output text.
Used when model does not return valid JSON (fallback / debugging).
"""
from __future__ import annotations

import json
import re
from typing import Optional


def parse_raw_output(raw: str) -> Optional[dict]:
    """
    Extract O1/O2 from raw model output text using robust regex patterns.

    Tries in order:
      1. JSON object block
      2. Line-by-line key-value regex
      3. Structural patterns (e.g. "O1: <text>", "Subject: <text>")

    Returns {"O1": str, "O2": str|null} or None if nothing found.
    """
    raw = raw.strip()

    # ── Pattern 1: JSON block ──────────────────────────────────────────────
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            o1 = obj.get("O1") or obj.get("o1") or obj.get("subject")
            o2 = obj.get("O2") or obj.get("o2") or obj.get("reference")
            if o1:
                return {"O1": o1, "O2": o2}
        except Exception:
            pass

    # ── Pattern 2: quoted key-value lines ───────────────────────────────────
    result = {}
    for line in raw.splitlines():
        line = line.strip()
        for key, field in [
            ("o1", "O1"), ("subject", "O1"),
            ("o2", "O2"), ("reference", "O2"),
        ]:
            m2 = re.search(
                rf'["\']?\s*{re.escape(key)}\s*["\']?\s*[:=]\s*["\']?([^"\'\n,}}]+)["\']?',
                line, re.IGNORECASE,
            )
            if m2:
                result[field] = m2.group(1).strip().strip("'\"")

    if result:
        return result

    # ── Pattern 3: "The O1 is ... relative to the O2" ──────────────────────
    m3 = re.search(
        r"(?:subject|object|o1)\s*[:\-]\s*(.+?)(?:\s+relative\s+to|\s+vs\.|\s+vs\s)",
        raw, re.IGNORECASE,
    )
    m4 = re.search(
        r"(?:reference|target|o2)\s*[:\-]\s*(.+?)(?:\.|$|\n)",
        raw, re.IGNORECASE,
    )
    if m3 or m4:
        return {
            "O1": m3.group(1).strip() if m3 else None,
            "O2": m4.group(1).strip() if m4 else None,
        }

    return None
