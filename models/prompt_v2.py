"""
prompt_v2.py - ASTRA v2 prompt templates.

Builds the two-image prompt used by v2 inference. The caller provides images
separately; this module only formats the text instruction.
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_options(options: list[str]) -> str:
    """Format options as (A) option1 | (B) option2 | ..."""
    letters = ["A", "B", "C", "D", "E", "F"]
    parts = []
    for i, opt in enumerate(options):
        letter = letters[i] if i < len(letters) else str(i + 1)
        parts.append(f"({letter}) {opt}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_PROMPT = """\
You are given 2 images of the same scene:
- Image 1: the original RGB photo with two detected objects marked.
- Image 2: an auxiliary depth heatmap of the same scene with the same detected objects marked.

Objects:
[1] = {o1_name}
[2] = {o2_name}

{relation_guidance}

Question: {question}

Options:
{options_block}

Think briefly, then answer in exactly this format:
Answer: (X)"""


def build_prompt(
    record: dict,
    marks_ok: bool,
    depth_o1: float,
    depth_o2: float,
    options: Optional[list[str]] = None,
) -> str:
    """
    Build the ASTRA v2 prompt for a record.

    The marks/depth arguments are kept for compatibility with existing callers,
    but v2 now always uses the same two-image prompt without fallback text or
    numeric depth hints.
    """
    del marks_ok, depth_o1, depth_o2

    if options is None:
        options = record.get("options", [])

    objects = record.get("Object", {})
    o1_name = objects.get("O1", "")
    o2_name = objects.get("O2", "")
    if record.get("O2_is_viewer", False):
        relation_guidance = (
            "Perspective:\n"
            f"[2] represents the current position/viewpoint of {o2_name}. "
            "Answer where [1] is located relative to [2] from that viewpoint."
        )
    else:
        relation_guidance = (
            "Spatial relation:\n"
            "Answer where [1] is located relative to [2]."
        )

    return _PROMPT.format(
        o1_name=o1_name,
        o2_name=o2_name,
        relation_guidance=relation_guidance,
        question=record.get("question", ""),
        options_block=format_options(options),
    )
