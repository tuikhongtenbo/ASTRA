"""
Prompts — Prompt templates for ASTRA inference.
Design principles:
  - English-only, unambiguous instructions.
  - Explicit output format requirement to maximize parse success.
  - Depth cue injected as a neutral spatial hint, not a directive.
  - Options always prefixed with a letter (A/B/C...) for reliable letter-based parsing.
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_options(options: list[str]) -> str:
    """Format options as (A) option1 | (B) option2 | ... corresponding to the sample's options."""
    letters = ["A", "B", "C", "D", "E", "F"]
    parts = []
    for i, opt in enumerate(options):
        letter = letters[i] if i < len(letters) else str(i + 1)
        parts.append(f"({letter}) {opt}")
    return " | ".join(parts)


def get_option_letter(answer: str, options: list[str]) -> str:
    """Return the letter (A/B/C...) for a given answer text."""
    letters = ["A", "B", "C", "D", "E", "F"]
    for i, opt in enumerate(options):
        if opt.strip().lower() == answer.strip().lower():
            return letters[i] if i < len(letters) else str(i + 1)
    return "A"


# ---------------------------------------------------------------------------
# Core prompt builders
# ---------------------------------------------------------------------------

def build_full_prompt(
    O1_name: Optional[str],
    O2_name: Optional[str],
    depth_cue: Optional[str],
    question: str,
    options: list[str],
) -> str:
    """
    Full ASTRA prompt: OGM visual markers + optional depth cue + question.
    Used when at least one of O1/O2/depth_cue is available.
    """
    lines = []

    # Visual grounding (from Set-of-Mark OGM)
    if O1_name and O2_name:
        lines.append(
            f"Examine the image. The object marked [1] is the {O1_name}. "
            f"The object marked [2] is the {O2_name}."
        )
    else:
        lines.append("Examine the image.")

    # Depth hint (from DLC) — injected as a neutral spatial cue
    if depth_cue:
        lines.append(depth_cue)

    # Question block
    lines.append("")
    lines.append(f"Question: {question}")
    lines.append("")

    # Options parsed dynamically from the specific sample
    lines.append("Answer choices:")
    lines.append(format_options(options))
    lines.append("")

    # Output instruction — explicit requirement to select exactly 1 option from the sample's choices
    lines.append(
        "Please select exactly one correct answer from the provided options above for this question.\n"
        "Output format: Answer: (LETTER) relation\n"
        "Example: Answer: (A) left of"
    )

    return "\n".join(lines)


def build_baseline_prompt(question: str, options: list[str]) -> str:
    """
    Baseline prompt: vanilla VLM prompt, no auxiliary cues.
    Used when M1 and M2 are disabled.
    """
    lines = []
    lines.append(f"Question: {question}")
    lines.append("")
    lines.append("Answer choices:")
    lines.append(format_options(options))
    lines.append("")
    lines.append(
        "Please select exactly one correct answer from the provided options above for this question.\n"
        "Output format: Answer: (LETTER) relation\n"
        "Example: Answer: (B) right of"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Legacy aliases (kept for backward compatibility with standalone module3_odv.run_odv)
# ---------------------------------------------------------------------------

build_astra_prompt = build_full_prompt  # noqa: N816
