"""
prompt_v2.py — ASTRA v2 prompt templates.
Thiết kế: dùng str.format() với dict đầy đủ field, default rỗng cho optional placeholders.
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
# Prompt templates (dùng str.format())
# ---------------------------------------------------------------------------

_PROMPT_FULL = """\
You are given 2 images of the same scene:
- Image 1: the photo with two reference points marked — [1] = {o1_name}{opt_o2_mark}.
- Image 2: an auxiliary depth heatmap of the same scene, where brighter/warmer colors = closer to
  the camera and darker/cooler colors = farther. A colorbar legend (Near -> Far) is shown in the
  bottom-right corner. This is an estimate from an external depth model and may be inaccurate —
  cross-check it against what you directly observe in Image 1 before concluding.

  Estimated relative depth (lower = closer to camera): [1]≈{depth_o1:.2f}{opt_o2_depth}
  {viewer_note}
  Depth hint: {depth_relation}

Question: {question}

Options:
{options_block}

Think briefly, then answer in exactly this format:
Answer: (X)"""

_PROMPT_FALLBACK = """\
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
    Build prompt cho VLM dựa trên trạng thái pipeline.

    Case đầy đủ (marks_ok=True):
      - 2 ảnh đầu vào: Image1=bbox-marked, Image2=depth-heatmap
      - Prompt đầy đủ với depth numbers + depth_relation_text

    Case fallback (marks_ok=False):
      - Chỉ question + options, không mô tả ảnh, không depth cue

    Args:
        record: dict từ test_objects_last.json
        marks_ok: True nếu M1 detect thành công
        depth_o1, depth_o2: giá trị depth đã tính (chỉ dùng khi marks_ok=True)
        options: list options (lấy từ record nếu None)
    """
    if options is None:
        options = record.get("options", [])

    o1_name = record.get("Object", {}).get("O1", "")
    o2_name = record.get("Object", {}).get("O2", "")
    is_viewer = bool(record.get("O2_is_viewer", False))

    options_block = format_options(options)
    question = record.get("question", "")

    if not marks_ok:
        return _PROMPT_FALLBACK.format(
            question=question,
            options_block=options_block,
        )

    # ── Case đầy đủ ───────────────────────────────────────────────────────
    # opt_o2_mark: mô tả [2] trong Image 1
    if is_viewer:
        opt_o2_mark = ""
        depth_o2_display = 0.0  # viewer depth = 0.0
    else:
        opt_o2_mark = f', [2] = {o2_name}'
        depth_o2_display = depth_o2

    # opt_o2_depth: giá trị số [2] trong text
    if is_viewer:
        opt_o2_depth = ""
        viewer_note = (
            f"Note: {o2_name} refers to your own viewpoint (the camera position), which is "
            f"not a physical object and is therefore not marked in the images. "
            f"Its reference depth is defined as 0 (the camera plane)."
        )
    else:
        opt_o2_depth = f", [2]≈{depth_o2:.2f}"
        viewer_note = ""

    # depth_relation_text
    from config.pipeline_config import DEPTH_DIFF_THRESHOLD
    from models.image_generator import depth_relation_text as _drt
    if is_viewer:
        effective_o2_name = o2_name
    else:
        effective_o2_name = o2_name
    depth_relation = _drt(depth_o1, depth_o2_display, o1_name, effective_o2_name, DEPTH_DIFF_THRESHOLD)

    return _PROMPT_FULL.format(
        o1_name=o1_name,
        opt_o2_mark=opt_o2_mark,
        opt_o2_depth=opt_o2_depth,
        viewer_note=viewer_note,
        depth_o1=depth_o1,
        depth_relation=depth_relation,
        question=question,
        options_block=options_block,
    )
