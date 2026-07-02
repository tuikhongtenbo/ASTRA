"""
Module 1 — Object-Grounded Marking (OGM)
Detect bbox cho O1, O2 bằng Grounding DINO, vẽ [1], [2] lên ảnh.
"""

from __future__ import annotations

import re
from typing import Optional

import torch
from PIL import Image, ImageDraw, ImageFont

from config.config import COCO_CATEGORIES, CONFIDENCE_THRESHOLD, DET_CONF_THRESHOLD

# ─── Legacy alias (ablation compatibility) ──────────────────────────────────

def extract_entities_regex_legacy(question: str) -> tuple[Optional[str], Optional[str]]:
    """Alias: rename of extract_entities_from_question for ablation clarity."""
    return extract_entities_from_question(question)

# Keep original name too for backward compatibility
extract_entities_from_question = extract_entities_regex_legacy


def extract_entities_from_question(question: str) -> tuple[Optional[str], Optional[str]]:
    """
    Extract (O1, O2) entities from a SpatialMQA question.

    Patterns covered:
      1. "If you were/are [ROLE], where is [X] located relative to you?"
         → O1=X, O2=you (the described role)
      2. "If you were/are [ROLE], [DESC] relative to you?"
         → O1=subject_of_description, O2=you
      3. "On which side of [X] is [Y] located?"
         → O1=Y, O2=X
      4. "Where is [X] located relative to [Y]?"
         → O1=X, O2=Y
      5. "Where is [X] located?"  (single object)
         → O1=X, O2=None
      6. "For the [desc], does the [X] point [above/below/left/right] of the [Y]?"
         → O1=X, O2=Y
      7. "For the [desc], where is [X] located relative to the [Y]?"
         → O1=X, O2=Y
    """
    q = question.strip()
    q_lower = q.lower()

    # ── Pattern 1: "If you were/are [ROLE], where is [X] located relative to you?" ──
    # Non-greedy `(.+?)` inside the optional group must NOT eat past "where is/would"
    # Use a constrained pattern: "where is" or "where would SOMETHING be"
    m = re.search(
        r'^If\s+you\s+(?:were|are)\s+(?:the\s+)?'
        r'(.+?)'                           # group(1): role description (non-greedy, stops at comma)
        r',\s*'
        r'(?:where\s+(?:is\s+|(?:would\s+[^\s]+\s+be\s+)))?'  # optional "where is / where would X be"
        r'(?:the\s+)?'
        r'(.+?)'                           # group(2): entity being asked about (non-greedy)
        r'\s+located\s+(?:relative\s+to|in\s+relation\s+to|on\s+you|in\s+you)\??$',
        q, re.IGNORECASE
    )
    if m:
        role_desc = m.group(1).strip()
        located = m.group(2).strip()
        skip = {"it", "you"}
        if located.lower() not in skip and len(located) > 1:
            return _clean_entity(located), _clean_entity(role_desc)
        return _clean_entity(role_desc), None

    # Pattern 1b: "If you were/are [ROLE], where is [X]? (no "located relative to")
    # e.g. "If you are the person in the image, where is your shadow?"
    # e.g. "If you were the girl in the image, where would the dog be in you?"
    m = re.search(
        r'^If\s+you\s+(?:were|are)\s+(?:the\s+)?'
        r'(.+?)'
        r',\s*where\s+(?:is|would\s+(?:[^\s]+\s+)?be\s+)'
        r'(?:your\s+)?'
        r'(.+?)'
        r'\s*(?:located\s+)?(?:relative\s+to|in\s+relation\s+to|on\s+you|in\s+you)?\??$',
        q, re.IGNORECASE
    )
    if m:
        role_desc = m.group(1).strip()
        entity = m.group(2).strip()
        skip = {"you", "it"}
        if entity.lower() not in skip and len(entity) > 1:
            entity = re.sub(r'^your\s+', '', entity, flags=re.IGNORECASE)
            return _clean_entity(entity), _clean_entity(role_desc)
        return _clean_entity(role_desc), None

    # Pattern 1c: "If you were/are [ROLE], on which side of you is [X]?"
    m = re.search(
        r'^If\s+you\s+(?:were|are)\s+(?:the\s+)?'
        r'(.+?)'
        r',\s*on\s+which\s+side\s+of\s+you\s+'
        r'(?:is|would\s+(?:the\s+)?(.+?)\s+be)',
        q, re.IGNORECASE
    )
    if m:
        role = m.group(1).strip()
        entity = (m.group(2) or "").strip()
        if entity:
            return _clean_entity(entity), _clean_entity(role)
        return _clean_entity(role), None

    # Pattern 2c: "If you were [ROLE], [QUESTION about you]" (no "relative to you")
    # e.g. "If you were a person walking on the beach, on which side of you would the water be?"
    m = re.search(
        r'^If\s+you\s+(?:were|are)\s+(?:a\s+)?(?:person\s+)?(?:walking\s+)?(?:on\s+)?(?:the\s+)?'
        r'(?:beach\s+)?,?\s*'
        r'(?:on\s+which\s+side\s+of\s+you\s+(?:is|would)\s+(?:the\s+)?(.+?)\s+be)',
        q, re.IGNORECASE
    )
    if m:
        located = m.group(1).strip()
        return _clean_entity(located), "you"

    # ── Pattern 3: "On which side of [X] is [Y] located?" ──
    m = re.search(
        r'^On\s+which\s+side\s+of\s+(?:the\s+)?(.+?)\s+is\s+(?:the\s+)?(.+?)\s+located',
        q, re.IGNORECASE
    )
    if m:
        ref = m.group(1).strip()   # X = reference (side of X)
        subj = m.group(2).strip()  # Y = subject (is Y located)
        return _clean_entity(subj), _clean_entity(ref)

    # ── Pattern 4: "Where is [X] located relative to [Y]?" ──
    m = re.search(
        r'^Where\s+is\s+(?:the\s+)?(.+?)\s+located\s+relative\s+to\s+(?:the\s+)?(.+?)\??$',
        q, re.IGNORECASE
    )
    if m:
        return _clean_entity(m.group(1).strip()), _clean_entity(m.group(2).strip())

    # Pattern 4b: "Where are the [X] located relative to [Y]?"
    m = re.search(
        r'^Where\s+are\s+(?:the\s+)?(.+?)\s+located\s+relative\s+to\s+(?:the\s+)?(.+?)\??$',
        q, re.IGNORECASE
    )
    if m:
        return _clean_entity(m.group(1).strip()), _clean_entity(m.group(2).strip())

    # Pattern 4c: "from your perspective, which side is [X] relative to [Y]?"
    m = re.search(
        r'(?:from\s+your\s+perspective,?\s*)?which\s+side\s+is\s+(?:the\s+)?(.+?)\s+'
        r'(?:located\s+)?relative\s+to\s+(?:the\s+)?(.+?)\??$',
        q, re.IGNORECASE
    )
    if m:
        return _clean_entity(m.group(1).strip()), _clean_entity(m.group(2).strip())

    # ── Pattern 5: "Where is [X] located?" (single object) ──
    m = re.search(
        r'^Where\s+is\s+(?:the\s+)?(.+?)\s+located\??$',
        q, re.IGNORECASE
    )
    if m:
        return _clean_entity(m.group(1).strip()), None

    # ── Pattern 6: "For the [desc], does the [X] point [above/below/left/right] of the [Y]?" ──
    m = re.search(
        r'^For\s+(?:the\s+)?.+?,\s+does\s+(?:the\s+)?(.+?)\s+point\s+'
        r'(?:to\s+(?:the\s+)?)?(?:the\s+)?(.+?)\s*(?:\??$)',
        q, re.IGNORECASE
    )
    if m:
        return _clean_entity(m.group(1).strip()), _clean_entity(m.group(2).strip())

    # Pattern 6b: "does the [X] point above or below the [Y]?"
    m = re.search(
        r'^does\s+(?:the\s+)?(.+?)\s+point\s+(?:above|below)\s+(?:or\s+(?:below|above)\s+)?'
        r'(?:the\s+)?(.+?)\??$',
        q, re.IGNORECASE
    )
    if m:
        return _clean_entity(m.group(1).strip()), _clean_entity(m.group(2).strip())

    # ── Pattern 7: "For the [desc], where is [X] located relative to the [Y]?" ──
    m = re.search(
        r'^For\s+(?:the\s+)?.+?,\s+where\s+is\s+(?:the\s+)?(.+?)\s+located\s+relative\s+to\s+'
        r'(?:the\s+)?(.+?)\??$',
        q, re.IGNORECASE
    )
    if m:
        return _clean_entity(m.group(1).strip()), _clean_entity(m.group(2).strip())

    # ── Pattern 7b: "For a clock on ..., does the [X] point [above/below] the [Y]?" ──
    m = re.search(
        r'For\s+(?:a\s+)?(?:the\s+)?(?:clock|watch).+?,\s+does\s+(?:the\s+)?(.+?)\s+'
        r'point\s+(?:to\s+(?:the\s+)?)?(?:above|below)\s+(?:or\s+(?:below|above)\s+)?'
        r'(?:the\s+)?(.+?)\??$',
        q, re.IGNORECASE
    )
    if m:
        return _clean_entity(m.group(1).strip()), _clean_entity(m.group(2).strip())

    # ── Pattern 8: "Is the [X] to the left or right of the [Y]?" ──
    m = re.search(
        r'^Is\s+(?:the\s+)?(.+?)\s+(?:to\s+)?the\s+(left|right)\s+of\s+(?:the\s+)?(.+?)\??$',
        q, re.IGNORECASE
    )
    if m:
        return _clean_entity(m.group(1).strip()), _clean_entity(m.group(3).strip())

    # ── Fallback: token-based extraction ──
    words = re.findall(r'\b[a-z]{3,}\b', q_lower)
    candidates = [w for w in words
                  if w not in {"the", "where", "which", "relative", "located", "side",
                               "left", "right", "above", "below", "front", "behind",
                               "in", "of", "on", "is", "are", "was", "were", "you",
                               "your", "would", "could", "should", "point", "does",
                               "for", "from", "this", "that", "these", "those",
                               "relative", "relation", "your", "perspective"}]
    if len(candidates) >= 2:
        return _clean_entity(candidates[-2]), _clean_entity(candidates[-1])
    if candidates:
        return _clean_entity(candidates[0]), None
    return None, None


def _clean_entity(entity: str) -> str:
    """Strip common determiners and descriptors from entity string."""
    if not entity:
        return entity
    entity = re.sub(r'^(the|a|an)\s+', '', entity.strip(), flags=re.IGNORECASE)
    entity = re.sub(r'\s+(is|are|was|were|located|relative\s+to|in\s+relation\s+to)\s*$',
                    '', entity.strip(), flags=re.IGNORECASE)
    return entity.strip()


def best_match(entity: str, vocabulary: list[str]) -> str:
    entity_lower = entity.lower().strip()
    if entity_lower in vocabulary:
        return entity_lower
    for v in vocabulary:
        if entity_lower in v or v in entity_lower:
            return v
    return entity_lower


def load_grounding_model(device: str = "cuda"):
    """
    Load Grounding DINO từ local repo clone + local weights.
    Cách cài đặt:
      git clone https://github.com/IDEA-Research/GroundingDINO.git
      mkdir -p GroundingDINO/weights
      wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \\
          -O GroundingDINO/weights/groundingdino_swint_ogc.pth
      pip install -e GroundingDINO
    """
    import os
    import sys as _sys

    _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _gd_path = os.path.join(_base, "GroundingDINO")
    if _gd_path not in _sys.path:
        _sys.path.insert(0, _gd_path)

    _patch_transformers_compat()

    _cfg = os.path.join(_gd_path, "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
    _ckpt = os.path.join(_gd_path, "weights", "groundingdino_swint_ogc.pth")

    if not os.path.exists(_cfg) or not os.path.exists(_ckpt):
        raise FileNotFoundError(
            f"GroundingDINO config or checkpoint not found.\n"
            f"  Config: {_cfg}\n  Checkpoint: {_ckpt}\n"
            f"Download: wget https://github.com/IDEA-Research/GroundingDINO/releases/"
            f"download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O {_ckpt}"
        )

    from groundingdino.util.inference import load_model as _gd_load
    model = _gd_load(_cfg, _ckpt, device=device)
    model.eval()
    return model, None, device


def _patch_transformers_compat():
    """Monkey-patch transformers >= 4.31: thêm get_head_mask cho BertPreTrainedModel."""
    try:
        import transformers.models.bert.modeling_bert as _bm
        from transformers.modeling_utils import PreTrainedModel
        if not hasattr(_bm.BertPreTrainedModel, "get_head_mask"):
            _bm.BertPreTrainedModel.get_head_mask = PreTrainedModel.get_head_mask
        if not hasattr(_bm.BertModel, "get_head_mask"):
            _bm.BertModel.get_head_mask = PreTrainedModel.get_head_mask
    except Exception:
        pass
    try:
        import groundingdino.models.GroundingDINO.bertwarper as _bw
        from transformers.modeling_utils import PreTrainedModel
        if not hasattr(_bw.BertModelWarper, "get_head_mask"):
            _bw.BertModelWarper.get_head_mask = PreTrainedModel.get_head_mask
    except Exception:
        pass


def detect_objects(
    image: Image.Image,
    entities: list[str],
    grounding_model=None,
    processor=None,
    device: str = "cuda",
    threshold: float = 0.3,
) -> dict:
    """
    Detect entities trên ảnh bằng Grounding DINO.
    Dùng groundingdino.util.inference.predict() — đúng cách từ notebook reference.
    """
    if grounding_model is None:
        grounding_model, processor, device = load_grounding_model(device)

    from groundingdino.util.inference import load_image as _load_img, predict as _predict

    import numpy as np

    results = {}
    for entity in entities:
        if not entity:
            continue
        text_prompt = entity.strip().rstrip(".") + "."
        try:
            with torch.no_grad():
                # Lưu ảnh tạm để predict (load_image cần path)
                import tempfile, os as _os
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, mode="wb") as f:
                    image.save(f.name)
                    tmp_path = f.name
                try:
                    _, image_tensor = _load_img(tmp_path)
                    boxes, logits, phrases = _predict(
                        model=grounding_model,
                        image=image_tensor,
                        caption=text_prompt,
                        box_threshold=threshold,
                        text_threshold=0.25,
                        device=device,
                    )
                finally:
                    _os.unlink(tmp_path)

                if boxes is None or len(boxes) == 0:
                    results[entity] = {"bbox": None, "score": 0.0}
                    continue

                best_idx = int(torch.argmax(logits))
                score = float(logits[best_idx])
                cx, cy, bw, bh = boxes[best_idx].tolist()
                x1 = max(0.0, cx - bw / 2)
                y1 = max(0.0, cy - bh / 2)
                x2 = min(1.0, cx + bw / 2)
                y2 = min(1.0, cy + bh / 2)
                results[entity] = {"bbox": [x1, y1, x2, y2], "score": score}
        except Exception:
            results[entity] = {"bbox": None, "score": 0.0}
    return results


def draw_marks(
    image: Image.Image,
    detections: dict,
    mark_color=(255, 60, 60),
    text_color=(255, 255, 255),
) -> Image.Image:
    img = image.copy()
    w, h = img.size
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", max(14, min(w, h) // 25))
    except Exception:
        font = ImageFont.load_default()

    for idx, entity in enumerate(detections.keys()):
        if entity is None:
            continue
        bbox, conf = detections.get(entity, ([0, 0, 0, 0], 0))
        x1, y1, x2, y2 = bbox
        px1, py1, px2, py2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        if px1 == px2 == py1 == py2 == 0:
            continue
        draw.rectangle([px1, py1, px2, py2], outline=mark_color, width=max(2, min(w, h) // 150))
        label = f"[{idx + 1}] {entity}"
        lw, lh = draw.textbbox((0, 0), label, font=font)[2:]
        draw.rectangle([px1, max(0, py1 - lh - 4), px1 + lw + 4, py1], fill=mark_color)
        draw.text((px1 + 2, py1 - lh - 2), label, fill=text_color, font=font)
    return img


def run_ogm(
    image: Image.Image,
    question: str,
    grounding_model=None,
    processor=None,
    device: str = "cuda",
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    coco_vocabulary: list = None,
    extraction_result=None,
    O2_is_viewer: bool = False,
) -> dict:
    """
    Detect and mark O1/O2 on image using Grounding DINO.

    Accepts either legacy tuple-based extraction (O1_raw, O2_raw) or
    ExtractionResult from the LLM extractor.

    All-or-nothing: both O1 and O2 must pass detection confidence.
    O2_is_viewer=True skips O2 detection (viewer has no physical bbox).
    """
    vocab = coco_vocabulary or COCO_CATEGORIES

    if extraction_result is not None:
        O1_raw = extraction_result.O1
        O2_raw = extraction_result.O2 if not O2_is_viewer else None
        O2_is_viewer = extraction_result.O2_is_viewer
    else:
        O1_raw, O2_raw = extract_entities_from_question(question)

    if O1_raw is None or (O2_raw is None and not O2_is_viewer):
        return _fail_result(image, O1_raw, O2_raw)

    O1_name = best_match(O1_raw, vocab)
    O2_name = best_match(O2_raw, vocab) if O2_raw else None

    # Detect O1 always
    detections = {}
    if grounding_model is not None:
        o1_results = detect_objects(
            image, [O1_name],
            grounding_model=grounding_model,
            processor=processor,
            device=device,
            threshold=confidence_threshold,
        )
        detections.update(o1_results)

        # Detect O2 only if O2_is_viewer=False (viewer has no physical bbox)
        if not O2_is_viewer and O2_name:
            o2_results = detect_objects(
                image, [O2_name],
                grounding_model=grounding_model,
                processor=processor,
                device=device,
                threshold=confidence_threshold,
            )
            detections.update(o2_results)

    O1_det = detections.get(O1_name, {"bbox": [0,0,0,0], "score": 0.0})
    O1_bbox = O1_det.get("bbox", [0,0,0,0])
    O1_conf = O1_det.get("score", 0.0)
    O2_bbox, O2_conf = [0,0,0,0], 0.0
    if not O2_is_viewer:
        O2_det = detections.get(O2_name, {"bbox": [0,0,0,0], "score": 0.0})
        O2_bbox = O2_det.get("bbox", [0,0,0,0])
        O2_conf = O2_det.get("score", 0.0)

    # All-or-nothing: both must pass threshold (O2 skipped if viewer)
    both_pass = O1_conf >= confidence_threshold
    if not O2_is_viewer:
        both_pass = both_pass and (O2_conf >= confidence_threshold)

    if not both_pass:
        return _fail_result(image, O1_name, O2_name)

    # Draw only the boxes that passed threshold
    det_for_draw = {}
    if O1_conf >= confidence_threshold:
        det_for_draw[O1_name] = (O1_bbox, O1_conf)
    if not O2_is_viewer and O2_conf >= confidence_threshold and O2_name:
        det_for_draw[O2_name] = (O2_bbox, O2_conf)

    marked_image = draw_marks(image, det_for_draw) if det_for_draw else image

    return {
        "marked_image": marked_image,
        "O1_name": O1_name,
        "O2_name": O2_name if not O2_is_viewer else None,
        "O1_bbox": O1_bbox if O1_conf >= confidence_threshold else None,
        "O2_bbox": O2_bbox if (not O2_is_viewer and O2_conf >= confidence_threshold) else None,
        "O1_conf": O1_conf,
        "O2_conf": O2_conf,
        "success": both_pass,
        "detections": detections,
    }


def _fail_result(image, O1_name, O2_name) -> dict:
    return {
        "marked_image": image, "O1_name": O1_name, "O2_name": O2_name,
        "O1_bbox": None, "O2_bbox": None,
        "O1_conf": 0.0, "O2_conf": 0.0,
        "success": False, "detections": {},
    }
