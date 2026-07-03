"""
Module 1 — Object-Grounded Marking (OGM)
Detect bbox cho O1, O2 bang YOLOE-26X, ve [1], [2] len anh.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFont

from config.config import CONFIDENCE_THRESHOLD, DET_CONF_THRESHOLD, YOLOE_IMGSZ, YOLOE_WEIGHTS

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


def _entity_prompt(entity: str) -> str:
    """Normalize a text prompt for YOLOE without collapsing descriptors."""
    return (entity or "").strip().rstrip(".")


def load_yoloe_model(device: str = "cuda", weights: str = YOLOE_WEIGHTS):
    """Load YOLOE-26X directly through the Ultralytics YOLOE API."""
    from ultralytics import YOLOE

    model = YOLOE(weights)
    if device:
        try:
            model.to(device)
        except Exception:
            # Ultralytics also accepts device during predict(); keep loading robust.
            pass
    return model


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value if isinstance(value, list) else list(value)


def _normalize_xyxy(xyxy: list, width: int, height: int) -> list | None:
    if width <= 0 or height <= 0 or len(xyxy) < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in xyxy[:4]]
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1 / width, y1 / height, x2 / width, y2 / height]


def detect_yoloe_objects(
    image: Image.Image,
    entities: list[str],
    yoloe_model=None,
    device: str = "cuda",
    threshold: float = DET_CONF_THRESHOLD,
    imgsz: int = YOLOE_IMGSZ,
) -> dict:
    """
    Detect entities with YOLOE text prompts.

    Returns {entity: {"bbox": [x1, y1, x2, y2] | None, "score": float}},
    where bbox coordinates are normalized to [0, 1].
    """
    valid_entities = [entity for entity in entities if entity and str(entity).strip()]
    results = {entity: {"bbox": None, "score": 0.0} for entity in valid_entities}
    prompts = [_entity_prompt(entity) for entity in valid_entities]
    if yoloe_model is None or not prompts:
        return results

    try:
        yoloe_model.set_classes(prompts)
        predictions = yoloe_model.predict(
            image.convert("RGB"),
            conf=threshold,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
    except Exception:
        return results

    if not predictions:
        return results

    boxes = getattr(predictions[0], "boxes", None)
    if boxes is None or len(boxes) == 0:
        return results

    xyxy_values = _as_list(getattr(boxes, "xyxy", None))
    conf_values = _as_list(getattr(boxes, "conf", None))
    cls_values = _as_list(getattr(boxes, "cls", None))
    width, height = image.size

    entity_by_class = dict(enumerate(valid_entities))

    for xyxy, score, cls_id in zip(xyxy_values, conf_values, cls_values):
        cls_idx = int(cls_id)
        entity = entity_by_class.get(cls_idx)
        if entity is None:
            continue
        score = float(score)
        if score < threshold or score <= results[entity]["score"]:
            continue
        bbox = _normalize_xyxy(xyxy, width, height)
        if bbox is None:
            continue
        results[entity] = {"bbox": bbox, "score": score}

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
    yoloe_model=None,
    device: str = "cuda",
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    extraction_result=None,
    O2_is_viewer: bool = False,
    imgsz: int = YOLOE_IMGSZ,
) -> dict:
    """
    Detect and mark O1/O2 on image using YOLOE-26X.

    Accepts either legacy tuple-based extraction (O1_raw, O2_raw) or
    ExtractionResult from the LLM extractor.

    All-or-nothing: both O1 and O2 must pass detection confidence.
    O2_is_viewer=True skips O2 detection (viewer has no physical bbox).
    """
    if extraction_result is not None:
        O1_raw = extraction_result.O1
        O2_raw = extraction_result.O2 if not O2_is_viewer else None
        O2_is_viewer = extraction_result.O2_is_viewer
    else:
        O1_raw, O2_raw = extract_entities_from_question(question)

    if O1_raw is None or (O2_raw is None and not O2_is_viewer):
        return _fail_result(image, O1_raw, O2_raw)

    O1_name = _entity_prompt(O1_raw)
    O2_name = _entity_prompt(O2_raw) if O2_raw else None

    if yoloe_model is None:
        yoloe_model = load_yoloe_model(device)

    entities = [O1_name]
    if not O2_is_viewer and O2_name:
        entities.append(O2_name)

    detections = detect_yoloe_objects(
        image=image,
        entities=entities,
        yoloe_model=yoloe_model,
        device=device,
        threshold=confidence_threshold,
        imgsz=imgsz,
    )

    O1_det = detections.get(O1_name, {"bbox": None, "score": 0.0})
    O1_bbox = O1_det.get("bbox")
    O1_conf = O1_det.get("score", 0.0)
    O2_bbox, O2_conf = None, 0.0
    if not O2_is_viewer and O2_name:
        O2_det = detections.get(O2_name, {"bbox": None, "score": 0.0})
        O2_bbox = O2_det.get("bbox")
        O2_conf = O2_det.get("score", 0.0)

    both_pass = O1_bbox is not None and O1_conf >= confidence_threshold
    if not O2_is_viewer:
        both_pass = both_pass and O2_bbox is not None and O2_conf >= confidence_threshold

    if not both_pass:
        return _fail_result(image, O1_name, O2_name)

    det_for_draw = {}
    if O1_bbox is not None:
        det_for_draw[O1_name] = (O1_bbox, O1_conf)
    if not O2_is_viewer and O2_bbox is not None and O2_name:
        det_for_draw[O2_name] = (O2_bbox, O2_conf)

    marked_image = draw_marks(image, det_for_draw) if det_for_draw else image

    return {
        "marked_image": marked_image,
        "O1_name": O1_name,
        "O2_name": O2_name if not O2_is_viewer else None,
        "O1_bbox": O1_bbox,
        "O2_bbox": O2_bbox if not O2_is_viewer else None,
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
