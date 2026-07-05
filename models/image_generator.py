"""
image_generator.py â€” ASTRA v2 core image processing functions.
Module 1 (bbox marking) + Module 2 (depth heatmap) Ä‘Æ°á»£c tÃ¡ch riÃªng,
má»—i hÃ m stateless, nháº­n record dict tá»« test_objects_last.json.

Quy Æ°á»›c xuyÃªn suá»‘t:
  - depth value cÃ ng THáº¤P = cÃ ng Gáº¦N camera
  - Táº¥t cáº£ bbox coordinates dáº¡ng [x1, y1, x2, y2], normalized [0,1] trÃªn width/height
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from config.pipeline_config import (
    DET_CONF_THRESHOLD,
    YOLOE_IMGSZ,
    MARK_COLOR_O1,
    MARK_COLOR_O2,
    BBOX_BORDER_WIDTH,
    BBOX_FONT_SIZE_RATIO,
    BBOX_MIN_FONT_SIZE,
    BBOX_MAX_FONT_SIZE,
    COLORBAR_WIDTH_RATIO,
    COLORBAR_HEIGHT_PX,
    COLORBAR_MARGIN_PX,
    DEPTH_COLORMAP,
)
from .module1_ogm import detect_yoloe_objects, load_yoloe_model


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Module 1 â€” Bbox Marking
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def should_run_yoloe(record: dict) -> bool:
    """Confidence gating: only run M1+M2 when extraction is reliable."""
    if record.get("confidence", 0.0) < 0.6:
        return False
    if record.get("O1_hallucinated", False) or record.get("O2_hallucinated", False):
        return False
    return True


def _detect_single(
    image: Image.Image,
    text: str,
    yoloe_model,
    device: str,
    threshold: float,
    imgsz: int = YOLOE_IMGSZ,
) -> tuple[Optional[list], float]:
    """
    Detect one entity with YOLOE-26X.
    Returns (bbox_norm, score). bbox_norm = [x1, y1, x2, y2] normalized,
    or None when detection fails.
    """
    if yoloe_model is None or not text.strip():
        return None, 0.0

    detections = detect_yoloe_objects(
        image=image,
        entities=[text],
        yoloe_model=yoloe_model,
        device=device,
        threshold=threshold,
        imgsz=imgsz,
    )
    result = detections.get(text, {})
    return result.get("bbox"), float(result.get("score", 0.0))


def _draw_box_outside_label(
    draw: ImageDraw.ImageDraw,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    color: tuple[int, int, int],
    img_w: int, img_h: int,
    font_size: int,
) -> None:
    """
    Váº½ bbox viá»n + label Ä‘áº·t PHÃA NGOÃ€I gÃ³c trÃªn-trÃ¡i box.
    KhÃ´ng fill bÃªn trong box. Box nhá» (< 20px) â†’ label Ä‘áº·t cÃ¡ch box 10px bÃªn ngoÃ i.
    """
    border_color = color
    text_color = (255, 255, 255)

    # bounding box
    draw.rectangle([x1, y1, x2, y2], outline=border_color, width=BBOX_BORDER_WIDTH)

    # font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    # label size
    bbox_lbl = draw.textbbox((0, 0), label, font=font)
    lw = bbox_lbl[2] - bbox_lbl[0]
    lh = bbox_lbl[3] - bbox_lbl[1]

    box_w = x2 - x1
    box_h = y2 - y1
    label_h = lh + 4

    # quyáº¿t vá»‹ trÃ­ label: Æ°u tiÃªn trÃªn-trÃ¡i box, ra ngoÃ i náº¿u box quÃ¡ nhá»
    if box_w >= lw + 8 and box_h >= label_h:
        # label bÃªn trong gÃ³c trÃªn-trÃ¡i box
        lx = x1 + 3
        ly = y1 - label_h - 2
        if ly < 0:
            ly = y1 + 3
    else:
        # label bÃªn ngoÃ i gÃ³c trÃªn-trÃ¡i, cÃ¡ch box 10px
        lx = max(0, x1 - lw - 4)
        ly = max(0, y1 - label_h - 10)

    # ná»n label
    draw.rectangle([lx, ly, lx + lw + 4, ly + label_h], fill=border_color)
    # text
    draw.text((lx + 2, ly + 2), label, fill=text_color, font=font)


def generate_bbox_image(
    image: Image.Image,
    record: dict,
    yoloe_model=None,
    device: str = "cuda",
    det_threshold: float = DET_CONF_THRESHOLD,
    imgsz: int = YOLOE_IMGSZ,
) -> tuple[Image.Image, dict]:
    """
    Module 1 â€” Detect O1/O2 bang YOLOE-26X, váº½ [1]/[2] lÃªn áº£nh.

    Args:
        image: PIL.Image gá»‘c (RGB)
        record: dict tá»« test_objects_last.json, cÃ³ keys:
            Object.O1, Object.O2, O2_is_viewer, confidence
        yoloe_model, device: YOLOE-26X model
        det_threshold: ngÆ°á»¡ng confidence cho detect

    Returns:
        (marked_image, box_info)
        marked_image: PIL.Image Ä‘Ã£ váº½ bbox (hoáº·c áº£nh gá»‘c náº¿u fail)
        box_info: dict {
            "marks_ok": bool,
            "box_o1": [x1,y1,x2,y2] | None,
            "box_o2": [x1,y1,x2,y2] | None,
            "o1_name": str,
            "o2_name": str | None,
            "o2_is_viewer": bool,
            "conf_o1": float,
            "conf_o2": float,
        }
    """
    o1_text = record.get("Object", {}).get("O1", "")
    o2_text = record.get("Object", {}).get("O2", "")
    is_viewer = bool(record.get("O2_is_viewer", False))

    w, h = image.size

    # Detect O1
    box_o1, conf_o1 = _detect_single(image, o1_text, yoloe_model, device, det_threshold, imgsz)

    # Detect O2 (chá»‰ náº¿u khÃ´ng pháº£i viewer)
    box_o2, conf_o2 = None, 0.0
    if not is_viewer:
        box_o2, conf_o2 = _detect_single(image, o2_text, yoloe_model, device, det_threshold, imgsz)

    # Determine marks_ok
    o1_ok = box_o1 is not None and conf_o1 >= det_threshold
    o2_ok = is_viewer or (box_o2 is not None and conf_o2 >= det_threshold)
    marks_ok = o1_ok and o2_ok

    if not marks_ok:
        # Fallback: tráº£ áº£nh gá»‘c khÃ´ng váº½ gÃ¬
        box_info = {
            "marks_ok": False,
            "box_o1": None,
            "box_o2": None,
            "o1_name": o1_text,
            "o2_name": o2_text if not is_viewer else None,
            "o2_is_viewer": is_viewer,
            "conf_o1": conf_o1,
            "conf_o2": conf_o2,
        }
        return image.copy(), box_info

    # Váº½ bbox lÃªn áº£nh
    marked = image.copy()
    draw = ImageDraw.Draw(marked)

    font_size = int(min(w, h) * BBOX_FONT_SIZE_RATIO)
    font_size = max(BBOX_MIN_FONT_SIZE, min(BBOX_MAX_FONT_SIZE, font_size))

    # O1 box â€” mÃ u Ä‘á», label [1]
    px1, py1, px2, py2 = int(box_o1[0]*w), int(box_o1[1]*h), int(box_o1[2]*w), int(box_o1[3]*h)
    _draw_box_outside_label(draw, px1, py1, px2, py2, "[1]", MARK_COLOR_O1, w, h, font_size)

    # O2 box â€” mÃ u xanh dÆ°Æ¡ng, label [2]
    if not is_viewer and box_o2 is not None:
        px1_o2, py1_o2, px2_o2, py2_o2 = int(box_o2[0]*w), int(box_o2[1]*h), int(box_o2[2]*w), int(box_o2[3]*h)
        _draw_box_outside_label(draw, px1_o2, py1_o2, px2_o2, py2_o2, "[2]", MARK_COLOR_O2, w, h, font_size)

    box_info = {
        "marks_ok": True,
        "box_o1": box_o1,
        "box_o2": box_o2,
        "o1_name": o1_text,
        "o2_name": o2_text if not is_viewer else None,
        "o2_is_viewer": is_viewer,
        "conf_o1": conf_o1,
        "conf_o2": conf_o2,
    }
    return marked, box_info


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Module 2 â€” Depth Heatmap
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_depth_model(model_size: str = "small", device: str = "cuda"):
    """Load Depth-Anything-V2 model."""
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError:
        raise ImportError(
            "Depth-Anything-V2 not found. Install: pip install depth-anything-v2"
        )
    configs = {
        "small":  {"encoder": "vitl", "features": 256,  "out_channels": [256, 512, 1024, 1024]},
        "base":   {"encoder": "vitl", "features": 512,  "out_channels": [256, 512, 1024, 1024]},
        "large":  {"encoder": "vitl", "features": 512,  "out_channels": [256, 512, 1024, 1024]},
    }
    model = DepthAnythingV2(**configs.get(model_size, configs["small"]))
    model = model.to(device).eval()
    return model


def compute_depth_map(
    image: Image.Image,
    depth_model,
    device: str = "cuda",
    target_size: int = 518,
) -> np.ndarray:
    """
    TÃ­nh depth map tá»« Depth-Anything-V2.
    Returns depth_map: 2D numpy array (H, W), giÃ¡ trá»‹ [0,1], nhá» = gáº§n camera.
    """
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        depth = depth_model(img_tensor).squeeze().cpu().numpy()

    # Resize vá» kÃ­ch thÆ°á»›c gá»‘c
    orig_w, orig_h = image.size
    depth_img = Image.fromarray((depth * 255 / depth.max()).astype(np.uint8))
    depth_img = depth_img.resize((orig_w, orig_h), Image.BILINEAR)
    depth_map = np.array(depth_img).astype(np.float32) / 255.0

    # Normalize vá» [0,1], INVERT: nhá» = gáº§n camera
    if depth_map.max() > depth_map.min():
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    return 1.0 - depth_map


def _mean_depth_in_bbox(
    depth_map: np.ndarray,
    bbox: Optional[list],
    img_h: int, img_w: int,
) -> float:
    """TÃ­nh depth trung bÃ¬nh trong má»™t bbox (normalized coordinates)."""
    if bbox is None:
        return float(np.median(depth_map))
    x1, y1, x2, y2 = bbox
    px1 = max(0, int(x1 * img_w))
    py1 = max(0, int(y1 * img_h))
    px2 = min(img_w, int(x2 * img_w))
    py2 = min(img_h, int(y2 * img_h))
    if px2 <= px1 or py2 <= py1:
        return float(np.median(depth_map))
    region = depth_map[py1:py2, px1:px2]
    return float(np.mean(region)) if region.size > 0 else float(np.median(depth_map))


def compute_depth_cue(
    image: Image.Image,
    box_info: dict,
    depth_model=None,
    device: str = "cuda",
    model_size: str = "small",
) -> tuple[np.ndarray, float, float]:
    """
    Module 2 (pháº§n 1) â€” TÃ­nh depth map vÃ  depth trung bÃ¬nh táº¡i bbox O1/O2.

    Args:
        image: PIL.Image gá»‘c
        box_info: dict tá»« generate_bbox_image(), pháº£i cÃ³ marks_ok=True
        depth_model: Depth-Anything-V2 model
        device, model_size

    Returns:
        (depth_map, depth_o1, depth_o2)
        depth_map: 2D numpy (H,W) [0,1], nhá» = gáº§n camera
        depth_o1: float
        depth_o2: float (0.0 náº¿u viewer)
    """
    if depth_model is None:
        depth_model = load_depth_model(model_size, device)

    depth_map = compute_depth_map(image, depth_model, device)
    h, w = depth_map.shape[:2]
    img_h, img_w = image.size[1], image.size[0]

    depth_o1 = _mean_depth_in_bbox(depth_map, box_info.get("box_o1"), h, w)
    is_viewer = box_info.get("o2_is_viewer", False)
    if is_viewer:
        depth_o2 = 0.0
    else:
        depth_o2 = _mean_depth_in_bbox(depth_map, box_info.get("box_o2"), h, w)

    return depth_map, depth_o1, depth_o2


def depth_relation_text(
    depth_o1: float,
    depth_o2: float,
    o1_name: str,
    o2_name: str,
    threshold: float = 0.05,
) -> str:
    """
    Sinh depth hint text dá»±a trÃªn chÃªnh lá»‡ch depth.
    Quy Æ°á»›c: depth tháº¥p = gáº§n camera.
    """
    diff = depth_o2 - depth_o1
    if abs(diff) < threshold:
        return (
            f"{o1_name} and {o2_name} appear to be at a similar depth "
            f"(estimate may be unreliable for flat or distant scenes)."
        )
    elif diff > 0:
        return (
            f"{o1_name} appears closer to the camera than {o2_name}."
        )
    else:
        return (
            f"{o2_name} appears closer to the camera than {o1_name}."
        )


def _colormap_viridis(depth: np.ndarray) -> np.ndarray:
    """Colorize depth map vá»›i viridis colormap (RGB)."""
    dm = (depth * 255).astype(np.uint8)
    colored = cv2.applyColorMap(dm, cv2.COLORMAP_VIRIDIS)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return colored


def _colormap_turbo(depth: np.ndarray) -> np.ndarray:
    """Colorize depth map vá»›i turbo colormap."""
    dm = (depth * 255).astype(np.uint8)
    colored = cv2.applyColorMap(dm, cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return colored


def _get_colormap(name: str):
    if name == "viridis":
        return _colormap_viridis
    if name == "turbo":
        return _colormap_turbo
    return _colormap_viridis


def _draw_colorbar_legend(
    img: np.ndarray,
    colormap_fn,
    bar_w: int,
    bar_h: int,
    margin: int,
    font_size: int = 13,
) -> np.ndarray:
    """
    Váº½ colorbar legend gÃ³c dÆ°á»›i-pháº£i: dáº£i mÃ u viridis + chá»¯ "Near"/"Far".
    img: numpy array (H,W,3), tráº£ vá» numpy array Ä‘Ã£ váº½ thÃªm colorbar.
    """
    h, w = img.shape[:2]

    # canvas cho legend náº±m riÃªng
    bar_x = w - bar_w - margin
    bar_y = h - bar_h - margin - 30  # chá»«a chá»— cho chá»¯

    # táº¡o gradient viridis cho colorbar
    gradient = np.linspace(0, 1, bar_h).astype(np.float32)
    gradient_2d = np.tile(gradient.reshape(-1, 1), (1, bar_w))
    legend_bar = (colormap_fn(gradient_2d) * 255).astype(np.uint8)

    # paste vÃ o áº£nh
    img[bar_y:bar_y+bar_h, bar_x:bar_x+bar_w] = legend_bar

    # váº½ viá»n
    cv2.rectangle(img, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (255,255,255), 1)

    # text labels
    try:
        font = cv2.FONT_HERSHEY_SIMPLEX
    except Exception:
        font = 0

    # "Near" bÃªn dÆ°á»›i Ä‘áº§u gáº§n (bottom cá»§a bar = giÃ¡ trá»‹ 0 = gáº§n)
    near_text = "Near"
    far_text = "Far"
    cv2.putText(img, near_text, (bar_x, bar_y + bar_h + 18),
                font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, far_text, (bar_x + bar_w - 30, bar_y + bar_h + 18),
                font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def _draw_depth_label(
    img: np.ndarray,
    bbox: list,
    label_text: str,
    color: tuple,
    font_scale: float = 0.55,
    thickness: int = 2,
) -> np.ndarray:
    """
    Váº½ text label táº¡i vá»‹ trÃ­ bbox trÃªn áº£nh depth.
    bbox: [x1,y1,x2,y2] normalized [0,1]
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    px, py = int(x1 * w), int(y1 * h)

    # Ä‘áº·t text cÃ¡ch gÃ³c trÃªn-trÃ¡i bbox má»™t chÃºt
    lx = min(w - 10, max(5, px + 2))
    ly = max(25, py - 5)

    # ná»n Ä‘en má»
    text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(img,
                  (lx - 2, ly - text_size[1] - 2),
                  (lx + text_size[0] + 2, ly + 2),
                  (0, 0, 0), -1)

    cv2.putText(img, label_text, (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    return img


def render_depth_image(
    depth_map: np.ndarray,
    box_info: dict,
    depth_o1: float,
    depth_o2: float,
    is_viewer: bool = False,
    colormap: str = DEPTH_COLORMAP,
) -> Image.Image:
    """
    Module 2 (pháº§n 2) â€” Váº½ depth heatmap + colorbar legend + depth labels táº¡i bbox.

    Args:
        depth_map: 2D numpy (H,W) [0,1], nhá» = gáº§n camera
        box_info: dict tá»« generate_bbox_image()
        depth_o1, depth_o2: giÃ¡ trá»‹ depth Ä‘Ã£ tÃ­nh
        is_viewer: náº¿u True, khÃ´ng váº½ label cho [2] (viewer khÃ´ng cÃ³ tá»a Ä‘á»™)
        colormap: "viridis" hoáº·c "turbo"

    Returns:
        PIL.Image Ä‘Ã£ váº½ depth heatmap + colorbar + labels
    """
    colormap_fn = _get_colormap(colormap)

    # Colorize depth map
    colored = (colormap_fn(depth_map) * 255).astype(np.uint8)

    h, w = colored.shape[:2]

    # Colorbar legend gÃ³c dÆ°á»›i-pháº£i
    bar_w = int(w * COLORBAR_WIDTH_RATIO)
    bar_h = COLORBAR_HEIGHT_PX
    margin = COLORBAR_MARGIN_PX
    colored = _draw_colorbar_legend(colored, colormap_fn, bar_w, bar_h, margin)

    # Depth labels táº¡i vá»‹ trÃ­ bbox
    box_o1 = box_info.get("box_o1")
    box_o2 = box_info.get("box_o2")

    label_color = (255, 255, 255)

    if box_o1 is not None:
        label1 = f"[1] depthâ‰ˆ{depth_o1:.2f}"
        colored = _draw_depth_label(colored, box_o1, label1, label_color)

    # [2] chá»‰ váº½ náº¿u khÃ´ng pháº£i viewer (viewer cÃ³ depth=0.0 nhÆ°ng khÃ´ng cÃ³ tá»a Ä‘á»™ pixel)
    if not is_viewer and box_o2 is not None:
        label2 = f"[2] depthâ‰ˆ{depth_o2:.2f}"
        colored = _draw_depth_label(colored, box_o2, label2, label_color)

    return Image.fromarray(colored)

