"""
image_generator.py — ASTRA v2 core image processing functions.
Module 1 (bbox marking) + Module 2 (depth heatmap) được tách riêng,
mỗi hàm stateless, nhận record dict từ test_objects_last.json.

Quy ước xuyên suốt:
  - depth value càng THẤP = càng GẦN camera
  - Tất cả bbox coordinates dạng [x1, y1, x2, y2], normalized [0,1] trên width/height
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from config.pipeline_config import (
    DET_CONF_THRESHOLD,
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


# ─────────────────────────────────────────────────────────────────────────────
# Module 1 — Bbox Marking
# ─────────────────────────────────────────────────────────────────────────────

def should_run_grounding(record: dict) -> bool:
    """
    Confidence gating: chỉ chạy M1+M2 nếu extraction đủ tin cậy.
    """
    if record.get("confidence", 0.0) < 0.6:
        return False
    if record.get("O1_hallucinated", False) or record.get("O2_hallucinated", False):
        return False
    return True


def _patch_groundingdino_transformers_compat():
    """
    Monkey-patch để GroundingDINO tương thích với transformers >= 4.31.
    Lỗi gốc: 'BertModel' object has no attribute 'get_head_mask'
    Nguyên nhân: transformers mới di chuyển get_head_mask sang PreTrainedModel
    nhưng BertPreTrainedModel trong GroundingDINO không kế thừa đúng.
    """
    try:
        import transformers.models.bert.modeling_bert as bert_module
        from transformers.modeling_utils import PreTrainedModel
        # Patch BertPreTrainedModel nếu thiếu get_head_mask
        if not hasattr(bert_module.BertPreTrainedModel, "get_head_mask"):
            bert_module.BertPreTrainedModel.get_head_mask = PreTrainedModel.get_head_mask
        # Patch BertModel trực tiếp (phòng thủ)
        if not hasattr(bert_module.BertModel, "get_head_mask"):
            bert_module.BertModel.get_head_mask = PreTrainedModel.get_head_mask
    except Exception:
        pass

    try:
        # Patch lớp BertModelWarper trong GroundingDINO nếu đã được import
        import groundingdino.models.GroundingDINO.bertwarper as bw
        from transformers.modeling_utils import PreTrainedModel
        for cls in [bw.BertModelWarper]:
            if not hasattr(cls, "get_head_mask"):
                cls.get_head_mask = PreTrainedModel.get_head_mask
    except Exception:
        pass


def load_grounding_model(device: str = "cuda"):
    """
    Load Grounding DINO model và processor.
    Hỗ trợ 2 cách cài:
      1. git clone IDEA-Research/GroundingDINO → package 'groundingdino'
      2. pip install grounding-dino            → package 'grounding_dino'
    """
    import sys
    import os

    # Inject thư mục GroundingDINO clone vào sys.path nếu chưa có
    _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _gd_clone_path = os.path.join(_base, "GroundingDINO")
    if os.path.isdir(_gd_clone_path) and _gd_clone_path not in sys.path:
        sys.path.insert(0, _gd_clone_path)

    # ── Thử import từ repo clone (IDEA-Research/GroundingDINO) ──
    try:
        # Patch trước khi import để tránh lỗi get_head_mask
        _patch_groundingdino_transformers_compat()

        from groundingdino.util.inference import load_model as gd_load_model

        # Patch sau khi import (BertModelWarper có thể chưa tồn tại trước đó)
        _patch_groundingdino_transformers_compat()

        _cfg = os.path.join(
            _gd_clone_path, "groundingdino", "config", "GroundingDINO_SwinT_OGC.py"
        )
        _ckpt = os.path.join(_gd_clone_path, "weights", "groundingdino_swint_ogc.pth")

        if not os.path.exists(_cfg):
            raise FileNotFoundError(f"Config không tìm thấy: {_cfg}")
        if not os.path.exists(_ckpt):
            raise FileNotFoundError(
                f"Checkpoint không tìm thấy: {_ckpt}\n"
                f"Tải về bằng lệnh:\n"
                f"  mkdir -p {os.path.join(_gd_clone_path, 'weights')}\n"
                f"  wget -q https://github.com/IDEA-Research/GroundingDINO/releases/"
                f"download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O {_ckpt}"
            )

        model = gd_load_model(_cfg, _ckpt, device=device)
        model.eval()
        # Trả về (model, None, device) — processor không cần, dùng API inference trực tiếp
        return model, None, device

    except Exception as e1:
        # ── Fallback: pip install grounding-dino ──
        try:
            from grounding_dino.grounding_dino import load_model
            from grounding_dino.grounding_dino_cfg import ModelConfig
            config = ModelConfig()
            model = load_model(config, "cogagent/grounding-dino-tiny")
            model = model.to(device).eval()
            try:
                from transformers import AutoProcessor
                processor = AutoProcessor.from_pretrained("cogagent/grounding-dino-tiny")
            except Exception:
                processor = None
            return model, processor, device
        except ImportError:
            raise ImportError(
                f"Grounding DINO not found. Install: pip install grounding-dino "
                f"or check your PYTHONPATH."
            )


def _detect_single(
    image: Image.Image,
    text: str,
    grounding_model,
    processor,
    device: str,
    threshold: float,
) -> tuple[Optional[list], float]:
    """
    Detect một entity trên ảnh bằng Grounding DINO.
    Returns (bbox_norm, score). bbox_norm = [x1, y1, x2, y2] normalized, hoặc None nếu fail.
    """
    if grounding_model is None or not text.strip():
        return None, 0.0

    text_prompt = text.strip() + "."
    try:
        with torch.no_grad():
            if processor is not None:
                inputs = processor(
                    images=image, texts=[text_prompt], return_tensors="pt"
                ).to(device)
                outputs = grounding_model(**inputs)
            else:
                from torchvision import transforms
                img_resized = image.resize((640, 640)).convert("RGB")
                transform = transforms.Compose([transforms.ToTensor()])
                img_tensor = transform(img_resized).unsqueeze(0).to(device)
                outputs = grounding_model(img_tensor, [text_prompt])

        if hasattr(outputs, "pred_boxes"):
            boxes = outputs.pred_boxes[0].cpu().numpy()
            scores = outputs.scores[0].cpu().numpy() if hasattr(outputs, "scores") else np.array([1.0])
        elif isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
            boxes = outputs[0][0].cpu().numpy()
            scores = outputs[1][0].cpu().numpy() if len(outputs) > 1 else np.array([1.0])
        else:
            return None, 0.0

        # best box
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score < threshold or best_idx >= len(boxes):
            return None, best_score

        cx, cy, bw, bh = boxes[best_idx]
        x1 = max(0.0, cx - bw / 2)
        y1 = max(0.0, cy - bh / 2)
        x2 = min(1.0, cx + bw / 2)
        y2 = min(1.0, cy + bh / 2)
        return [float(x1), float(y1), float(x2), float(y2)], best_score

    except Exception as e:
        return None, 0.0


def _draw_box_outside_label(
    draw: ImageDraw.ImageDraw,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    color: tuple[int, int, int],
    img_w: int, img_h: int,
    font_size: int,
) -> None:
    """
    Vẽ bbox viền + label đặt PHÍA NGOÀI góc trên-trái box.
    Không fill bên trong box. Box nhỏ (< 20px) → label đặt cách box 10px bên ngoài.
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

    # quyết vị trí label: ưu tiên trên-trái box, ra ngoài nếu box quá nhỏ
    if box_w >= lw + 8 and box_h >= label_h:
        # label bên trong góc trên-trái box
        lx = x1 + 3
        ly = y1 - label_h - 2
        if ly < 0:
            ly = y1 + 3
    else:
        # label bên ngoài góc trên-trái, cách box 10px
        lx = max(0, x1 - lw - 4)
        ly = max(0, y1 - label_h - 10)

    # nền label
    draw.rectangle([lx, ly, lx + lw + 4, ly + label_h], fill=border_color)
    # text
    draw.text((lx + 2, ly + 2), label, fill=text_color, font=font)


def generate_bbox_image(
    image: Image.Image,
    record: dict,
    grounding_model=None,
    processor=None,
    device: str = "cuda",
    det_threshold: float = DET_CONF_THRESHOLD,
) -> tuple[Image.Image, dict]:
    """
    Module 1 — Detect O1/O2 bằng Grounding DINO, vẽ [1]/[2] lên ảnh.

    Args:
        image: PIL.Image gốc (RGB)
        record: dict từ test_objects_last.json, có keys:
            Object.O1, Object.O2, O2_is_viewer, confidence
        grounding_model, processor, device: Grounding DINO model
        det_threshold: ngưỡng confidence cho detect

    Returns:
        (marked_image, box_info)
        marked_image: PIL.Image đã vẽ bbox (hoặc ảnh gốc nếu fail)
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
    box_o1, conf_o1 = _detect_single(image, o1_text, grounding_model, processor, device, det_threshold)

    # Detect O2 (chỉ nếu không phải viewer)
    box_o2, conf_o2 = None, 0.0
    if not is_viewer:
        box_o2, conf_o2 = _detect_single(image, o2_text, grounding_model, processor, device, det_threshold)

    # Determine marks_ok
    o1_ok = box_o1 is not None and conf_o1 >= det_threshold
    o2_ok = is_viewer or (box_o2 is not None and conf_o2 >= det_threshold)
    marks_ok = o1_ok and o2_ok

    if not marks_ok:
        # Fallback: trả ảnh gốc không vẽ gì
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

    # Vẽ bbox lên ảnh
    marked = image.copy()
    draw = ImageDraw.Draw(marked)

    font_size = int(min(w, h) * BBOX_FONT_SIZE_RATIO)
    font_size = max(BBOX_MIN_FONT_SIZE, min(BBOX_MAX_FONT_SIZE, font_size))

    # O1 box — màu đỏ, label [1]
    px1, py1, px2, py2 = int(box_o1[0]*w), int(box_o1[1]*h), int(box_o1[2]*w), int(box_o1[3]*h)
    _draw_box_outside_label(draw, px1, py1, px2, py2, "[1]", MARK_COLOR_O1, w, h, font_size)

    # O2 box — màu xanh dương, label [2]
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


# ─────────────────────────────────────────────────────────────────────────────
# Module 2 — Depth Heatmap
# ─────────────────────────────────────────────────────────────────────────────

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
    Tính depth map từ Depth-Anything-V2.
    Returns depth_map: 2D numpy array (H, W), giá trị [0,1], nhỏ = gần camera.
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

    # Resize về kích thước gốc
    orig_w, orig_h = image.size
    depth_img = Image.fromarray((depth * 255 / depth.max()).astype(np.uint8))
    depth_img = depth_img.resize((orig_w, orig_h), Image.BILINEAR)
    depth_map = np.array(depth_img).astype(np.float32) / 255.0

    # Normalize về [0,1], INVERT: nhỏ = gần camera
    if depth_map.max() > depth_map.min():
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    return 1.0 - depth_map


def _mean_depth_in_bbox(
    depth_map: np.ndarray,
    bbox: Optional[list],
    img_h: int, img_w: int,
) -> float:
    """Tính depth trung bình trong một bbox (normalized coordinates)."""
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
    Module 2 (phần 1) — Tính depth map và depth trung bình tại bbox O1/O2.

    Args:
        image: PIL.Image gốc
        box_info: dict từ generate_bbox_image(), phải có marks_ok=True
        depth_model: Depth-Anything-V2 model
        device, model_size

    Returns:
        (depth_map, depth_o1, depth_o2)
        depth_map: 2D numpy (H,W) [0,1], nhỏ = gần camera
        depth_o1: float
        depth_o2: float (0.0 nếu viewer)
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
    Sinh depth hint text dựa trên chênh lệch depth.
    Quy ước: depth thấp = gần camera.
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
    """Colorize depth map với viridis colormap (RGB)."""
    dm = (depth * 255).astype(np.uint8)
    colored = cv2.applyColorMap(dm, cv2.COLORMAP_VIRIDIS)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return colored


def _colormap_turbo(depth: np.ndarray) -> np.ndarray:
    """Colorize depth map với turbo colormap."""
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
    Vẽ colorbar legend góc dưới-phải: dải màu viridis + chữ "Near"/"Far".
    img: numpy array (H,W,3), trả về numpy array đã vẽ thêm colorbar.
    """
    h, w = img.shape[:2]

    # canvas cho legend nằm riêng
    bar_x = w - bar_w - margin
    bar_y = h - bar_h - margin - 30  # chừa chỗ cho chữ

    # tạo gradient viridis cho colorbar
    gradient = np.linspace(0, 1, bar_h).astype(np.float32)
    gradient_2d = np.tile(gradient.reshape(-1, 1), (1, bar_w))
    legend_bar = (colormap_fn(gradient_2d) * 255).astype(np.uint8)

    # paste vào ảnh
    img[bar_y:bar_y+bar_h, bar_x:bar_x+bar_w] = legend_bar

    # vẽ viền
    cv2.rectangle(img, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (255,255,255), 1)

    # text labels
    try:
        font = cv2.FONT_HERSHEY_SIMPLEX
    except Exception:
        font = 0

    # "Near" bên dưới đầu gần (bottom của bar = giá trị 0 = gần)
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
    Vẽ text label tại vị trí bbox trên ảnh depth.
    bbox: [x1,y1,x2,y2] normalized [0,1]
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    px, py = int(x1 * w), int(y1 * h)

    # đặt text cách góc trên-trái bbox một chút
    lx = min(w - 10, max(5, px + 2))
    ly = max(25, py - 5)

    # nền đen mờ
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
    Module 2 (phần 2) — Vẽ depth heatmap + colorbar legend + depth labels tại bbox.

    Args:
        depth_map: 2D numpy (H,W) [0,1], nhỏ = gần camera
        box_info: dict từ generate_bbox_image()
        depth_o1, depth_o2: giá trị depth đã tính
        is_viewer: nếu True, không vẽ label cho [2] (viewer không có tọa độ)
        colormap: "viridis" hoặc "turbo"

    Returns:
        PIL.Image đã vẽ depth heatmap + colorbar + labels
    """
    colormap_fn = _get_colormap(colormap)

    # Colorize depth map
    colored = (colormap_fn(depth_map) * 255).astype(np.uint8)

    h, w = colored.shape[:2]

    # Colorbar legend góc dưới-phải
    bar_w = int(w * COLORBAR_WIDTH_RATIO)
    bar_h = COLORBAR_HEIGHT_PX
    margin = COLORBAR_MARGIN_PX
    colored = _draw_colorbar_legend(colored, colormap_fn, bar_w, bar_h, margin)

    # Depth labels tại vị trí bbox
    box_o1 = box_info.get("box_o1")
    box_o2 = box_info.get("box_o2")

    label_color = (255, 255, 255)

    if box_o1 is not None:
        label1 = f"[1] depth≈{depth_o1:.2f}"
        colored = _draw_depth_label(colored, box_o1, label1, label_color)

    # [2] chỉ vẽ nếu không phải viewer (viewer có depth=0.0 nhưng không có tọa độ pixel)
    if not is_viewer and box_o2 is not None:
        label2 = f"[2] depth≈{depth_o2:.2f}"
        colored = _draw_depth_label(colored, box_o2, label2, label_color)

    return Image.fromarray(colored)
