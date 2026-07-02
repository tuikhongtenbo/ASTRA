"""
Module 2 — Depth-Layer Cue (DLC)
Chạy Depth-Anything-V2 để tạo depth map, sinh depth cue.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image

from config.config import DEPTH_EPSILON, DEPTH_MODEL_SIZE


def load_depth_model(model_size: str = "small", device: str = "cuda"):
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError:
        raise ImportError("Depth-Anything-V2 not found. Install: pip install depth-anything-v2")

    configs = {
        "small": {"encoder": "vitl", "features": 256},
        "base": {"encoder": "vitl", "features": 512},
        "large": {"encoder": "vitl", "features": 512},
    }
    model = DepthAnythingV2(**configs.get(model_size, configs["small"]))
    model = model.to(device).eval()
    return model


def compute_depth_map(image: Image.Image, model, device: str = "cuda", target_size: int = 518) -> np.ndarray:
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        depth = model(img_tensor).squeeze().cpu().numpy()

    orig_w, orig_h = image.size
    depth_img = Image.fromarray((depth * 255 / depth.max()).astype(np.uint8))
    depth_img = depth_img.resize((orig_w, orig_h), Image.BILINEAR)
    depth_map = np.array(depth_img).astype(np.float32) / 255.0

    if depth_map.max() > depth_map.min():
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    return 1.0 - depth_map  # smaller = closer


def get_mean_depth_in_bbox(depth_map: np.ndarray, bbox: Optional[list], image_shape: tuple) -> float:
    if bbox is None:
        return float(np.median(depth_map))
    h, w = image_shape
    x1, y1, x2, y2 = bbox
    px1, py1 = max(0, int(x1 * w)), max(0, int(y1 * h))
    px2, py2 = min(w, int(x2 * w)), min(h, int(y2 * h))
    if px2 <= px1 or py2 <= py1:
        return float(np.median(depth_map))
    region = depth_map[py1:py2, px1:px2]
    return float(np.mean(region)) if region.size > 0 else float(np.median(depth_map))


def generate_depth_cue(O1_name, O2_name, depth_O1, depth_O2, epsilon: float = DEPTH_EPSILON) -> Optional[str]:
    """
    Generate a depth cue as a soft hint (not a hard assertion).
    Phrases depth information as auxiliary/may-be-inaccurate.
    """
    if O1_name is None or O2_name is None:
        return None
    diff = abs(depth_O1 - depth_O2)
    if diff > epsilon:
        if depth_O1 < depth_O2:
            closer, farther = O1_name, O2_name
        else:
            closer, farther = O2_name, O1_name
        return (
            f"Depth hint (auxiliary, may be inaccurate): object [1] ({closer}) "
            f"appears closer to the camera than object [2] ({farther}) "
            f"based on an external depth model. Please cross-check this against "
            f"what you directly observe in the image before concluding — "
            f"do not rely on this hint alone."
        )
    return (
        f"Depth hint (auxiliary, may be inaccurate): objects [1] ({O1_name}) "
        f"and [2] ({O2_name}) appear to be at a similar depth. "
        f"This estimate may be unreliable for flat or distant scenes — verify visually."
    )


def run_dlc(
    image: Image.Image,
    O1_bbox, O2_bbox, O1_name, O2_name,
    depth_model=None,
    device: str = "cuda",
    epsilon: float = DEPTH_EPSILON,
    O2_is_viewer: bool = False,
) -> dict:
    """
    Run depth estimation and generate depth cue.

    Args:
        O2_is_viewer: if True, O2 is the viewer (no physical bbox) -> depth_O2 = 0.0 (camera plane).
    """
    if O1_name is None and O2_name is None:
        return {"depth_cue": None, "depth_map": None, "depth_O1": 0.0, "depth_O2": 0.0, "success": False}

    if depth_model is None:
        depth_model = load_depth_model(DEPTH_MODEL_SIZE, device)

    try:
        depth_map = compute_depth_map(image, depth_model, device)
    except Exception:
        return {"depth_cue": None, "depth_map": None, "depth_O1": 0.0, "depth_O2": 0.0, "success": False}

    h, w = image.size[1], image.size[0]
    depth_O1 = get_mean_depth_in_bbox(depth_map, O1_bbox, (h, w))

    # Viewer is at camera plane -> depth = 0.0 (closest possible)
    if O2_is_viewer:
        depth_O2 = 0.0
    else:
        depth_O2 = get_mean_depth_in_bbox(depth_map, O2_bbox, (h, w))

    effective_O2_name = "the viewer" if O2_is_viewer else O2_name
    depth_cue = generate_depth_cue(O1_name, effective_O2_name, depth_O1, depth_O2, epsilon)

    return {"depth_cue": depth_cue, "depth_map": depth_map,
            "depth_O1": depth_O1, "depth_O2": depth_O2, "success": True}


def depth_map_to_colormap(depth_map: np.ndarray) -> Image.Image:
    import cv2
    dm = (depth_map * 255).astype(np.uint8)
    colored = cv2.applyColorMap(dm, cv2.COLORMAP_VIRIDIS)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return Image.fromarray(colored)
