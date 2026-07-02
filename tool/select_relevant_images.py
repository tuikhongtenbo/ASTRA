"""
Tool: Select relevant images for SpatialMQA samples.
Dựa trên thamkhao/SpatialMQA/Dataset/tool/select_relevant_images.py
"""

from __future__ import annotations

import json
import os
from typing import Optional

from PIL import Image


def select_relevant_images(
    data_file: str,
    output_file: str,
    image_dir: str,
    max_images: int = 10,
) -> list[str]:
    """
    Chọn top-K images có liên quan nhất từ dataset.
    Trả về list image paths.
    """
    if not os.path.exists(data_file):
        print(f"[Tool] Data file not found: {data_file}")
        return []

    selected = []
    with open(data_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_images:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            img_name = obj.get("image", "")
            img_path = os.path.join(image_dir, img_name)
            if os.path.exists(img_path):
                selected.append(img_path)
            else:
                # Try to find in subdirs
                for subdir in ("relevant_images", "images"):
                    candidate = os.path.join(image_dir, subdir, img_name)
                    if os.path.exists(candidate):
                        selected.append(candidate)
                        break

    # Save selected paths
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for p in selected:
            f.write(p + "\n")

    print(f"[Tool] Selected {len(selected)} images -> {output_file}")
    return selected


def verify_images(image_paths: list[str]) -> dict:
    """
    Kiểm tra tất cả images có đọc được không.
    Returns dict: {path: valid, ...}
    """
    results = {}
    for p in image_paths:
        try:
            Image.open(p).convert("RGB")
            results[p] = True
        except Exception:
            results[p] = False
    valid = sum(1 for v in results.values() if v)
    print(f"[Tool] {valid}/{len(results)} images valid")
    return results
