"""
step1_bbox_images.py — ASTRA v2, Module 1: Sinh bbox-marked images.
Chạy standalone: test_objects_last.json + ảnh gốc -> output/m1_bbox/

Usage:
  python scripts/step1_bbox_images.py
  python scripts/step1_bbox_images.py --max-samples 50
  python scripts/step1_bbox_images.py --device cpu
"""

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from PIL import Image, ImageDraw, ImageFont

from config.pipeline_config import (
    EXTRACTION_FILE, IMAGE_DIR, M1_OUTPUT_DIR,
    DET_CONF_THRESHOLD,
)
from models.image_generator import (
    generate_bbox_image, load_yoloe_model,
)
from utils.utils import find_image_path, load_image


def _draw_fallback_boxes(image: Image.Image, record: dict) -> tuple[Image.Image, dict]:
    o1_text = record.get("Object", {}).get("O1", "")
    o2_text = record.get("Object", {}).get("O2", "")
    is_viewer = bool(record.get("O2_is_viewer", False))

    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    w, h = image.size
    o1_box = [0.08, 0.18, 0.45, 0.82]
    o2_box = None if is_viewer else [0.55, 0.20, 0.92, 0.82]

    def draw_box(box, label, color):
        x1, y1, x2, y2 = int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        lx = max(0, x1)
        ly = max(0, y1 - th - 6)
        draw.rectangle([lx, ly, lx + tw + 6, ly + th + 4], fill=color)
        draw.text((lx + 3, ly + 2), label, fill=(255, 255, 255), font=font)

    if o1_text:
        draw_box(o1_box, "[1]", (255, 60, 60))
    if o2_box is not None and o2_text:
        draw_box(o2_box, "[2]", (30, 90, 255))

    return marked, {
        "marks_ok": bool(o1_text) and (is_viewer or bool(o2_text)),
        "box_o1": o1_box if o1_text else None,
        "box_o2": None if is_viewer else o2_box,
        "o1_name": o1_text,
        "o2_name": None if is_viewer else o2_text,
        "o2_is_viewer": is_viewer,
        "conf_o1": 0.0,
        "conf_o2": 0.0,
        "fallback_used": True,
        "skip_reason": "yoloe_miss_fallback",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction", default=EXTRACTION_FILE)
    parser.add_argument("--image-dir", default=IMAGE_DIR)
    parser.add_argument("--output-dir", default=M1_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="0 = chạy tất cả")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.extraction, "r", encoding="utf-8") as f:
        records = json.load(f)

    if args.max_samples > 0:
        records = records[:args.max_samples]

    print("[M1] Loading YOLOE-26X...")
    yoloe_model, device = None, args.device
    try:
        yoloe_model = load_yoloe_model(args.device)
        print(f"[M1] YOLOE-26X loaded on {device}")
    except Exception as e:
        print(f"[M1] WARNING: Could not load YOLOE-26X: {e}")
        print("[M1] Will still run with fallback boxes when YOLOE misses.")

    bbox_info = {}
    ok_count = 0
    fail_count = 0

    for i, record in enumerate(records):
        sid = str(record.get("id", i))
        img_name = record.get("image", "")

        img_path = find_image_path(args.image_dir, img_name)
        if not img_path:
            print(f"[{i+1}/{len(records)}] id={sid} ERROR: image not found: {img_name}")
            bbox_info[sid] = {"marks_ok": False, "skip_reason": "image_not_found"}
            continue

        img = load_image(img_path)
        if img is None:
            print(f"[{i+1}/{len(records)}] id={sid} ERROR: image open failed: {img_name}")
            bbox_info[sid] = {"marks_ok": False, "skip_reason": "image_open_failed"}
            continue

        marked_img, box_info = generate_bbox_image(
            image=img,
            record=record,
            yoloe_model=yoloe_model,
            device=device,
            det_threshold=DET_CONF_THRESHOLD,
        )

        if not box_info.get("marks_ok", False):
            print(f"[{i+1}/{len(records)}] id={sid} YOLOE MISS -> fallback boxes")
            marked_img, box_info = _draw_fallback_boxes(img, record)

        out_path = os.path.join(args.output_dir, f"{sid}_bbox.jpg")
        marked_img.save(out_path, quality=95)
        bbox_info[sid] = box_info

        if box_info.get("marks_ok", False):
            ok_count += 1
            print(f"[{i+1}/{len(records)}] id={sid} OK  (fallback={box_info.get('fallback_used', False)})")
        else:
            fail_count += 1
            print(f"[{i+1}/{len(records)}] id={sid} FAIL")

    bbox_info_path = os.path.join(args.output_dir, "bbox_info.json")
    with open(bbox_info_path, "w", encoding="utf-8") as f:
        json.dump(bbox_info, f, ensure_ascii=False, indent=2)

    print(f"\n[M1] Done: {ok_count} OK, {fail_count} FAIL | Total: {len(records)}")
    print(f"[M1] Images -> {args.output_dir}/")
    print(f"[M1] Info   -> {bbox_info_path}")


if __name__ == "__main__":
    main()
