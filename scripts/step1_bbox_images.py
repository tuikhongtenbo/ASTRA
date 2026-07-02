"""
step1_bbox_images.py — ASTRA v2, Module 1: Sinh bbox-marked images.
Chạy standalone: JSON + ảnh gốc → output/m1_bbox/

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

from PIL import Image

from config.pipeline_config import (
    EXTRACTION_FILE, IMAGE_DIR, M1_OUTPUT_DIR, M1_BBOX_INFO_FILE,
    DET_CONF_THRESHOLD,
)
from models.image_generator import (
    generate_bbox_image, should_run_grounding, load_grounding_model,
)


def find_image_path(image_name: str) -> str | None:
    candidates = [
        os.path.join(IMAGE_DIR, image_name),
        os.path.join(IMAGE_DIR, image_name.replace(".jpg", ".png")),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


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

    # Load extraction
    with open(args.extraction, "r", encoding="utf-8") as f:
        records = json.load(f)

    if args.max_samples > 0:
        records = records[:args.max_samples]

    # Load grounding model
    print("[M1] Loading Grounding DINO...")
    grounding_model, processor, device = None, None, args.device
    try:
        grounding_model, processor, device = load_grounding_model(args.device)
        print(f"[M1] Grounding DINO loaded on {device}")
    except Exception as e:
        print(f"[M1] WARNING: Could not load Grounding DINO: {e}")
        print("[M1] Will still run but detect will fail for all samples.")

    bbox_info = {}
    ok_count = 0
    skip_count = 0
    fail_count = 0

    for i, record in enumerate(records):
        sid = str(record.get("id", i))
        img_name = record.get("image", "")

        # Confidence gating
        if not should_run_grounding(record):
            skip_count += 1
            # Vẫn lưu ảnh gốc vào output để giữ đồng bộ id
            img_path = find_image_path(img_name)
            if img_path:
                try:
                    img = Image.open(img_path).convert("RGB")
                    img.save(os.path.join(args.output_dir, f"{sid}_bbox.jpg"), quality=95)
                except Exception:
                    pass
            bbox_info[sid] = {
                "marks_ok": False,
                "box_o1": None,
                "box_o2": None,
                "o1_name": record.get("Object", {}).get("O1", ""),
                "o2_name": record.get("Object", {}).get("O2", ""),
                "o2_is_viewer": bool(record.get("O2_is_viewer", False)),
                "conf_o1": 0.0,
                "conf_o2": 0.0,
                "skip_reason": "confidence_gating",
            }
            print(f"[{i+1}/{len(records)}] id={sid} SKIP (confidence gating)")
            continue

        # Load image
        img_path = find_image_path(img_name)
        if not img_path:
            print(f"[{i+1}/{len(records)}] id={sid} ERROR: image not found: {img_name}")
            bbox_info[sid] = {"marks_ok": False, "skip_reason": "image_not_found"}
            continue

        img = Image.open(img_path).convert("RGB")

        # Run M1
        marked_img, box_info = generate_bbox_image(
            image=img,
            record=record,
            grounding_model=grounding_model,
            processor=processor,
            device=device,
            det_threshold=DET_CONF_THRESHOLD,
        )

        # Save image
        out_path = os.path.join(args.output_dir, f"{sid}_bbox.jpg")
        marked_img.save(out_path, quality=95)

        bbox_info[sid] = box_info

        if box_info["marks_ok"]:
            ok_count += 1
            print(f"[{i+1}/{len(records)}] id={sid} OK  (o1={box_info['conf_o1']:.2f}, "
                  f"o2={box_info['conf_o2']:.2f})")
        else:
            fail_count += 1
            print(f"[{i+1}/{len(records)}] id={sid} FAIL")

    # Save bbox_info.json
    with open(M1_BBOX_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(bbox_info, f, ensure_ascii=False, indent=2)

    print(f"\n[M1] Done: {ok_count} OK, {fail_count} FAIL, {skip_count} SKIP "
          f"(confidence gating) | Total: {len(records)}")
    print(f"[M1] Images → {args.output_dir}/")
    print(f"[M1] Info   → {M1_BBOX_INFO_FILE}")


if __name__ == "__main__":
    main()
