"""
step2_depth_images.py — ASTRA v2, Module 2: Sinh depth heatmap images.
Chạy standalone: bbox images + bbox_info.json → output/m2_depth/

Usage:
  python scripts/step2_depth_images.py
  python scripts/step2_depth_images.py --max-samples 50
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
    IMAGE_DIR, M1_OUTPUT_DIR, M2_OUTPUT_DIR, M2_DEPTH_INFO_FILE,
    DEPTH_MODEL_SIZE, DEPTH_COLORMAP,
)
from models.image_generator import (
    compute_depth_cue, render_depth_image,
    load_depth_model, should_run_grounding,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox-info", default=M1_BBOX_INFO_FILE)
    parser.add_argument("--image-dir", default=IMAGE_DIR)
    parser.add_argument("--output-dir", default=M2_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load bbox_info
    with open(args.bbox_info, "r", encoding="utf-8") as f:
        bbox_info = json.load(f)

    # Load extraction để lấy record gốc (cho should_run_grounding)
    from config.pipeline_config import EXTRACTION_FILE
    with open(EXTRACTION_FILE, "r", encoding="utf-8") as f:
        extraction_records = {str(r["id"]): r for r in json.load(f)}

    if args.max_samples > 0:
        bbox_items = [(k, v) for k, v in list(bbox_info.items())[:args.max_samples]]
    else:
        bbox_items = list(bbox_info.items())

    # Load depth model
    print("[M2] Loading Depth-Anything-V2...")
    depth_model = None
    try:
        depth_model = load_depth_model(DEPTH_MODEL_SIZE, args.device)
        print(f"[M2] Depth-Anything-V2 loaded on {args.device}")
    except Exception as e:
        print(f"[M2] WARNING: Could not load Depth model: {e}")
        print("[M2] Will still generate images but depth values will be zero.")

    depth_info = {}
    ok_count = 0
    skip_count = 0

    for i, (sid, box_info) in enumerate(bbox_items):
        record = extraction_records.get(sid, {})

        # Confidence gating (lặp lại để đồng bộ)
        if not should_run_grounding(record):
            skip_count += 1
            depth_info[sid] = {
                "depth_ok": False,
                "depth_o1": None,
                "depth_o2": None,
                "relation_text": None,
                "skip_reason": "confidence_gating",
            }
            print(f"[{i+1}/{len(bbox_items)}] id={sid} SKIP (confidence gating)")
            continue

        # marks_ok phải True mới chạy depth
        if not box_info.get("marks_ok", False):
            skip_count += 1
            depth_info[sid] = {
                "depth_ok": False,
                "depth_o1": None,
                "depth_o2": None,
                "relation_text": None,
                "skip_reason": "marks_failed",
            }
            # Vẫn tạo placeholder image (ảnh gốc ko có depth)
            bbox_img_path = os.path.join(args.image_dir, record.get("image", ""))
            img = Image.open(bbox_img_path).convert("RGB")
            out_path = os.path.join(args.output_dir, f"{sid}_depth.jpg")
            img.save(out_path, quality=85)
            print(f"[{i+1}/{len(bbox_items)}] id={sid} SKIP (marks_failed)")
            continue

        # Load ảnh gốc để tính depth
        img_name = record.get("image", "")
        img_path = os.path.join(args.image_dir, img_name)
        if not os.path.exists(img_path):
            print(f"[{i+1}/{len(bbox_items)}] id={sid} ERROR: image not found: {img_name}")
            depth_info[sid] = {"depth_ok": False, "skip_reason": "image_not_found"}
            continue

        img = Image.open(img_path).convert("RGB")

        # Compute depth
        depth_map, depth_o1, depth_o2 = compute_depth_cue(
            image=img,
            box_info=box_info,
            depth_model=depth_model,
            device=args.device,
            model_size=DEPTH_MODEL_SIZE,
        )

        is_viewer = box_info.get("o2_is_viewer", False)

        # Render depth image
        depth_img = render_depth_image(
            depth_map=depth_map,
            box_info=box_info,
            depth_o1=depth_o1,
            depth_o2=depth_o2,
            is_viewer=is_viewer,
            colormap=DEPTH_COLORMAP,
        )

        # Save
        out_path = os.path.join(args.output_dir, f"{sid}_depth.jpg")
        depth_img.save(out_path, quality=85)

        from models.image_generator import depth_relation_text
        o1_name = box_info.get("o1_name", "")
        o2_name = box_info.get("o2_name", "")
        relation_text = depth_relation_text(
            depth_o1, depth_o2, o1_name, o2_name or record.get("Object", {}).get("O2", "")
        )

        depth_info[sid] = {
            "depth_ok": True,
            "depth_o1": round(depth_o1, 4),
            "depth_o2": round(depth_o2, 4),
            "relation_text": relation_text,
        }
        ok_count += 1
        print(f"[{i+1}/{len(bbox_items)}] id={sid} OK  depth=[1]={depth_o1:.3f}, "
              f"[2]={depth_o2:.3f} (viewer={is_viewer})")

    # Save depth_info.json
    with open(M2_DEPTH_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(depth_info, f, ensure_ascii=False, indent=2)

    print(f"\n[M2] Done: {ok_count} OK, {skip_count} SKIP | Total: {len(bbox_items)}")
    print(f"[M2] Images → {args.output_dir}/")
    print(f"[M2] Info   → {M2_DEPTH_INFO_FILE}")


if __name__ == "__main__":
    main()
