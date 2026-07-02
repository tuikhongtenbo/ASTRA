"""
debug_pipeline.py — ASTRA v2: Chạy full pipeline trên 10-20 samples,
xuất tất cả intermediate outputs ra output/debug/ để review bằng mắt.

Ưu tiên chọn samples có:
  - O2_is_viewer=True và False
  - confidence >= threshold và < threshold
  - marks_ok=True và False

Usage:
  python scripts/debug_pipeline.py
  python scripts/debug_pipeline.py --num-samples 15 --device cuda
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from PIL import Image

from config.pipeline_config import (
    EXTRACTION_FILE, IMAGE_DIR, DEBUG_DIR, M1_OUTPUT_DIR,
    DET_CONF_THRESHOLD, DEPTH_MODEL_SIZE, DEPTH_COLORMAP,
    DEPTH_DIFF_THRESHOLD,
)
from models.image_generator import (
    generate_bbox_image, compute_depth_cue, render_depth_image,
    should_run_grounding, load_grounding_model, load_depth_model,
    depth_relation_text,
)
from models.prompt_v2 import build_prompt


def find_image_path(image_name: str) -> str | None:
    candidates = [
        os.path.join(IMAGE_DIR, image_name),
        os.path.join(IMAGE_DIR, image_name.replace(".jpg", ".png")),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def select_diverse_samples(records: list, num: int = 15) -> list:
    """
    Chọn num samples đa dạng, ưu tiên có đủ các loại:
    - viewer=True, viewer=False
    - confidence >= 0.6, confidence < 0.6
    - O1/O2_hallucinated=True, False
    """
    groups = {
        "high_conf_viewer": [],       # conf>=0.6, viewer=True
        "high_conf_nonviewer": [],    # conf>=0.6, viewer=False
        "low_conf": [],               # conf<0.6
        "hallucination": [],          # O1/O2_hallucinated=True
    }

    for r in records:
        sid = str(r.get("id", ""))
        is_viewer = bool(r.get("O2_is_viewer", False))
        conf = r.get("confidence", 0.0)
        halluc = r.get("O1_hallucinated", False) or r.get("O2_hallucinated", False)

        if halluc:
            groups["hallucination"].append(sid)
        elif conf < 0.6:
            groups["low_conf"].append(sid)
        elif is_viewer:
            groups["high_conf_viewer"].append(sid)
        else:
            groups["high_conf_nonviewer"].append(sid)

    selected = []
    per_group = max(2, num // len(groups))
    for group_name, sids in groups.items():
        sample = sids[:per_group]
        selected.extend(sample)

    # Shuffle and trim
    random.shuffle(selected)
    return selected[:num]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=DEBUG_DIR)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load extraction
    with open(EXTRACTION_FILE, "r", encoding="utf-8") as f:
        all_records = json.load(f)
    records_by_id = {str(r["id"]): r for r in all_records}

    # Select diverse samples
    selected_ids = select_diverse_samples(all_records, args.num_samples)
    print(f"[Debug] Selected {len(selected_ids)} diverse samples:")
    for sid in selected_ids:
        r = records_by_id.get(sid, {})
        print(f"  id={sid}: viewer={r.get('O2_is_viewer')}, "
              f"conf={r.get('confidence', 0):.2f}, "
              f"halluc={r.get('O1_hallucinated', False) or r.get('O2_hallucinated', False)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    print("\n[Debug] Loading models...")
    grounding_model, grounding_processor = None, None
    try:
        grounding_model, grounding_processor, _ = load_grounding_model(args.device)
        print(f"[Debug] Grounding DINO loaded on {args.device}")
    except Exception as e:
        print(f"[Debug] WARNING: Grounding DINO failed: {e}")

    depth_model = None
    try:
        depth_model = load_depth_model(DEPTH_MODEL_SIZE, args.device)
        print(f"[Debug] Depth-Anything loaded on {args.device}")
    except Exception as e:
        print(f"[Debug] WARNING: Depth-Anything failed: {e}")

    # Process each sample
    results = []
    for idx, sid in enumerate(selected_ids):
        record = records_by_id.get(sid, {})
        debug_dir = os.path.join(args.output_dir, f"{idx}_{sid}")
        os.makedirs(debug_dir, exist_ok=True)

        print(f"\n[{idx+1}/{len(selected_ids)}] Processing id={sid}...")

        img_name = record.get("image", "")
        img_path = find_image_path(img_name)

        if not img_path:
            print(f"  ERROR: image not found: {img_name}")
            continue

        img = Image.open(img_path).convert("RGB")

        # Save original
        img.save(os.path.join(debug_dir, "image_orig.jpg"), quality=95)

        meta = {
            "id": sid,
            "image": img_name,
            "question": record.get("question", ""),
            "options": record.get("options", []),
            "answer": record.get("answer", ""),
            "o1": record.get("Object", {}).get("O1", ""),
            "o2": record.get("Object", {}).get("O2", ""),
            "o2_is_viewer": bool(record.get("O2_is_viewer", False)),
            "confidence": record.get("confidence", 0.0),
            "o1_hallucinated": bool(record.get("O1_hallucinated", False)),
            "o2_hallucinated": bool(record.get("O2_hallucinated", False)),
        }

        # ── Confidence gating check ──
        gating_pass = should_run_grounding(record)
        meta["gating_pass"] = gating_pass

        # ── Module 1: Bbox ──
        marks_ok = False
        box_info = {}
        marked_img = img.copy()

        if gating_pass:
            marked_img, box_info = generate_bbox_image(
                image=img,
                record=record,
                grounding_model=grounding_model,
                processor=grounding_processor,
                device=args.device,
                det_threshold=DET_CONF_THRESHOLD,
            )
            marks_ok = box_info.get("marks_ok", False)
        else:
            box_info = {
                "marks_ok": False,
                "box_o1": None,
                "box_o2": None,
                "o1_name": record.get("Object", {}).get("O1", ""),
                "o2_name": record.get("Object", {}).get("O2", ""),
                "o2_is_viewer": bool(record.get("O2_is_viewer", False)),
                "conf_o1": 0.0,
                "conf_o2": 0.0,
            }

        meta["marks_ok"] = marks_ok
        meta["box_info"] = box_info

        # Save Image 1 (bbox-marked)
        img1_path = os.path.join(debug_dir, "image1_bbox.jpg")
        marked_img.save(img1_path, quality=95)

        print(f"  marks_ok={marks_ok}, box_o1={box_info.get('box_o1')}, "
              f"box_o2={box_info.get('box_o2')}")

        # ── Module 2: Depth ──
        depth_ok = False
        depth_img = img.copy()
        depth_map = None
        depth_o1 = 0.0
        depth_o2 = 0.0
        relation_text = ""

        if marks_ok and depth_model is not None:
            depth_map, depth_o1, depth_o2 = compute_depth_cue(
                image=img,
                box_info=box_info,
                depth_model=depth_model,
                device=args.device,
                model_size=DEPTH_MODEL_SIZE,
            )
            is_viewer = box_info.get("o2_is_viewer", False)
            depth_img = render_depth_image(
                depth_map=depth_map,
                box_info=box_info,
                depth_o1=depth_o1,
                depth_o2=depth_o2,
                is_viewer=is_viewer,
                colormap=DEPTH_COLORMAP,
            )
            depth_ok = True

            o1_name = box_info.get("o1_name", "")
            o2_name = box_info.get("o2_name", "")
            relation_text = depth_relation_text(
                depth_o1, depth_o2, o1_name, o2_name, DEPTH_DIFF_THRESHOLD
            )

        meta["depth_ok"] = depth_ok
        meta["depth_o1"] = round(depth_o1, 4)
        meta["depth_o2"] = round(depth_o2, 4)
        meta["relation_text"] = relation_text

        # Save Image 2 (depth heatmap)
        img2_path = os.path.join(debug_dir, "image2_depth.jpg")
        depth_img.save(img2_path, quality=85)

        print(f"  depth_ok={depth_ok}, d1={depth_o1:.3f}, d2={depth_o2:.3f}")

        # ── Build prompt ──
        prompt = build_prompt(
            record=record,
            marks_ok=marks_ok and depth_ok,
            depth_o1=depth_o1,
            depth_o2=depth_o2,
        )
        meta["prompt"] = prompt

        prompt_path = os.path.join(debug_dir, "prompt.txt")
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)

        # Save meta.json
        meta_path = os.path.join(debug_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        results.append(meta)
        print(f"  → {debug_dir}/")

    # Summary
    gating_pass_count = sum(1 for r in results if r.get("gating_pass"))
    marks_ok_count = sum(1 for r in results if r.get("marks_ok"))
    depth_ok_count = sum(1 for r in results if r.get("depth_ok"))

    print(f"\n[Debug] Summary:")
    print(f"  Total samples: {len(results)}")
    print(f"  Gating pass: {gating_pass_count}/{len(results)}")
    print(f"  M1 marks_ok: {marks_ok_count}/{len(results)}")
    print(f"  M2 depth_ok: {depth_ok_count}/{len(results)}")
    print(f"  Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
