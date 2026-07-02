"""
step3_vlm_inference.py — ASTRA v2, Module 3 (ODV) + VLM Inference.
Chạy standalone: 2 ảnh (bbox + depth) + prompt → results.jsonl

Usage:
  python scripts/step3_vlm_inference.py
  python scripts/step3_vlm_inference.py --model Qwen3-VL-4B
  python scripts/step3_vlm_inference.py --model Qwen3-VL-2B --max-samples 50
"""

import argparse
import json
import os
import time
from collections import Counter

from PIL import Image

from config.pipeline_config import (
    EXTRACTION_FILE, IMAGE_DIR, M1_OUTPUT_DIR, M2_OUTPUT_DIR,
    VLM_OUTPUT_DIR, VLM_RESULTS_FILE, N_PERMS, MAX_NEW_TOKENS,
)
from models.prompt_v2 import build_prompt
from utils.utils import normalize_relation


def load_all_data():
    """Load extraction records, bbox_info, depth_info."""
    with open(EXTRACTION_FILE, "r", encoding="utf-8") as f:
        records = {str(r["id"]): r for r in json.load(f)}

    with open(os.path.join(M1_OUTPUT_DIR, "bbox_info.json"), "r", encoding="utf-8") as f:
        bbox_info = json.load(f)

    depth_path = os.path.join(M2_OUTPUT_DIR, "depth_info.json")
    if os.path.exists(depth_path):
        with open(depth_path, "r", encoding="utf-8") as f:
            depth_info = json.load(f)
    else:
        depth_info = {}

    return records, bbox_info, depth_info


def load_vlm_model(model_name: str, device: str):
    """Load Qwen3-VL model và processor."""
    print(f"[VLM] Loading {model_name} on {device}...")
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    from config.config import MODEL_ALIASES
    model_id = MODEL_ALIASES.get(model_name, model_name)

    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"[VLM] Model loaded.")
    return model, processor


def generate(model, processor, image1: Image.Image, image2: Image.Image | None,
             prompt: str, device: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Generate answer từ VLM với 1 hoặc 2 ảnh."""
    from qwen_vl_utils import process_vision_info

    # Chuẩn bị images list
    if image2 is not None:
        images_for_vlm = [
            {"role": "user", "content": [
                {"type": "image", "image": image1},
                {"type": "image", "image": image2},
                {"type": "text", "text": prompt},
            ]}
        ]
    else:
        images_for_vlm = [
            {"role": "user", "content": [
                {"type": "image", "image": image1},
                {"type": "text", "text": prompt},
            ]}
        ]

    try:
        inputs = processor(
            text=images_for_vlm,
            images=image1 if image2 is None else [image1, image2],
            return_tensors="pt",
            padding=True,
        ).to(device)

        # Try process_vision_info for grid info
        try:
            pv, ig, _ = process_vision_info(
                {"image": image1} if image2 is None else {"image": [image1, image2]},
                return_image_grid=True,
            )
            if pv is not None and hasattr(pv, "to"):
                inputs["pixel_values"] = pv.to(device)
            if ig is not None and hasattr(ig, "to"):
                inputs["image_grid"] = ig.to(device)
        except Exception:
            pass

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        ilen = inputs["input_ids"].shape[1]
        return processor.batch_decode(
            output_ids[:, ilen:], skip_special_tokens=True
        )[0].strip()
    except Exception as e:
        # Fallback: simple single-image call
        try:
            inputs = processor(
                text=images_for_vlm,
                images=image1,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            ilen = inputs["input_ids"].shape[1]
            return processor.batch_decode(
                output_ids[:, ilen:], skip_special_tokens=True
            )[0].strip()
        except Exception:
            return ""


def generate_permutations(options, n):
    """Generate n random permutations of options for ODV."""
    import random
    perms = []
    seen = set()
    all_options = list(options)
    for _ in range(n * 3):
        perm = all_options.copy()
        random.shuffle(perm)
        key = tuple(perm)
        if key not in seen:
            seen.add(key)
            perms.append(perm)
        if len(perms) >= n:
            break
    # Luôn có original permutation
    if all_options not in perms:
        perms.insert(0, all_options)
    return perms[:n]


def parse_answer_from_output(output: str, perm_opts, original_opts) -> str | None:
    """Parse answer từ VLM output, chuẩn hóa về original option."""
    if not output:
        return None
    pred = normalize_relation(output, perm_opts)
    if pred is None:
        return None
    # Map từ perm_opt về original_opt
    idx = perm_opts.index(pred) if pred in perm_opts else -1
    if idx >= 0 and idx < len(original_opts):
        return original_opts[idx]
    return pred


def vote_answers(answers: list[str | None]) -> str:
    """Majority vote."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return ""
    counter = Counter(valid)
    return counter.most_common(1)[0][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen3-VL-4B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-dir", default=VLM_OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "results.jsonl")

    records, bbox_info, depth_info = load_all_data()
    sids = list(records.keys())
    if args.max_samples > 0:
        sids = sids[:args.max_samples]

    # Load VLM model
    model, processor = None, None
    try:
        model, processor = load_vlm_model(args.model, args.device)
    except Exception as e:
        print(f"[VLM] ERROR loading model: {e}")
        print("[VLM] Will write dummy results.")
        model, processor = None, None

    results = []
    correct_count = 0

    for i, sid in enumerate(sids):
        record = records[sid]
        box = bbox_info.get(sid, {})
        depth = depth_info.get(sid, {})

        options = record.get("options", [])
        answer = record.get("answer", "")
        question = record.get("question", "")
        marks_ok = bool(box.get("marks_ok", False))
        depth_ok = bool(depth.get("depth_ok", False))

        # Load 2 ảnh
        img1_path = os.path.join(M1_OUTPUT_DIR, f"{sid}_bbox.jpg")
        img2_path = os.path.join(M2_OUTPUT_DIR, f"{sid}_depth.jpg")

        img1 = None
        if os.path.exists(img1_path):
            img1 = Image.open(img1_path).convert("RGB")

        img2 = None
        if marks_ok and depth_ok and os.path.exists(img2_path):
            img2 = Image.open(img2_path).convert("RGB")

        # Build prompt
        depth_o1 = depth.get("depth_o1", 0.0) or 0.0
        depth_o2 = depth.get("depth_o2", 0.0) or 0.0

        if img2 is None:
            # Fallback: chỉ 1 ảnh
            prompt = build_prompt(record, marks_ok=False,
                                  depth_o1=depth_o1, depth_o2=depth_o2, options=options)
        else:
            prompt = build_prompt(record, marks_ok=True,
                                  depth_o1=depth_o1, depth_o2=depth_o2, options=options)

        # ODV voting
        perms = generate_permutations(options, N_PERMS)
        votes = []
        vote_outputs = []

        for perm_opts in perms:
            # Rebuild prompt với permuted options
            if img2 is None:
                p = build_prompt(record, marks_ok=False,
                                 depth_o1=depth_o1, depth_o2=depth_o2, options=perm_opts)
            else:
                p = build_prompt(record, marks_ok=True,
                                 depth_o1=depth_o1, depth_o2=depth_o2, options=perm_opts)

            if model is not None and img1 is not None:
                output = generate(model, processor, img1, img2, p, args.device, MAX_NEW_TOKENS)
            else:
                output = ""
            vote_outputs.append(output)
            parsed = parse_answer_from_output(output, perm_opts, options)
            votes.append(parsed)

        predicted = vote_answers(votes)
        pred_norm = normalize_relation(predicted, options) or predicted or ""
        ans_norm = normalize_relation(answer, options) or answer
        correct = pred_norm.lower().strip() == ans_norm.lower().strip()
        if correct:
            correct_count += 1

        result = {
            "id": sid,
            "question": question,
            "options": options,
            "answer": answer,
            "predicted": pred_norm,
            "correct": correct,
            "marks_ok": marks_ok,
            "depth_ok": depth_ok,
            "votes": votes,
            "vote_outputs": vote_outputs,
            "depth_o1": depth_o1,
            "depth_o2": depth_o2,
            "relation_text": depth.get("relation_text"),
        }
        results.append(result)

        acc = correct_count / (i + 1)
        print(f"[{i+1}/{len(sids)}] id={sid} pred={pred_norm} | ans={ans_norm} | "
              f"correct={correct} | acc={acc:.3f} | marks_ok={marks_ok} depth_ok={depth_ok}")

    # Save
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(results)
    acc = correct_count / total if total > 0 else 0
    print(f"\n[VLM] Done: {correct_count}/{total} correct ({acc:.1%})")
    print(f"[VLM] Results → {output_file}")


if __name__ == "__main__":
    main()
