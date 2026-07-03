"""
step3_vlm_inference.py — ASTRA v2, Module 3 (ODV) + VLM Inference với vLLM.
Dùng vLLM engine để batch inference nhanh, thay vì HF transformers sequential.

Usage:
  python scripts/step3_vlm_inference.py
  python scripts/step3_vlm_inference.py --model Qwen3-VL-4B --max-samples 50
  python scripts/step3_vlm_inference.py --model Qwen3-VL-2B --tensor-parallel-size 2
  python scripts/step3_vlm_inference.py --model Qwen3-VL-4B --resume          # tiếp tục từ checkpoint
"""

import argparse
import base64
import json
import os
import random
import sys
import time
from collections import Counter
from io import BytesIO
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from PIL import Image

from config.pipeline_config import (
    EXTRACTION_FILE, M1_OUTPUT_DIR, M2_OUTPUT_DIR,
    VLM_OUTPUT_DIR, VLM_RESULTS_FILE, N_PERMS, MAX_NEW_TOKENS,
)
from models.prompt_v2 import build_prompt
from utils.utils import normalize_relation


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def image_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    """Convert PIL Image → base64 data URL."""
    buf = BytesIO()
    img.save(buf, format=fmt, quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


def load_all_data():
    """Load extraction records, bbox_info, depth_info."""
    with open(EXTRACTION_FILE, "r", encoding="utf-8") as f:
        records = {str(r["id"]): r for r in json.load(f)}

    bbox_path = os.path.join(M1_OUTPUT_DIR, "bbox_info.json")
    with open(bbox_path, "r", encoding="utf-8") as f:
        bbox_info = json.load(f)

    depth_path = os.path.join(M2_OUTPUT_DIR, "depth_info.json")
    if os.path.exists(depth_path):
        with open(depth_path, "r", encoding="utf-8") as f:
            depth_info = json.load(f)
    else:
        depth_info = {}

    return records, bbox_info, depth_info


def generate_permutations(options, n):
    """Generate n unique permutations for ODV."""
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
    if all_options not in perms:
        perms.insert(0, all_options)
    return perms[:n]


def parse_answer(output: str, perm_opts, original_opts) -> str | None:
    """Parse VLM output and normalize to original option text."""
    if not output:
        return None
    pred = normalize_relation(output, perm_opts)
    if pred is None:
        return None
    for original in original_opts:
        if original.strip().lower() == pred.strip().lower():
            return original
    return pred


def vote_answers(answers: list[str | None]) -> str:
    """Majority vote."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return ""
    counter = Counter(valid)
    return counter.most_common(1)[0][0]


# ─────────────────────────────────────────────────────────────────────────────
# vLLM Inference Engine
# ─────────────────────────────────────────────────────────────────────────────

class VLMInferenceEngine:
    """
    vLLM-based inference cho Qwen3-VL.
    Dùng batched requests để tận dụng GPU qua vLLM engine.

    Supported models:
      - Qwen3-VL-2B / Qwen3-VL-4B / Qwen3-VL-8B
      - Qwen2-VL-2B / Qwen2-VL-7B
    """

    def __init__(self, model_name: str, device: str = "cuda",
                 tensor_parallel_size: int = 1,
                 max_model_len: int = 8192,
                 gpu_memory_utilization: float = 0.85,
                 **llm_kwargs):
        from config.config import MODEL_ALIASES

        self.model_name = MODEL_ALIASES.get(model_name, model_name)
        self.device = device
        self.tensor_parallel_size = tensor_parallel_size
        self.llm = None
        self.processor = None
        self._load(model_name, max_model_len, gpu_memory_utilization, llm_kwargs)

    def _load(self, model_name: str, max_model_len: int,
              gpu_memory_utilization: float, llm_kwargs):
        """Load vLLM LLM engine + HuggingFace processor (chỉ dùng cho preprocess)."""
        print(f"[vLLM] Loading {self.model_name} ...")
        t0 = time.time()

        # Load HuggingFace processor cho image preprocessing
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        # Load vLLM engine
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError(
                "vllm not installed. Install: pip install vllm>=0.6.0"
            )

        self.llm = LLM(
            model=self.model_name,
            trust_remote_code=True,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="bfloat16" if self.device == "cuda" else "float16",
            limit_mm_per_prompt={"image": 2},
            **llm_kwargs,
        )
        self.tokenizer = self.llm.get_tokenizer()
        print(f"[vLLM] Engine ready in {time.time() - t0:.1f}s")

    def _build_chat_messages(self, image1_url: str, image2_url: str | None,
                              prompt: str) -> list[dict]:
        """Build chatml messages dict cho vLLM."""
        if image2_url is not None:
            content = [
                {"type": "image", "image": image1_url},
                {"type": "image", "image": image2_url},
                {"type": "text", "text": prompt},
            ]
        else:
            content = [
                {"type": "image", "image": image1_url},
                {"type": "text", "text": prompt},
            ]
        return [{"role": "user", "content": content}]

    def _to_mm_data(self, image1_url: str, image2_url: str | None) -> dict | None:
        """Build mm_data dict cho vLLM multi-modal."""
        if image2_url is not None:
            return {"image": [image1_url, image2_url]}
        return {"image": image1_url}

    def generate_batch(self, requests: list[dict],
                       max_new_tokens: int = MAX_NEW_TOKENS) -> list[str]:
        """
        Batch generate cho nhiều requests cùng lúc qua vLLM.

        Args:
            requests: list of {
                "image1_url": str,       # base64 data URL
                "image2_url": str|None,  # base64 data URL hoặc None
                "prompt": str,
            }

        Returns:
            list of generated strings (cùng thứ tự với requests)
        """
        from vllm.distributed.parallel_utils import ParallelState
        from vllm.sampling_params import SamplingParams

        messages_list = [
            self._build_chat_messages(r["image1_url"], r.get("image2_url"), r["prompt"])
            for r in requests
        ]
        mm_data_list = [
            self._to_mm_data(r["image1_url"], r.get("image2_url"))
            for r in requests
        ]

        # Apply chat template
        prompt_list = [
            self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_list
        ]

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0,
            stop_strings=["<|im_end|>", "<|endoftext|>"],
        )

        outputs = self.llm.generate(
            prompt_list,
            sampling_params=sampling_params,
            multi_modal_data=mm_data_list,
        )

        results = []
        for output in outputs:
            text = output.outputs[0].text.strip()
            results.append(text)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Prepare batch data
# ─────────────────────────────────────────────────────────────────────────────

def prepare_requests(records: dict, bbox_info: dict, depth_info: dict,
                     sids: list[str]) -> list[dict]:
    """
    Chuẩn bị tất cả inference requests cho batch processing.
    Mỗi sample × N_PERMS = 1 request.
    """
    requests = []
    request_meta = []  # track metadata để reconstruct results

    for sid in sids:
        record = records.get(sid, {})
        box = bbox_info.get(sid, {})
        depth = depth_info.get(sid, {})

        options = record.get("options", [])
        marks_ok = bool(box.get("marks_ok", False))
        depth_ok = bool(depth.get("depth_ok", False))
        depth_o1 = depth.get("depth_o1", 0.0) or 0.0
        depth_o2 = depth.get("depth_o2", 0.0) or 0.0

        # Load 2 ảnh → base64
        img1_url = None
        img1_path = os.path.join(M1_OUTPUT_DIR, f"{sid}_bbox.jpg")
        if os.path.exists(img1_path):
            img1 = Image.open(img1_path).convert("RGB")
            img1_url = image_to_base64(img1, "JPEG")

        img2_url = None
        img2_path = os.path.join(M2_OUTPUT_DIR, f"{sid}_depth.jpg")
        if marks_ok and depth_ok and os.path.exists(img2_path):
            img2 = Image.open(img2_path).convert("RGB")
            img2_url = image_to_base64(img2, "JPEG")

        use_two = (img1_url is not None and img2_url is not None)
        prompt_marks_ok = marks_ok and depth_ok

        # ODV permutations
        perms = generate_permutations(options, N_PERMS)

        for perm_idx, perm_opts in enumerate(perms):
            p = build_prompt(
                record=record,
                marks_ok=prompt_marks_ok,
                depth_o1=depth_o1,
                depth_o2=depth_o2,
                options=perm_opts,
            )

            req = {
                "image1_url": img1_url,
                "image2_url": img2_url,
                "prompt": p,
            }
            requests.append(req)

            request_meta.append({
                "sid": sid,
                "perm_idx": perm_idx,
                "perm_opts": perm_opts,
                "original_opts": options,
                "answer": record.get("answer", ""),
                "question": record.get("question", ""),
                "marks_ok": marks_ok,
                "depth_ok": depth_ok,
                "depth_o1": depth_o1,
                "depth_o2": depth_o2,
                "relation_text": depth.get("relation_text", ""),
            })

    return requests, request_meta


def reconstruct_results(request_meta: list, outputs: list[str]) -> list[dict]:
    """
    Reconstruct per-sample results từ batch outputs + meta.
    Mỗi sample có N_PERMS outputs → majority vote.
    """
    # Group by sid
    from collections import defaultdict
    samples = defaultdict(list)

    for meta, output in zip(request_meta, outputs):
        samples[meta["sid"]].append((meta, output))

    results = []
    correct_count = 0

    for sid, items in samples.items():
        meta0 = items[0][0]

        votes = []
        vote_outputs = []
        for m, output in items:
            parsed = parse_answer(output, m["perm_opts"], m["original_opts"])
            votes.append(parsed)
            vote_outputs.append(output)

        predicted = vote_answers(votes)
        pred_norm = normalize_relation(predicted, m["original_opts"]) or predicted or ""
        ans_norm = normalize_relation(m["answer"], m["original_opts"]) or m["answer"]
        correct = pred_norm.lower().strip() == ans_norm.lower().strip()
        if correct:
            correct_count += 1

        results.append({
            "id": sid,
            "question": m["question"],
            "options": m["original_opts"],
            "answer": m["answer"],
            "predicted": pred_norm,
            "correct": correct,
            "marks_ok": m["marks_ok"],
            "depth_ok": m["depth_ok"],
            "depth_o1": m["depth_o1"],
            "depth_o2": m["depth_o2"],
            "relation_text": m["relation_text"],
            "votes": votes,
            "vote_outputs": vote_outputs,
        })

    return results, correct_count


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ASTRA v2 Step 3: vLLM batch inference với ODV voting"
    )
    parser.add_argument("--model", default="Qwen3-VL-4B",
                        help="Model name hoặc path (Qwen3-VL-4B, Qwen3-VL-2B, ...)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="0 = chạy tất cả")
    parser.add_argument("--output-dir", default=VLM_OUTPUT_DIR)
    parser.add_argument("--tensor-parallel-size", "--tp", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", "--gpu-mem", type=float, default=0.85)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Số requests/batch gửi cho vLLM engine")
    parser.add_argument("--resume", action="store_true",
                        help="Đọc kết quả cũ, bỏ qua những sample đã chạy xong")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "results.jsonl")

    # Load data
    records, bbox_info, depth_info = load_all_data()
    sids = list(records.keys())
    if args.max_samples > 0:
        sids = sids[:args.max_samples]

    # Resume: đọc kết quả cũ, xác định sids đã chạy
    done_sids = set()
    if args.resume and os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done_sids.add(str(r["id"]))
        sids = [s for s in sids if s not in done_sids]
        print(f"[vLLM] Resuming: {len(done_sids)} done, {len(sids)} remaining")

    if not sids:
        print("[vLLM] All samples already processed.")
        return

    print(f"[vLLM] Processing {len(sids)} samples × {N_PERMS} perms = "
          f"{len(sids) * N_PERMS} total requests")

    # Load vLLM engine
    engine = VLMInferenceEngine(
        model_name=args.model,
        device=args.device,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Prepare batch requests
    print("[vLLM] Preparing requests...")
    requests, request_meta = prepare_requests(records, bbox_info, depth_info, sids)
    total_reqs = len(requests)
    print(f"[vLLM] {total_reqs} requests prepared")

    # Batch inference
    print(f"[vLLM] Running batch inference (batch_size={args.batch_size})...")
    all_outputs = []
    t_start = time.time()

    for batch_start in range(0, total_reqs, args.batch_size):
        batch_end = min(batch_start + args.batch_size, total_reqs)
        batch_reqs = requests[batch_start:batch_end]
        batch_meta = request_meta[batch_start:batch_end]

        outputs = engine.generate_batch(batch_reqs, MAX_NEW_TOKENS)
        all_outputs.extend(outputs)

        elapsed = time.time() - t_start
        done = len(all_outputs)
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total_reqs - done) / rate if rate > 0 else 0
        print(f"[vLLM]  batch {batch_start}-{batch_end} / {total_reqs}  "
              f"({done}/{total_reqs})  {rate:.1f} req/s  ETA={eta/60:.1f}m")

    # Reconstruct results (per-sample, ODV vote)
    print("[vLLM] Reconstructing results & voting...")
    results, correct_count = reconstruct_results(request_meta, all_outputs)

    # Merge with existing results if resuming
    if args.resume and os.path.exists(output_file):
        existing = []
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                existing.append(json.loads(line))
        existing_ids = {str(r["id"]) for r in existing}
        new_results = [r for r in results if str(r["id"]) not in existing_ids]
        results = existing + new_results
        correct_count = sum(1 for r in results if r["correct"])

    # Save results.jsonl
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Print summary per marks_ok / depth_ok
    marks_ok_count = sum(1 for r in results if r["marks_ok"])
    depth_ok_count = sum(1 for r in results if r["depth_ok"])
    marks_ok_correct = sum(1 for r in results if r["marks_ok"] and r["correct"])
    depth_ok_correct = sum(1 for r in results if r["depth_ok"] and r["correct"])

    total = len(results)
    acc = correct_count / total if total > 0 else 0
    acc_marks = marks_ok_correct / marks_ok_count if marks_ok_count > 0 else 0
    acc_depth = depth_ok_correct / depth_ok_count if depth_ok_count > 0 else 0

    elapsed = time.time() - t_start
    print(f"\n[vLLM] ========== RESULTS ==========")
    print(f"[vLLM] Total:      {correct_count}/{total} correct ({acc:.1%})")
    print(f"[vLLM] marks_ok:  {marks_ok_correct}/{marks_ok_count} ({acc_marks:.1%})")
    print(f"[vLLM] depth_ok:   {depth_ok_correct}/{depth_ok_count} ({acc_depth:.1%})")
    print(f"[vLLM] Time:       {elapsed/60:.1f} min  ({total_reqs/elapsed:.1f} req/s)")
    print(f"[vLLM] Output:     {output_file}")


if __name__ == "__main__":
    main()
