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
import inspect
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
from tqdm import tqdm

from config.pipeline_config import (
    DATA_DIR, EXTRACTION_FILE, V2_BBOX_IMAGE_DIR, V2_DEPTH_BBOX_IMAGE_DIR,
    VLM_OUTPUT_DIR, VLM_RESULTS_FILE, N_PERMS, MAX_NEW_TOKENS,
)
from models.prompt_v2 import build_prompt, format_options
from utils.utils import normalize_relation

ZERO_SHOT_FILE = os.path.join(DATA_DIR, "test.jsonl")
ZERO_SHOT_IMAGE_DIR = os.path.join(DATA_DIR, "test_images")

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
    """Load extraction records keyed by sample id."""
    with open(EXTRACTION_FILE, "r", encoding="utf-8") as f:
        return {str(r["id"]): r for r in json.load(f)}



def load_jsonl_data(path: str):
    """Load JSONL records keyed by sample id."""
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "id" not in record:
                raise ValueError(f"Record at {path}:{line_no} is missing the 'id' field")
            records[str(record["id"])] = record
    return records


def get_v2_image_paths(record: dict) -> tuple[str, str]:
    """Return marked RGB and marked depth image paths for a record."""
    image_name = record.get("image")
    if not image_name:
        raise ValueError(f"Record {record.get('id')} is missing the 'image' field")

    image_name = os.path.basename(image_name)
    image_stem, _ = os.path.splitext(image_name)
    img1_path = os.path.join(V2_BBOX_IMAGE_DIR, image_name)
    img2_path = os.path.join(V2_DEPTH_BBOX_IMAGE_DIR, f"{image_stem}_depth.jpg")

    missing = [path for path in (img1_path, img2_path) if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            f"Missing required v2 image(s) for record {record.get('id')} "
            f"({image_name}): {missing}"
        )

    return img1_path, img2_path



def get_zero_shot_image_path(record: dict) -> str:
    """Return the raw test image path for a zero-shot record."""
    image_name = record.get("image")
    if not image_name:
        raise ValueError(f"Record {record.get('id')} is missing the 'image' field")

    image_name = os.path.basename(image_name)
    image_path = os.path.join(ZERO_SHOT_IMAGE_DIR, image_name)
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Missing zero-shot image for record {record.get('id')} ({image_name}): "
            f"{image_path}. Expected images under {ZERO_SHOT_IMAGE_DIR}"
        )
    return image_path


def build_zero_shot_prompt(record: dict, options: list[str] | None = None) -> str:
    """Build a one-image zero-shot prompt from question/options."""
    if options is None:
        options = record.get("options", [])

    return (
        "You are given one image.\n"
        "Answer the spatial reasoning question using only the visual content.\n\n"
        f"Question: {record.get('question', '')}\n\n"
        "Options:\n"
        f"{format_options(options)}\n\n"
        "Think briefly, then answer in exactly this format:\n"
        "Answer: (X)"
    )


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

    def _to_mm_data(self, image1, image2=None) -> dict | None:
        """Build mm_data dict for vLLM multi-modal inputs."""
        if image2 is not None:
            return {"image": [image1, image2]}
        return {"image": image1}

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
        from vllm.sampling_params import SamplingParams

        messages_list = [
            self._build_chat_messages(r["image1_url"], r.get("image2_url"), r["prompt"])
            for r in requests
        ]
        mm_data_list = [
            self._to_mm_data(
                r.get("image1", r["image1_url"]),
                r.get("image2", r.get("image2_url")),
            )
            for r in requests
        ]

        # Apply chat template
        prompt_list = [
            self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_list
        ]

        stop_sequences = ["<|im_end|>", "<|endoftext|>"]
        sampling_kwargs = {
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
        }
        sampling_sig = inspect.signature(SamplingParams).parameters
        if "stop" in sampling_sig:
            sampling_kwargs["stop"] = stop_sequences
        elif "stop_strings" in sampling_sig:
            sampling_kwargs["stop_strings"] = stop_sequences
        sampling_params = SamplingParams(**sampling_kwargs)

        generate_sig = inspect.signature(self.llm.generate).parameters
        if "multi_modal_data" in generate_sig:
            outputs = self.llm.generate(
                prompt_list,
                sampling_params=sampling_params,
                multi_modal_data=mm_data_list,
            )
        else:
            prompt_inputs = [
                {"prompt": prompt, "multi_modal_data": mm_data}
                for prompt, mm_data in zip(prompt_list, mm_data_list)
            ]
            outputs = self.llm.generate(
                prompt_inputs,
                sampling_params=sampling_params,
            )

        results = []
        for output in outputs:
            text = output.outputs[0].text.strip()
            results.append(text)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Prepare batch data
# ─────────────────────────────────────────────────────────────────────────────

def prepare_requests(records: dict, sids: list[str]) -> list[dict]:
    """
    Prepare all inference requests for batch processing.
    Each sample x N_PERMS becomes one request.
    """
    requests = []
    request_meta = []

    for sid in sids:
        record = records.get(sid, {})
        options = record.get("options", [])

        img1_path, img2_path = get_v2_image_paths(record)
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        img1_url = image_to_base64(img1, "JPEG")
        img2_url = image_to_base64(img2, "JPEG")

        perms = generate_permutations(options, N_PERMS)

        for perm_idx, perm_opts in enumerate(perms):
            p = build_prompt(
                record=record,
                marks_ok=True,
                depth_o1=0.0,
                depth_o2=0.0,
                options=perm_opts,
            )

            requests.append({
                "image1_url": img1_url,
                "image2_url": img2_url,
                "image1": img1,
                "image2": img2,
                "prompt": p,
            })

            request_meta.append({
                "sid": sid,
                "image": record.get("image", ""),
                "perm_idx": perm_idx,
                "perm_opts": perm_opts,
                "original_opts": options,
                "answer": record.get("answer", ""),
                "question": record.get("question", ""),
                "marks_ok": True,
                "depth_ok": True,
                "depth_o1": 0.0,
                "depth_o2": 0.0,
                "relation_text": "",
                "mode": "astra",
            })

    return requests, request_meta



def prepare_zero_shot_requests(records: dict, sids: list[str]) -> tuple[list[dict], list[dict]]:
    """
    Prepare one-image zero-shot requests.
    Each sample x N_PERMS becomes one request for ODV voting.
    """
    requests = []
    request_meta = []

    for sid in sids:
        record = records.get(sid, {})
        options = record.get("options", [])

        img_path = get_zero_shot_image_path(record)
        img = Image.open(img_path).convert("RGB")
        img_url = image_to_base64(img, "JPEG")

        perms = generate_permutations(options, N_PERMS)

        for perm_idx, perm_opts in enumerate(perms):
            p = build_zero_shot_prompt(record, options=perm_opts)

            requests.append({
                "image1_url": img_url,
                "image2_url": None,
                "image1": img,
                "image2": None,
                "prompt": p,
            })

            request_meta.append({
                "sid": sid,
                "image": record.get("image", ""),
                "perm_idx": perm_idx,
                "perm_opts": perm_opts,
                "original_opts": options,
                "answer": record.get("answer", ""),
                "question": record.get("question", ""),
                "marks_ok": False,
                "depth_ok": False,
                "depth_o1": 0.0,
                "depth_o2": 0.0,
                "relation_text": "",
                "mode": "zero-shot",
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
            "image": m.get("image", ""),
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
            "mode": m.get("mode", "astra"),
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
    parser.add_argument("--mode", choices=["astra", "zero-shot"], default="astra",
                        help="Inference mode: astra = two annotated images, zero-shot = raw test image")
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
    if args.mode == "zero-shot":
        records = load_jsonl_data(ZERO_SHOT_FILE)
        print(f"[vLLM] Mode: zero-shot | data: {ZERO_SHOT_FILE}")
        print(f"[vLLM] Zero-shot image dir: {ZERO_SHOT_IMAGE_DIR}")
    else:
        records = load_all_data()
        print(f"[vLLM] Mode: astra | data: {EXTRACTION_FILE}")

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

    print(f"[vLLM] Processing {len(sids)} samples x {N_PERMS} perms = "
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
    if args.mode == "zero-shot":
        requests, request_meta = prepare_zero_shot_requests(records, sids)
    else:
        requests, request_meta = prepare_requests(records, sids)
    total_reqs = len(requests)
    print(f"[vLLM] {total_reqs} requests prepared")
    # Batch inference
    print(f"[vLLM] Running batch inference (batch_size={args.batch_size})...")
    all_outputs = []
    t_start = time.time()

    progress = tqdm(
        range(0, total_reqs, args.batch_size),
        total=(total_reqs + args.batch_size - 1) // args.batch_size,
        desc="[vLLM] batches",
        unit="batch",
    )
    for batch_start in progress:
        batch_end = min(batch_start + args.batch_size, total_reqs)
        batch_reqs = requests[batch_start:batch_end]

        outputs = engine.generate_batch(batch_reqs, MAX_NEW_TOKENS)
        all_outputs.extend(outputs)

        elapsed = time.time() - t_start
        done = len(all_outputs)
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total_reqs - done) / rate if rate > 0 else 0
        progress.set_postfix({
            "req": f"{done}/{total_reqs}",
            "req/s": f"{rate:.1f}",
            "eta_m": f"{eta / 60:.1f}",
        })

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
    req_rate = total_reqs / elapsed if elapsed > 0 else 0.0
    print(f"\n[vLLM] ========== RESULTS ==========")
    print(f"[vLLM] Mode:       {args.mode}")
    print(f"[vLLM] Total:      {correct_count}/{total} correct ({acc:.1%})")
    if args.mode == "astra":
        print(f"[vLLM] marks_ok:  {marks_ok_correct}/{marks_ok_count} ({acc_marks:.1%})")
        print(f"[vLLM] depth_ok:   {depth_ok_correct}/{depth_ok_count} ({acc_depth:.1%})")
    else:
        print("[vLLM] marks_ok:   n/a for zero-shot")
        print("[vLLM] depth_ok:   n/a for zero-shot")
    print(f"[vLLM] Time:       {elapsed/60:.1f} min  ({req_rate:.1f} req/s)")
    print(f"[vLLM] Output:     {output_file}")

if __name__ == "__main__":
    main()
