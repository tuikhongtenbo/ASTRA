"""
main.py — CLI entry point cho ASTRA.
Chỉ có 2 chế độ: baseline (không module) và full ASTRA (M1+M2+M3).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import torch
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.pipeline import ASTRAPipeline
from config.config import (
    MODEL_ALIASES, DEFAULT_MODEL,
    DATA_DIR, IMAGE_DIR,
    DEPTH_EPSILON, CONFIDENCE_THRESHOLD, N_PERMS,
    ESCALATION_LOG_FILE,
)
from data_processing.dataset import SpatialMQADataset
from evaluation.evaluator import (
    evaluate_predictions, save_metrics, print_eval_report,
    build_ablation_summary, print_ablation_summary, export_ablation_csv,
)
from utils.utils import format_time, get_device, set_seed


def load_split(split: str, max_samples: int = None, lazy_load: bool = True):
    split_file = os.path.join(DATA_DIR, f"{split}.jsonl")
    if not os.path.exists(split_file):
        print(f"[ERROR] Data file not found: {split_file}")
        sys.exit(1)
    dataset = SpatialMQADataset(split_file, image_dir=IMAGE_DIR, lazy_load=lazy_load)
    samples = list(dataset)
    if max_samples:
        samples = samples[:max_samples]
    print(f"[Data] Loaded {len(samples)} samples from {split}.jsonl")
    return samples


def infer_split(pipeline: ASTRAPipeline, samples: list, output_file: str, desc: str = "Inference") -> list:
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    results = []
    total_time = 0.0
    for sample in tqdm(samples, desc=desc):
        try:
            result = pipeline.run(sample)
        except Exception as e:
            print(f"\n[ERROR] Sample {sample.get('id', '?')}: {e}")
            result = {
                "id": sample.get("id", 0),
                "question": sample.get("question", ""),
                "options": sample.get("options", []),
                "answer": sample.get("answer", ""),
                "predicted": "", "correct": False, "error": str(e),
            }
        results.append(result)
        total_time += result.get("t_total", 0)

    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    correct = sum(1 for r in results if r.get("correct", False))
    acc = correct / len(results) * 100 if results else 0
    avg = total_time / len(results) if results else 0
    print(f"\n[Result] {correct}/{len(results)} correct = {acc:.2f}%")
    print(f"[Time]   {avg:.2f}s/sample, total {format_time(total_time)}")
    errors = Counter(r.get("error", "") for r in results if r.get("error"))
    if errors:
        print("[Errors] Top failures:")
        for message, count in errors.most_common(3):
            print(f"  {count}x {message}")
        details = [r.get("error_detail") for r in results if r.get("error_detail")]
        if details:
            print("[Errors] Examples:")
            for detail in details[:3]:
                print(f"  - {detail}")
    return results


def _parse_modules_arg(raw: str | None) -> list[int] | None:
    """Parse a comma-separated module list like '1' or '1,2,3'."""
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]

def cmd_eval(args):
    """Ch?y dnh gi: baseline, full ASTRA, ho?c escalation."""
    modules = _parse_modules_arg(args.modules)
    if modules is None and args.baseline:
        modules = []
        tag = "baseline"
    elif modules is None:
        modules = [1, 2, 3]
        tag = "ASTRA_full"
    else:
        tag = "modules_" + "-".join(str(m) for m in modules)

    if args.escalation:
        tag = "escalation"
        modules = [1, 2, 3]

    print(f"[Command] eval | model={args.model} | modules={modules or 'none'} "
          f"| split={args.split} | escalation={args.escalation}")

    pipeline = ASTRAPipeline(
        model_name=args.model,
        device=args.device or get_device(),
        enable_modules=modules,
        n_perms=args.n_perms,
        depth_epsilon=args.depth_epsilon,
        confidence_threshold=args.confidence_threshold,
        load_models=True,
        use_escalation=args.escalation,
    )
    samples = load_split(args.split, args.max_samples)
    results = infer_split(pipeline, samples, args.output, desc=f"{tag} {args.model}")
    metrics = evaluate_predictions(results)
    if metrics:
        print_eval_report(metrics, f"{tag} {args.model}")
        save_metrics(metrics, args.output.replace("results.jsonl", "metrics.json"))
    pipeline.unload()
def cmd_run_all(args):
    """Chạy cả baseline và full ASTRA cho tất cả model + tổng hợp so sánh."""
    models = args.models
    for model_name in models:
        mshort = model_name.replace("Qwen/Qwen3-VL-", "").replace("-Instruct", "")
        print(f"\n{'=' * 60}\n  Model: {mshort}\n{'=' * 60}")

        for tag, modules in [("baseline", []), ("ASTRA_full", [1, 2, 3])]:
            odir = os.path.join(args.output_dir, mshort, tag)
            os.makedirs(odir, exist_ok=True)
            ofile = os.path.join(odir, "results.jsonl")
            print(f"\n  [{tag}]")
            pipeline = ASTRAPipeline(
                model_name=model_name,
                device=args.device or get_device(),
                enable_modules=modules,
                n_perms=args.n_perms,
                depth_epsilon=args.depth_epsilon,
                confidence_threshold=args.confidence_threshold,
                load_models=True,
            )
            samples = load_split(args.split, args.max_samples)
            results = infer_split(pipeline, samples, ofile, desc=f"{mshort} {tag}")
            metrics = evaluate_predictions(results)
            if metrics:
                save_metrics(metrics, os.path.join(odir, "metrics.json"))
            pipeline.unload()
            if args.device != "cpu":
                torch.cuda.empty_cache()

    # So sánh & xuất CSV
    summary = build_ablation_summary(args.output_dir)
    print_ablation_summary(summary)
    csv_path = os.path.join(args.output_dir, "comparison.csv")
    export_ablation_csv(summary, csv_path)
    if args.save_summary:
        with open(args.save_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[Saved] {args.save_summary}")


def cmd_compare(args):
    print(f"\n[Command] compare | results_dir={args.results_dir}")
    summary = build_ablation_summary(args.results_dir)
    print_ablation_summary(summary)
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n[Saved] {args.save}")
    export_ablation_csv(summary, args.save.replace(".json", ".csv") if args.save else "comparison.csv")


def cmd_single(args):
    from PIL import Image
    modules = [int(m) for m in args.modules.split(",")] if args.modules else [1, 2, 3]
    pipeline = ASTRAPipeline(
        model_name=args.model,
        device=args.device or get_device(),
        enable_modules=modules,
        load_models=True,
    )
    image = Image.open(args.image).convert("RGB")
    sample = {
        "image": image,
        "question": args.question,
        "options": args.options.split("|"),
        "answer": ""
    }
    t0 = time.time()
    result = pipeline.run(sample)
    elapsed = time.time() - t0
    print(f"\n[Result]  Predicted: {result.get('predicted', '')}")
    print(f"  O1: {result.get('O1_name', '')} | O2: {result.get('O2_name', '')}")
    print(f"  Depth cue: {result.get('depth_cue', 'N/A')}")
    print(f"  OGM: {result.get('ogm_success', False)} | DLC: {result.get('dlc_success', False)}")
    print(f"  Time: {elapsed:.2f}s")
    pipeline.unload()


def build_parser():
    p = argparse.ArgumentParser(
        description="ASTRA — Auxiliary Spatial Tools for Robust Answering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baseline: model gốc không có module
  python main.py eval --model Qwen3-VL-4B --baseline --split test \\
      --output outputs/astra/4B/baseline/results.jsonl

  # Full ASTRA: M1 (OGM) + M2 (DLC) + M3 (ODV)
  python main.py eval --model Qwen3-VL-4B --split test \\
      --output outputs/astra/4B/ASTRA_full/results.jsonl

  # Escalation: zero-shot first, augment on disagreement
  python main.py eval --model Qwen3-VL-4B --escalation --split test \\
      --output outputs/astra/4B/escalation/results.jsonl

  # Extract O1/O2 via LLM for all test samples
  python scripts/extract_objects.py --split test --max-samples 20

  # Run Module 1 ablation harness
  python scripts/run_module1_ablation.py --max-samples 50

  # Chạy tất cả: cả 3 model × cả baseline và full ASTRA
  python main.py run-all --models Qwen3-VL-2B Qwen3-VL-4B Qwen3-VL-8B \\
      --split test --output-dir outputs/astra

  # So sánh kết quả
  python main.py compare --results-dir outputs/astra --save outputs/astra/summary.json

  # Chạy thử một câu hỏi đơn lẻ
  python main.py single --model Qwen3-VL-4B --image img.jpg \\
      --question "Where is the dog located relative to the cat?" \\
      --options "in front of|behind|left of|right of"
""")
    sub = p.add_subparsers(dest="command", required=True)

    # eval: baseline hoặc full ASTRA
    sp = sub.add_parser("eval")
    sp.add_argument("--model", required=True)
    sp.add_argument("--baseline", action="store_true",
                    help="Chạy model gốc không bật module nào")
    sp.add_argument("--modules", default=None,
                    help="Override modules (v� d?: '1' ho?c '1,2,3'). N?u d? tr?ng th� d�ng full [1,2,3] ho?c baseline khi c� --baseline")
    sp.add_argument("--split", default="test")
    sp.add_argument("--output", required=True)
    sp.add_argument("--max-samples", type=int, default=None)
    sp.add_argument("--device", default=None)
    sp.add_argument("--n-perms", type=int, default=N_PERMS)
    sp.add_argument("--depth-epsilon", type=float, default=DEPTH_EPSILON)
    sp.add_argument("--confidence-threshold", type=float, default=CONFIDENCE_THRESHOLD)
    sp.add_argument("--escalation", action="store_true",
                   help="Bật escalation logic: zero-shot first, augment on disagreement")

    # run-all: chạy baseline + full ASTRA cho nhiều model
    sp = sub.add_parser("run-all")
    sp.add_argument("--models", nargs="+", required=True,
                    help="Danh sách model, ví dụ: Qwen3-VL-2B Qwen3-VL-4B Qwen3-VL-8B")
    sp.add_argument("--split", default="test")
    sp.add_argument("--output-dir", default="outputs/astra/")
    sp.add_argument("--max-samples", type=int, default=None)
    sp.add_argument("--device", default=None)
    sp.add_argument("--n-perms", type=int, default=N_PERMS)
    sp.add_argument("--depth-epsilon", type=float, default=DEPTH_EPSILON)
    sp.add_argument("--confidence-threshold", type=float, default=CONFIDENCE_THRESHOLD)
    sp.add_argument("--save-summary", default=None)

    # compare: so sánh kết quả
    sp = sub.add_parser("compare")
    sp.add_argument("--results-dir", required=True)
    sp.add_argument("--save", default=None)

    # single: chạy một câu hỏi đơn lẻ
    sp = sub.add_parser("single")
    sp.add_argument("--model", required=True)
    sp.add_argument("--image", required=True)
    sp.add_argument("--question", required=True)
    sp.add_argument("--options", required=True)
    sp.add_argument("--modules", default="1,2,3")
    sp.add_argument("--device", default=None)

    return p


def main():
    args = build_parser().parse_args()
    set_seed(42)
    if args.command == "eval":
        cmd_eval(args)
    elif args.command == "run-all":
        cmd_run_all(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "single":
        cmd_single(args)
    else:
        build_parser().print_help()


if __name__ == "__main__":
    main()

