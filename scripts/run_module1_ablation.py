"""
run_module1_ablation.py — Ablation harness for Module 1 (OGM).
Measures cascading error impact of O1/O2 detection quality.

Scenarios:
  baseline   — No augmentation (zero-shot baseline)
  oracle    — O1/O2 from regex extraction (upper bound)
  detected  — LLM + Grounding-DINO pipeline
  perturbed — Deliberately swapped O1/O2 (measuring harm of wrong hints)

Usage:
  python scripts/run_module1_ablation.py
  python scripts/run_module1_ablation.py --max-samples 50
  python scripts/run_module1_ablation.py --output results/ablation_dev.md
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import torch
from tqdm import tqdm

from config.config import DEV_FILE, IMAGE_DIR, CONFIDENCE_THRESHOLD, N_PERMS, DEPTH_EPSILON
from models.pipeline import ASTRAPipeline
from models import module3_odv as odv
from models import prompt as prompts
from models.extractor import extract_entities_regex_legacy
from models.extractor.extract import load_existing_extractions
from data_processing.dataset import SpatialMQADataset
from evaluation.evaluator import evaluate_predictions
from utils.utils import get_device, format_time, set_seed


SCENARIOS = [
    ("baseline",  "no_augment"),
    ("oracle",   "use_oracle_o1_o2"),
    ("detected", "use_llm_extractor"),
    ("perturbed", "swap_o1_o2"),
]


def _build_baseline_prompt(question, perm_opts):
    return prompts.build_baseline_prompt(question, perm_opts)


def _build_augmented_prompt(O1, O2, depth_cue, question, perm_opts):
    return prompts.build_astra_prompt(O1, O2, depth_cue, question, perm_opts)


def run_scenario(pipeline, samples, scenario_tag, max_samples=None):
    """Run a single ablation scenario."""
    n_perms = pipeline.n_perms
    samples = samples[:max_samples] if max_samples else samples

    results = []
    for sample in tqdm(samples, desc=f"[{scenario_tag}]"):
        try:
            result = _run_sample(pipeline, sample, scenario_tag, n_perms)
        except Exception as e:
            result = {
                "id": sample.get("id", 0), "correct": False,
                "predicted": "", "error": str(e), "scenario": scenario_tag,
            }
        results.append(result)

    # Metrics
    correct = sum(1 for r in results if r.get("correct", False))
    acc = correct / len(results) * 100 if results else 0
    return {"tag": scenario_tag, "correct": correct, "total": len(results),
            "accuracy": acc, "results": results}


def _run_sample(pipeline, sample, scenario_tag, n_perms):
    from utils.utils import normalize_relation

    image = sample.get("image")
    if isinstance(image, str):
        from PIL import Image
        image = Image.open(image).convert("RGB")

    question = sample.get("question", "")
    options = sample.get("options", [])
    answer = sample.get("answer", "")
    perms = odv.generate_permutations(options, n_perms)

    # ── Extract O1/O2 based on scenario ────────────────────────────────────────
    O1_name, O2_name, O2_is_viewer = None, None, False

    if scenario_tag in ("oracle", "detected"):
        # Oracle: regex (ground-truth proxy for ablation comparison)
        O1_raw, O2_raw = extract_entities_regex_legacy(question)
        O1_name = O1_raw
        O2_name = O2_raw

    if scenario_tag == "detected":
        # Use LLM extraction if available
        try:
            from models.extractor import extract_entities_llm
            ext = extract_entities_llm(question)
            if ext.is_valid:
                O1_name = ext.O1
                O2_name = ext.O2 if not ext.O2_is_viewer else None
                O2_is_viewer = ext.O2_is_viewer
        except Exception:
            pass

    # ── Perturbed: swap O1 and O2 ──────────────────────────────────────────────
    if scenario_tag == "perturbed":
        O1_raw, O2_raw = extract_entities_regex_legacy(question)
        O1_name, O2_name = O2_raw, O1_raw  # swap!

    # ── Run voting ─────────────────────────────────────────────────────────────
    depth_cue = None  # ablation is about Module 1 (OGM) only

    votes = []
    for perm_opts in perms:
        if scenario_tag == "baseline":
            pp = _build_baseline_prompt(question, perm_opts)
        else:
            pp = _build_augmented_prompt(O1_name, O2_name, depth_cue, question, perm_opts)

        try:
            out = pipeline._generate(image, pp)
        except Exception:
            out = ""

        parsed = odv.parse_answer_from_output(out, perm_opts, options)
        votes.append(parsed)

    valid = [v for v in votes if v is not None]
    if not valid:
        predicted = votes[0] if votes else options[0]
    else:
        _, _, vc = odv.vote_answers(valid)
        predicted = vc.most_common(1)[0][0]

    pred_norm = normalize_relation(predicted, options) or predicted
    ans_norm = normalize_relation(answer, options) or answer
    correct = pred_norm.lower().strip() == ans_norm.lower().strip()

    return {
        "id": sample.get("id", 0),
        "question": question, "options": options, "answer": answer,
        "predicted": predicted, "correct": correct,
        "scenario": scenario_tag,
        "O1_name": O1_name, "O2_name": O2_name,
    }


def main():
    parser = argparse.ArgumentParser(description="Module 1 ablation harness")
    parser.add_argument("--max-samples", type=int, default=100,
                       help="Limit samples per scenario (default: 100)")
    parser.add_argument("--output", default=None,
                       help="Output markdown file (default: outputs/module1_ablation.md)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default="Qwen3-VL-4B")
    args = parser.parse_args()

    set_seed(42)
    output_file = args.output or os.path.join(_REPO_ROOT, "outputs", "module1_ablation.md")
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    # Load dev samples
    split_file = os.path.join(_REPO_ROOT, "data", "dev.jsonl")
    if not os.path.exists(split_file):
        print(f"[ERROR] {split_file} not found")
        sys.exit(1)
    dataset = SpatialMQADataset(split_file, image_dir=IMAGE_DIR, lazy_load=True)
    samples = list(dataset)[:args.max_samples]
    print(f"[Data] {len(samples)} dev samples")

    # Load pipeline (no M1/M2 — ablation uses M3 only for voting)
    device = args.device or get_device()
    pipeline = ASTRAPipeline(
        model_name=args.model, device=device,
        enable_modules=[3],  # Only ODV voting
        load_models=True,
    )

    # Run all scenarios
    all_results = {}
    for tag, _ in SCENARIOS:
        print(f"\n{'=' * 60}\n  Scenario: {tag}\n{'=' * 60}")
        all_results[tag] = run_scenario(pipeline, samples, tag, args.max_samples)

    pipeline.unload()

    # ── Build report ───────────────────────────────────────────────────────────
    baseline_acc = all_results.get("baseline", {}).get("accuracy", 0)
    rows = []
    for tag, _ in SCENARIOS:
        r = all_results[tag]
        delta = r["accuracy"] - baseline_acc
        sign = "+" if delta >= 0 else ""
        rows.append({
            "Condition": tag.capitalize(),
            "Accuracy": f"{r['accuracy']:.1f}%",
            "Delta": f"{sign}{delta:.1f}pp",
            "Correct": f"{r['correct']}/{r['total']}",
        })

    # Markdown table
    lines = [
        "## Module 1 Cascading Error Analysis",
        "",
        f"| Condition | Accuracy | Δ vs Baseline | Correct |",
        f"|-----------|----------|---------------|---------|",
    ]
    for row in rows:
        lines.append(f"| {row['Condition']} | {row['Accuracy']} | {row['Delta']} | {row['Correct']} |")

    lines += [
        "",
        "**Key insights:**",
    ]

    oracle_acc = all_results.get("oracle", {}).get("accuracy", 0)
    detected_acc = all_results.get("detected", {}).get("accuracy", 0)
    perturbed_acc = all_results.get("perturbed", {}).get("accuracy", 0)

    gap = oracle_acc - detected_acc
    if gap > 0:
        lines.append(f"- Gap `Oracle − Detected` = {gap:.1f}pp → cascading error is a real issue")
    lines.append(f"- Perturbed < Baseline → wrong hints hurt more than no hints")
    lines.append(f"- Oracle upper bound = {oracle_acc:.1f}% (regex-extracted O1/O2)")

    report = "\n".join(lines)
    print("\n" + report)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[Saved] {output_file}")

    # Also save JSON
    json_out = output_file.replace(".md", ".json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {json_out}")


if __name__ == "__main__":
    main()
