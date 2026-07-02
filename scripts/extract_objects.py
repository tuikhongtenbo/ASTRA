"""
extract_objects.py — Standalone CLI: extract O1/O2 via LLM for all samples.
Saves to test_objects.json with resume support.

Usage:
  python scripts/extract_objects.py
  python scripts/extract_objects.py --split dev --output data/dev_objects.json
  python scripts/extract_objects.py --max-samples 50
  python scripts/extract_objects.py --no-resume   # re-extract everything
"""
import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from tqdm import tqdm

from config.config import (
    EXTRACTOR_API_KEY, EXTRACTION_OUTPUT_FILE,
    DATA_DIR, TEST_FILE, DEV_FILE, TRAIN_FILE,
)
from models.extractor import extract_entities_llm, ExtractionResult
from models.extractor.extract import load_existing_extractions, save_extractions


def _resolve_api_key() -> str:
    """Try EXTRACTOR_API_KEY env var first, then .env file."""
    key = os.environ.get("QWEN_API_KEY", "")
    if key:
        return key
    key = os.environ.get("EXTRACTOR_API_KEY", "")
    if key:
        return key
    for _dir in [Path.cwd(), _REPO_ROOT]:
        env = _dir / ".env"
        if env.exists():
            with open(env) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("QWEN_API_KEY") or line.startswith("EXTRACTOR_API_KEY"):
                        parts = line.split("=", 1)
                        if len(parts) == 2:
                            return parts[1].strip().strip("'\"")

    # Try ASTRA-hi root
    for candidate in [_REPO_ROOT.parent.parent / "ASTRA" / ".env",
                     _REPO_ROOT.parent / ".env"]:
        if candidate.exists():
            with open(candidate) as f:
                for line in f:
                    line = line.strip()
                    if "QWEN_API_KEY" in line or "EXTRACTOR_API_KEY" in line:
                        parts = line.split("=", 1)
                        if len(parts) == 2:
                            return parts[1].strip().strip("'\"")
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Extract O1/O2 via LLM for SpatialMQA — saves to test_objects.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python scripts/extract_objects.py --split test
  python scripts/extract_objects.py --split dev --max-samples 20
  python scripts/extract_objects.py --output my_objects.json --no-resume
""")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--output", default=None,
                       help="Output JSON file (default: EXTRACTION_OUTPUT_FILE from config)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Limit number of samples to process")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                       help="Disable resume (re-extract all)")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Save to disk every N samples (default: 50)")
    args = parser.parse_args()

    api_key = _resolve_api_key()
    if not api_key:
        print("[ERROR] QWEN_API_KEY not found. Set env var or .env file.")
        sys.exit(1)

    split_map = {"train": TRAIN_FILE, "dev": DEV_FILE, "test": TEST_FILE}
    split_file = split_map[args.split]
    output_file = args.output or EXTRACTION_OUTPUT_FILE

    # Ensure ASTRA-hi paths work regardless of where script is called from
    if not os.path.isabs(split_file) or not os.path.exists(split_file):
        split_file = os.path.join(_REPO_ROOT, "data", f"{args.split}.jsonl")

    if not os.path.exists(split_file):
        print(f"[ERROR] Data file not found: {split_file}")
        sys.exit(1)

    # Load data
    with open(split_file, "r", encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"[Data] {len(samples)} samples in {args.split}.jsonl")

    # Resume: skip already-extracted
    already_done = set()
    if args.resume and os.path.exists(output_file):
        already_done = set(load_existing_extractions(output_file).keys())
        skipped = sum(1 for s in samples if str(s.get("id", "")) in already_done)
        print(f"[Resume] {skipped}/{len(samples)} already extracted. Skipping.")

    to_process = [s for s in samples if str(s.get("id", "")) not in already_done]
    print(f"[Processing] {len(to_process)} samples")

    results = []
    errors = 0
    for sample in tqdm(to_process, desc="Extracting"):
        qid = str(sample.get("id", sample.get("question_id", 0)))
        question = sample.get("question", "")
        try:
            ext = extract_entities_llm(question)
        except Exception as e:
            print(f"\n[WARN] Sample {qid}: {e}")
            ext = ExtractionResult(
                O1="", O2=None, O2_is_viewer=False,
                confidence=0.0, raw_json="",
            )
            errors += 1

        # Build Object list
        obj_list = [ext.O1]
        if ext.O2:
            obj_list.append(ext.O2)

        results.append({
            "id": qid,
            "question": question,
            "Object": obj_list,
            "O2_is_viewer": ext.O2_is_viewer,
            "confidence": ext.confidence,
            "raw_json": ext.raw_json,
            "O1_hallucinated": ext.O1_hallucinated,
            "O2_hallucinated": ext.O2_hallucinated,
            **sample,
        })

        if len(results) >= args.batch_size:
            save_extractions(results, output_file)
            results = []

    if results:
        save_extractions(results, output_file)

    total_in_file = len(load_existing_extractions(output_file))
    print(f"\n[Saved] {len(to_process) - errors} new extractions to {output_file}")
    print(f"[Done]  {total_in_file} total samples in file")
    if errors:
        print(f"[Warn]  {errors} samples had errors")


if __name__ == "__main__":
    main()
