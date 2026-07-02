"""
convert_extraction.py
Chuyển đổi test_objects.json (dict-of-dicts) thành test_objects_last.json
với cấu trúc:
  - Giữ nguyên tất cả cột gốc từ test.jsonl (image, question, options, answer, id)
  - Bổ sung cột "Object": {"O1": "...", "O2": "..."}
  - Bổ sung "O2_is_viewer": bool
  - Bổ sung metadata từ extraction (confidence, raw_json, O1_hallucinated, O2_hallucinated)

Usage:
  python scripts/convert_extraction.py
  python scripts/convert_extraction.py --extraction output/test_objects.json --output output/test_objects_last.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extraction",
        default=str(_REPO_ROOT / "output" / "test_objects.json"),
        help="Đường dẫn file extraction (test_objects.json)"
    )
    parser.add_argument(
        "--input",
        default=str(_REPO_ROOT / "data" / "test.jsonl"),
        help="Đường dẫn file test.jsonl gốc"
    )
    parser.add_argument(
        "--output",
        default=str(_REPO_ROOT / "output" / "test_objects_last.json"),
        help="Đường dẫn file output"
    )
    args = parser.parse_args()

    # 1. Load extraction results
    if not os.path.exists(args.extraction):
        print(f"[ERROR] Không tìm thấy file extraction: {args.extraction}")
        return

    with open(args.extraction, "r", encoding="utf-8") as f:
        extraction = json.load(f)  # dict-of-dicts: {id_str: {fields...}}

    print(f"[Load] {len(extraction)} samples từ extraction file")

    # 2. Load original test.jsonl
    if not os.path.exists(args.input):
        print(f"[ERROR] Không tìm thấy file test.jsonl: {args.input}")
        return

    with open(args.input, "r", encoding="utf-8") as f:
        original = [json.loads(line) for line in f if line.strip()]

    print(f"[Load] {len(original)} samples từ test.jsonl")

    # 3. Merge: giữ nguyên cột gốc + bổ sung Object, O2_is_viewer, extraction metadata
    results = []
    missing_ids = []

    for sample in original:
        sid = str(sample.get("id", ""))

        # Lấy extraction tương ứng
        ext = extraction.get(sid, {})

        # Build Object structured as {O1, O2}
        obj_list = ext.get("Object", [])
        if isinstance(obj_list, list) and len(obj_list) >= 2:
            O1_val = obj_list[0] if len(obj_list) > 0 else ""
            O2_val = obj_list[1] if len(obj_list) > 1 else ""
        elif isinstance(obj_list, list) and len(obj_list) == 1:
            O1_val = obj_list[0]
            O2_val = ""
        else:
            O1_val = ""
            O2_val = ""

        # Build merged row
        merged = {
            # Cột gốc từ test.jsonl
            "id": sample.get("id"),
            "image": sample.get("image"),
            "question": sample.get("question"),
            "options": sample.get("options"),
            "answer": sample.get("answer"),
            # Cấu trúc Object mới
            "Object": {
                "O1": O1_val,
                "O2": O2_val,
            },
            "O2_is_viewer": bool(ext.get("O2_is_viewer", False)),
            # Metadata từ extraction
            "confidence": ext.get("confidence", 0.0),
            "O1_hallucinated": bool(ext.get("O1_hallucinated", False)),
            "O2_hallucinated": bool(ext.get("O2_hallucinated", False)),
            "raw_json": ext.get("raw_json", ""),
        }

        results.append(merged)

        # Check missing extraction
        if sid not in extraction:
            missing_ids.append(sid)

    # 4. Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[Save] Đã lưu {len(results)} samples vào {args.output}")
    if missing_ids:
        print(f"[Warn] {len(missing_ids)} samples không có trong extraction file: {missing_ids[:10]}...")


if __name__ == "__main__":
    main()
