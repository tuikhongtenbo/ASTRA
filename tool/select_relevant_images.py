"""
select_relevant_images.py — Copy ảnh cần dùng từ COCO2017 vào relevant_images.

Cách chạy (từ thư mục ASTRA/):
    mkdir -p data/images/relevant_images
    cd tool
    python select_relevant_images.py

Cấu trúc thư mục kỳ vọng:
    ASTRA/
    ├── data/
    │   ├── train.jsonl
    │   ├── dev.jsonl
    │   ├── test.jsonl
    │   ├── test_500.jsonl
    │   ├── test_objects_last.json
    │   └── images/
    │       ├── COCO2017/         ← ảnh COCO gốc (nguồn)
    │       └── relevant_images/  ← ảnh đã lọc (đích)
"""

import json
import os
import shutil

# ─── Đường dẫn tính từ vị trí file script ───────────────────────────────────
TOOL_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR  = os.path.dirname(TOOL_DIR)          # ASTRA/
DATA_DIR  = os.path.join(BASE_DIR, "data")
SOURCE_DIR = os.path.join(DATA_DIR, "images", "COCO2017")
DEST_DIR  = os.path.join(DATA_DIR, "images", "relevant_images")


def collect_images_from_file(file_path: str) -> list[str]:
    """
    Đọc danh sách tên ảnh từ một file JSON hoặc JSONL.
    Trả về list tên file ảnh (ví dụ: '000000024001.jpg').
    """
    images = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().strip()

    if not content:
        return images

    # ── Thử JSON array / object trước (test_objects_last.json) ──
    if content.startswith("[") or content.startswith("{"):
        try:
            data = json.loads(content)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "image" in item:
                        images.append(item["image"])
            elif isinstance(data, dict) and "image" in data:
                images.append(data["image"])
            return images
        except json.JSONDecodeError:
            pass

    # ── Fallback: JSONL (mỗi dòng là một JSON object) ──
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "image" in obj:
                images.append(obj["image"])
        except json.JSONDecodeError:
            continue  # bỏ qua dòng không parse được

    return images


def copy_images(source_dir: str, dest_dir: str, image_names: list[str]) -> tuple[int, int]:
    """Copy ảnh từ source_dir sang dest_dir. Trả về (copied, not_found)."""
    os.makedirs(dest_dir, exist_ok=True)
    copied = 0
    not_found = 0
    for img in image_names:
        src = os.path.join(source_dir, img)
        dst = os.path.join(dest_dir, img)
        if os.path.exists(dst):
            continue  # đã có, bỏ qua
        if os.path.exists(src):
            shutil.copyfile(src, dst)
            copied += 1
        else:
            not_found += 1
    return copied, not_found


def main():
    print(f"[Paths]")
    print(f"  Nguồn (COCO2017)       : {SOURCE_DIR}")
    print(f"  Đích  (relevant_images): {DEST_DIR}")

    if not os.path.isdir(SOURCE_DIR):
        print(f"\n[ERROR] Không tìm thấy thư mục nguồn: {SOURCE_DIR}")
        print("  Hãy tải COCO2017 vào đúng vị trí trên.")
        return

    # ── Quét tất cả file .jsonl và .json trong data/ ──
    json_files = sorted([
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if (f.endswith(".jsonl") or f.endswith(".json")) and os.path.isfile(os.path.join(DATA_DIR, f))
    ])

    if not json_files:
        print(f"\n[ERROR] Không tìm thấy file .jsonl/.json nào trong {DATA_DIR}")
        return

    print(f"\n[Files] Phát hiện {len(json_files)} file data:")
    for fp in json_files:
        print(f"  - {os.path.basename(fp)}")

    # ── Gom danh sách ảnh từ tất cả file, khử trùng ──
    all_images: set[str] = set()
    for fp in json_files:
        imgs = collect_images_from_file(fp)
        print(f"  [{os.path.basename(fp)}] → {len(imgs)} ảnh")
        all_images.update(imgs)

    print(f"\n[Total] {len(all_images)} ảnh độc bản cần copy.")

    # ── Copy ──
    copied, not_found = copy_images(SOURCE_DIR, DEST_DIR, list(all_images))
    print(f"[Done]  Đã copy: {copied}  |  Không tìm thấy trong COCO2017: {not_found}")
    print(f"        Thư mục đích: {DEST_DIR}")


if __name__ == "__main__":
    main()
