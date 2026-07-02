import json
import os
import shutil


def copy_images(file_path, source_dir, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    images = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('[') or line.endswith(']'):
                    raise ValueError("Có vẻ là file JSON array chứ không phải JSONL")
                data = json.loads(line)
                if 'image' in data:
                    images.append(data['image'])
    except Exception:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'image' in item:
                            images.append(item['image'])
                elif isinstance(data, dict) and 'image' in data:
                    images.append(data['image'])
        except Exception as e:
            print(f"Bỏ qua file {file_path} do không parse được JSON: {e}")
            return

    # Khử trùng lặp ảnh
    images = list(set(images))
    basename = os.path.basename(file_path)
    print(f"[{basename}] Tìm thấy {len(images)} ảnh độc bản cần copy.")

    copied = 0
    not_found = 0
    for image_filename in images:
        source_path = os.path.join(source_dir, image_filename)
        destination_path = os.path.join(destination_dir, image_filename)

        if os.path.exists(destination_path):
            continue

        if os.path.exists(source_path):
            shutil.copyfile(source_path, destination_path)
            copied += 1
        else:
            not_found += 1

    print(f"[{basename}] Đã copy thêm {copied} ảnh mới. (Không tìm thấy {not_found} ảnh trong source)")


# Cấu hình đường dẫn tuyệt đối dựa trên vị trí của file script
TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(TOOL_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
SOURCE_DIR = os.path.join(DATA_DIR, "images", "COCO2017")
DEST_DIR = os.path.join(DATA_DIR, "images", "relevant_images")

print(f"Thư mục nguồn (COCO2017): {SOURCE_DIR}")
print(f"Thư mục đích (relevant_images): {DEST_DIR}")

# Quét tất cả các file .json và .jsonl trong thư mục data/ (không quét thư mục con)
if os.path.exists(DATA_DIR):
    all_files = os.listdir(DATA_DIR)
    json_files = []
    for f in all_files:
        full_path = os.path.join(DATA_DIR, f)
        if (f.endswith('.json') or f.endswith('.jsonl')) and os.path.isfile(full_path):
            json_files.append(full_path)

    print(f"Các file data phát hiện được: {[os.path.basename(p) for p in json_files]}")

    for file in json_files:
        copy_images(file, SOURCE_DIR, DEST_DIR)
    print("Hoàn thành xử lý tất cả các file data!")
else:
    print(f"Thư mục dữ liệu {DATA_DIR} không tồn tại!")
