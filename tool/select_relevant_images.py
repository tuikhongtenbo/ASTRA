import json
import os
import glob
import shutil


def _copy_single_image(image_filename, source_dir, destination_dir):
    source_path = os.path.join(source_dir, image_filename)
    if os.path.exists(source_path):
        destination_path = os.path.join(destination_dir, image_filename)
        shutil.copyfile(source_path, destination_path)
    else:
        print(f"Source file {image_filename} not found in {source_dir}")


def copy_images(file_path, source_dir, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    count = 0

    if file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                count += 1
                data = json.loads(line)
                image_filename = data['image']
                _copy_single_image(image_filename, source_dir, destination_dir)
    else:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            data_list = json.load(f)
            if isinstance(data_list, list):
                for item in data_list:
                    count += 1
                    image_filename = item['image']
                    _copy_single_image(image_filename, source_dir, destination_dir)
            elif isinstance(data_list, dict):
                for item in data_list.values():
                    if isinstance(item, dict) and 'image' in item:
                        count += 1
                        image_filename = item['image']
                        _copy_single_image(image_filename, source_dir, destination_dir)

    print(f"{file_path}: {count}")
    print("Copy successful!")


# Lấy các file JSONL/JSON trong dataset/data/
json_files = []
json_files.extend(glob.glob('../dataset/data/*.jsonl'))
json_files.extend(glob.glob('../dataset/data/*.json'))

source_dir = '../dataset/images/COCO2017'
destination_dir = '../dataset/images/relevant_images'

for file in json_files:
    copy_images(file, source_dir, destination_dir)
