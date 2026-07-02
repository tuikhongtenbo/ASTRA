import json
import os
import glob
import shutil


def copy_images(jsonl_file, source_dir, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    count = 0
    with open(jsonl_file, 'r', encoding='gbk', errors='ignore') as f:
        for line in f:
            count += 1
            data = json.loads(line)
            image_filename = data['image']

            source_path = os.path.join(source_dir, image_filename)

            if os.path.exists(source_path):
                destination_path = os.path.join(destination_dir, image_filename)
                shutil.copyfile(source_path, destination_path)
            else:
                print(f"Source file {image_filename} not found in {source_dir}")

    print(jsonl_file+': '+str(count))
    print("Copy successful!")


# Lấy các file JSONL/JSON trong dataset/data/
json_files = []
json_files.extend(glob.glob('../dataset/data/*.jsonl'))
json_files.extend(glob.glob('../dataset/data/*.json'))

source_dir = '../data/images/COCO2017'
destination_dir = '../data/images/relevant_images'

for file in json_files:
    copy_images(file, source_dir, destination_dir)
