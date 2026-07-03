"""
Dataset — Dataset loading utilities cho ASTRA.
"""

from __future__ import annotations

import json
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from config.config import DATA_DIR, IMAGE_DIR
from utils.utils import find_image_path


class SpatialMQADataset(Dataset):
    """Dataset for SpatialMQA."""

    def __init__(self, data_file: str, image_dir: str = IMAGE_DIR, lazy_load: bool = True):
        self.data_file = data_file
        self.image_dir = image_dir
        self.lazy_load = lazy_load
        self.samples = self._load(data_file)

    def _load(self, path: str):
        data = []
        if not os.path.exists(path):
            return data
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                img_name = obj.get('image', '')
                img_path = find_image_path(self.image_dir, img_name)
                obj['image_name'] = img_name
                obj['image_path'] = img_path

                if img_path and not self.lazy_load:
                    try:
                        obj['image'] = Image.open(img_path).convert('RGB')
                    except Exception:
                        obj['image'] = img_path
                else:
                    obj['image'] = img_path or img_name

                data.append(obj)
        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "id": s.get('id', idx),
            "image": s.get('image'),
            "image_name": s.get('image_name'),
            "image_path": s.get('image_path'),
            "question": s['question'],
            "options": s['options'],
            "answer": s['answer'],
        }


def custom_collate_fn(batch):
    return batch


def get_dataloader(
    data_file: str,
    image_dir: str = IMAGE_DIR,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    lazy_load: bool = True,
) -> DataLoader:
    if not os.path.exists(data_file):
        data_file = os.path.join(DATA_DIR, data_file)

    dataset = SpatialMQADataset(data_file, image_dir, lazy_load=lazy_load)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
    )


def get_data_files():
    return {
        "train": TRAIN_FILE,
        "dev": DEV_FILE,
        "test": TEST_FILE,
        "test_500": TEST_500_FILE,
    }
