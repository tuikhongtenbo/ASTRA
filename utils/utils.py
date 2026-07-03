"""
Utils — Utility functions cho ASTRA.
"""

from __future__ import annotations

import os
import random
import re
import zipfile
from functools import lru_cache
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image

from config.config import RELATIONS, AXIS_MAP


# ---------------------------------------------------------------------------
# Relation normalization
# ---------------------------------------------------------------------------

RELATION_KEYWORDS = {
    "on/above": ["on", "above", "on top of"],
    "below": ["below", "beneath", "under"],
    "in front of": ["front", "in front", "infront"],
    "behind": ["behind", "back", "in back"],
    "left of": ["left"],
    "right of": ["right"],
}


def normalize_relation(text: str, options: list) -> Optional[str]:
    if not text:
        return None
    text_lower = text.lower()

    ans_match = re.search(r'answer:\s*[\(\[]?\s*([A-F])\s*[\)\]]?(?=\s|$|[:.,;-])', text, re.IGNORECASE)
    if ans_match:
        letter = ans_match.group(1).upper()
        idx = ord(letter) - ord('A')
        if 0 <= idx < len(options):
            return options[idx]

    ans_text_match = re.search(r'answer:\s*(.*)', text_lower)
    if ans_text_match:
        ans_portion = ans_text_match.group(1)
        for opt in options:
            if opt.lower() in ans_portion:
                return opt

    for opt in options:
        if opt.lower() in text_lower:
            return opt

    for rel, keywords in RELATION_KEYWORDS.items():
        if rel in options:
            for kw in keywords:
                if kw in text_lower:
                    return rel

    return None


def parse_think_trace(output_text: str) -> tuple:
    m = re.search(r'<think>(.*?)<\/think>', output_text, re.DOTALL)
    if m:
        return m.group(1).strip(), output_text.replace(m.group(0), "").strip()
    return "", output_text.strip()


def get_relation_axis(relation: str) -> str:
    for axis, rels in AXIS_MAP.items():
        if relation in rels:
            return axis
    return "unknown"


def map_option_letter(letter: str, options: list) -> Optional[str]:
    if not letter or not options:
        return None
    letter = letter.strip().upper().strip("()[]{}")
    if len(letter) == 1 and letter.isalpha():
        idx = ord(letter) - ord('A')
        if 0 <= idx < len(options):
            return options[idx]
    return None


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
_IMAGE_ZIP_NAMES = ("test_images.zip", "test2017.zip", "relevant_images.zip", "COCO2017.zip")
_ZIP_PREFIX = "zip://"


def _repo_base_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _candidate_image_dirs(image_dir: str) -> list[str]:
    image_dir = image_dir or ""
    parent = os.path.dirname(image_dir)
    grandparent = os.path.dirname(parent)
    base_dir = _repo_base_dir()
    candidates = [
        image_dir,
        os.path.join(image_dir, "relevant_images"),
        os.path.join(image_dir, "test_images"),
        os.path.join(image_dir, "COCO2017"),
        os.path.join(parent, "relevant_images"),
        os.path.join(parent, "test_images"),
        os.path.join(parent, "COCO2017"),
        os.path.join(grandparent, "relevant_images"),
        os.path.join(grandparent, "test_images"),
        os.path.join(grandparent, "COCO2017"),
        "relevant_images",
        "dataset/images/relevant_images",
        "dataset/images/test_images",
        "dataset/images/COCO2017",
        "dataset/data/test_images",
        "dataset/data/relevant_images",
        os.path.join(base_dir, "relevant_images"),
        os.path.join(base_dir, "dataset", "data", "test_images"),
        os.path.join(base_dir, "dataset", "data", "relevant_images"),
        os.path.join(base_dir, "dataset", "images"),
        os.path.join(base_dir, "dataset", "images", "relevant_images"),
        os.path.join(base_dir, "dataset", "images", "test_images"),
        os.path.join(base_dir, "dataset", "images", "COCO2017"),
        os.path.join(base_dir, "data", "test_images"),
        os.path.join(base_dir, "data", "relevant_images"),
        os.path.join(base_dir, "data", "images"),
        os.path.join(base_dir, "data", "images", "test_images"),
        os.path.join(base_dir, "data", "images", "relevant_images"),
    ]
    seen = set()
    unique = []
    for path in candidates:
        if path and path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


@lru_cache(maxsize=16)
def _zip_member_index(zip_path: str) -> dict[str, str]:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            index = {}
            for member in zf.namelist():
                if member.endswith("/"):
                    continue
                name = os.path.basename(member)
                if name.lower().endswith(_IMAGE_EXTS):
                    index.setdefault(name, member)
            return index
    except Exception:
        return {}


@lru_cache(maxsize=32)
def _image_zip_paths(image_dir: str) -> tuple[str, ...]:
    roots = _candidate_image_dirs(image_dir)
    zips = []
    seen = set()

    for root in roots:
        if root.lower().endswith(".zip") and os.path.exists(root):
            path = os.path.abspath(root)
            if path not in seen:
                seen.add(path)
                zips.append(path)
            continue
        for zip_name in _IMAGE_ZIP_NAMES:
            path = os.path.join(root, zip_name)
            if os.path.exists(path):
                path = os.path.abspath(path)
                if path not in seen:
                    seen.add(path)
                    zips.append(path)

    kaggle_input = "/kaggle/input"
    if os.path.isdir(kaggle_input):
        for dirpath, _, filenames in os.walk(kaggle_input):
            for filename in filenames:
                if filename in _IMAGE_ZIP_NAMES:
                    path = os.path.abspath(os.path.join(dirpath, filename))
                    if path not in seen:
                        seen.add(path)
                        zips.append(path)
    return tuple(zips)


@lru_cache(maxsize=1)
def _kaggle_image_index() -> dict[str, str]:
    root = "/kaggle/input"
    if not os.path.isdir(root):
        return {}
    index = {}
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith(_IMAGE_EXTS):
                index.setdefault(filename, os.path.join(dirpath, filename))
    return index


def _zip_uri(zip_path: str, member: str) -> str:
    return f"{_ZIP_PREFIX}{zip_path}!{member}"


def _split_zip_uri(path: str) -> tuple[str, str]:
    spec = path[len(_ZIP_PREFIX):]
    zip_path, member = spec.rsplit("!", 1)
    return zip_path, member


def find_image_path(image_dir: str, img_name: str) -> Optional[str]:
    if not img_name:
        return None
    if isinstance(img_name, str) and img_name.startswith(_ZIP_PREFIX):
        return img_name
    if os.path.exists(img_name):
        return os.path.abspath(img_name)
    if os.path.isabs(img_name) and os.path.exists(img_name):
        return img_name

    for directory in _candidate_image_dirs(image_dir):
        path = os.path.join(directory, img_name)
        if os.path.exists(path):
            return os.path.abspath(path)

    basename = os.path.basename(img_name)
    for zip_path in _image_zip_paths(image_dir or ""):
        member = _zip_member_index(zip_path).get(basename)
        if member:
            return _zip_uri(zip_path, member)

    indexed = _kaggle_image_index().get(basename)
    if indexed:
        return indexed
    return None


def load_image(image_path: str):
    try:
        if isinstance(image_path, str) and image_path.startswith(_ZIP_PREFIX):
            zip_path, member = _split_zip_uri(image_path)
            with zipfile.ZipFile(zip_path) as zf:
                with zf.open(member) as f:
                    return Image.open(BytesIO(f.read())).convert("RGB")
        return Image.open(image_path).convert("RGB")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def count_lines(file_path: str) -> int:
    if not os.path.exists(file_path):
        return 0
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m {s}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"


def aggregate_ablation_results(results_dir: str) -> dict:
    import json
    summary = {}
    if not os.path.exists(results_dir):
        return summary
    for variant in os.listdir(results_dir):
        variant_path = os.path.join(results_dir, variant)
        if not os.path.isdir(variant_path):
            continue
        results_file = os.path.join(variant_path, "last_results.json")
        if os.path.exists(results_file):
            with open(results_file, 'r', encoding='utf-8') as f:
                summary[variant] = json.load(f)
    return summary


def print_ablation_table(results: dict):
    if not results:
        print("No results found.")
        return
    header = (f"{'Variant':<20} {'Overall':>10} {'Depth(Z)':>10} "
              f"{'Horiz(X)':>10} {'Vert(Y)':>10} {'BSTF':>8}")
    print()
    print(header)
    print("-" * len(header))
    for variant, metrics in sorted(results.items()):
        overall = metrics.get("overall_acc", 0.0)
        depth_z = metrics.get("per_axis", {}).get("z", 0.0)
        horiz_x = metrics.get("per_axis", {}).get("x", 0.0)
        vert_y = metrics.get("per_axis", {}).get("y", 0.0)
        bstf = metrics.get("bstf_rate", 0.0)
        print(f"{variant:<20} {overall:>9.1%} {depth_z:>9.1%} "
              f"{horiz_x:>9.1%} {vert_y:>9.1%} {bstf:>7.1%}")
    print()
