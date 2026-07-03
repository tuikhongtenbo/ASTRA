"""
Utils — Utility functions cho ASTRA.
"""

from __future__ import annotations

import os
import random
import re
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

def find_image_path(image_dir: str, img_name: str) -> Optional[str]:
    if not img_name:
        return None
    if os.path.exists(img_name):
        return os.path.abspath(img_name)
    if os.path.isabs(img_name) and os.path.exists(img_name):
        return img_name
    p = os.path.join(image_dir, img_name)
    if os.path.exists(p):
        return p
    parent = os.path.dirname(image_dir)
    grandparent = os.path.dirname(parent)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
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
        os.path.join(base_dir, "relevant_images"),
        os.path.join(base_dir, "dataset", "images", "relevant_images"),
        os.path.join(base_dir, "dataset", "images", "test_images"),
        os.path.join(base_dir, "dataset", "images", "COCO2017"),
        os.path.join(base_dir, "data", "images"),
        os.path.join(base_dir, "data", "images", "test_images"),
        os.path.join(base_dir, "data", "images", "relevant_images"),
    ]
    for c in candidates:
        p = os.path.join(c, img_name)
        if os.path.exists(p):
            return p
    return None


def load_image(image_path: str):
    try:
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
