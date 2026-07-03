"""
pipeline_config.py — Configuration cho ASTRA v2 pipeline.
Tách riêng thresholds và visual config, không hard-code trong các module.
"""

import os


# ======== PATHS ========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_DIR = os.path.join(BASE_DIR, "dataset", "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DEBUG_DIR = os.path.join(OUTPUT_DIR, "debug")

# Image directory (images live under data/images/, extraction JSON lives under dataset/data/)
_IMAGE_CANDIDATES = [
    os.path.join(BASE_DIR, "data", "images", "relevant_images"),
    os.path.join(BASE_DIR, "data", "images"),
]
for _p in _IMAGE_CANDIDATES:
    if os.path.exists(_p):
        IMAGE_DIR = _p
        break
else:
    IMAGE_DIR = _IMAGE_CANDIDATES[0]

# ======== EXTRACTION INPUT ========
EXTRACTION_FILE = os.path.join(DATA_DIR, "test_objects_last.json")

# ======== MODULE 1 — Confidence Gating & Detection ========
# Confidence gating: bỏ qua M1+M2 nếu extraction confidence thấp
EXTRACT_CONF_THRESHOLD = 0.6

# YOLOE-26X detection settings
DET_CONF_THRESHOLD = 0.35
YOLOE_WEIGHTS = "yoloe-26x-seg.pt"
YOLOE_IMGSZ = 640

# ======== MODULE 2 — Depth ========
# Ngưỡng chênh lệch depth để sinh depth_relation_text
DEPTH_DIFF_THRESHOLD = 0.05

# Depth model size (small=25M params)
DEPTH_MODEL_SIZE = "small"

# Depth colormap (viridis = good perceptual uniformity, works B&W-friendly)
DEPTH_COLORMAP = "viridis"

# ======== VISUAL DRAWING ========
# Bbox colors (R, G, B) — không dùng tên hex string để PIL/OpenCV dùng trực tiếp
MARK_COLOR_O1 = (255, 60, 60)     # đỏ cho [1]
MARK_COLOR_O2 = (30, 90, 255)      # xanh dương cho [2]

# Bbox border width in pixels
BBOX_BORDER_WIDTH = 3

# Font size: auto-scaled by image dimension
BBOX_FONT_SIZE_RATIO = 0.04        # font_size = min(w,h) * ratio
BBOX_MIN_FONT_SIZE = 12
BBOX_MAX_FONT_SIZE = 28

# Colorbar legend size (fraction of image width/height)
COLORBAR_WIDTH_RATIO = 0.25         # chiều rộng colorbar / chiều rộng ảnh
COLORBAR_HEIGHT_PX = 20            # chiều cao colorbar bar cố định
COLORBAR_MARGIN_PX = 10            # khoảng cách từ cạnh ảnh

# ======== MODULE 3 — ODV ========
N_PERMS = 3

# ======== VLM INFERENCE ========
MAX_NEW_TOKENS = 128

# ======== INTERMEDIATE OUTPUT PATHS ========
M1_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "m1_bbox")
M2_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "m2_depth")
VLM_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "vlm_results")

# Output filenames
M1_BBOX_INFO_FILE = os.path.join(M1_OUTPUT_DIR, "bbox_info.json")
M2_DEPTH_INFO_FILE = os.path.join(M2_OUTPUT_DIR, "depth_info.json")
VLM_RESULTS_FILE = os.path.join(VLM_OUTPUT_DIR, "results.jsonl")
