"""
configs.py — Configuration cho ASTRA.
Tái sử dụng và mở rộng từ CODA config.
"""

import os


# ======== PATHS ========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _first_existing_path(*paths: str) -> str:
    for path in paths:
        if path and os.path.exists(path):
            return path
    return paths[0]


_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
_IMAGE_ZIPS = ("test_images.zip", "test2017.zip", "relevant_images.zip", "COCO2017.zip")


def _path_has_image_source(path: str) -> bool:
    if not path or not os.path.exists(path):
        return False
    if os.path.isfile(path):
        lower = path.lower()
        return lower.endswith(_IMAGE_EXTS) or os.path.basename(path) in _IMAGE_ZIPS
    try:
        for name in os.listdir(path):
            lower = name.lower()
            if lower.endswith(_IMAGE_EXTS) or name in _IMAGE_ZIPS:
                return True
    except OSError:
        return False
    return False


def _first_image_source_path(*paths: str) -> str:
    env_path = os.getenv("ASTRA_IMAGE_DIR", "").strip()
    candidates = ([env_path] if env_path else []) + list(paths)
    for path in candidates:
        if _path_has_image_source(path):
            return path
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return paths[0]


# Local data can live either in ASTRA/data or ASTRA/dataset/data depending on the clone.
DATA_DIR = _first_existing_path(
    os.path.join(BASE_DIR, "dataset", "data"),
    os.path.join(BASE_DIR, "data"),
)
JSON_DIR = os.path.join(DATA_DIR, "json")
TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl")
DEV_FILE = os.path.join(DATA_DIR, "dev.jsonl")
TEST_FILE = os.path.join(DATA_DIR, "test.jsonl")
TEST_500_FILE = os.path.join(DATA_DIR, "test_500.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ======== IMAGE PATH ========
# Thử nhiều candidate paths, ưu tiên local
_CANDIDATE_IMAGE_PATHS = [
    os.path.join(DATA_DIR, "test_images"),
    os.path.join(DATA_DIR, "relevant_images"),
    os.path.join(BASE_DIR, "dataset", "data", "test_images"),
    os.path.join(BASE_DIR, "dataset", "images", "test_images"),
    os.path.join(BASE_DIR, "dataset", "images", "COCO2017"),
    os.path.join(BASE_DIR, "dataset", "images", "relevant_images"),
    os.path.join(BASE_DIR, "dataset", "images"),
    os.path.join(BASE_DIR, "data", "test_images"),
    os.path.join(BASE_DIR, "data", "relevant_images"),
    os.path.join(BASE_DIR, "data", "images", "test_images"),
    os.path.join(BASE_DIR, "data", "images", "relevant_images"),
    os.path.join(BASE_DIR, "data", "images"),
    os.path.join(BASE_DIR, "relevant_images"),
    os.path.join(BASE_DIR, "..", "thamkhao", "SpatialMQA", "Dataset", "relevant_images"),
]
IMAGE_DIR = _first_image_source_path(*_CANDIDATE_IMAGE_PATHS)

# ======== MODELS ========
MODEL_NAME_2B = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_NAME_4B = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_NAME_8B = "Qwen/Qwen3-VL-8B-Instruct"

MODEL_ALIASES = {
    "Qwen3-VL-2B": MODEL_NAME_2B,
    "Qwen3-VL-4B": MODEL_NAME_4B,
    "Qwen3-VL-8B": MODEL_NAME_8B,
    "2B": MODEL_NAME_2B,
    "4B": MODEL_NAME_4B,
    "8B": MODEL_NAME_8B,
}

DEFAULT_MODEL = MODEL_NAME_4B

# ======== INFERENCE ========
MAX_NEW_TOKENS = 128
MAX_IMAGE_SIZE = 1280

# ======== MODULE PARAMETERS ========
# Module 1 - OGM (YOLOE-26X bbox detection)
CONFIDENCE_THRESHOLD = 0.3
YOLOE_WEIGHTS = "yoloe-26x-seg.pt"
YOLOE_IMGSZ = 640

# Module 2 — DLC
DEPTH_EPSILON = 0.05
DEPTH_MODEL_SIZE = "small"  # small=25M, base, large

# Module 3 — ODV
N_PERMS = 3

# ======== RELATIONS ========
RELATIONS = ["on/above", "below", "in front of", "behind", "left of", "right of"]
DEPTH_RELATIONS = ["in front of", "behind"]
AXIS_MAP = {
    "y": ["on/above", "below"],
    "z": ["in front of", "behind"],
    "x": ["left of", "right of"],
}

# ======== LLM EXTRACTOR (Module 1 pre-processing) ========
EXTRACTOR_MODEL = "qwen3.7-max"
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://ws-vhe3s06410otzxtw.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
)
EXTRACTOR_API_KEY = os.getenv("QWEN_API_KEY", "")
EXTRACTOR_MAX_TOKENS = 64
EXTRACTOR_TEMPERATURE = 0.0
EXTRACTOR_MAX_RETRIES = 2

# Object extraction output
EXTRACTION_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "test_objects.json")

# Confidence thresholds
CONF_THRESHOLD_EXTRACT = 0.6
DET_CONF_THRESHOLD = 0.35
DEPTH_SAME_PLANE_EPS = 0.05

# Escalation
ESCALATION_LOG_FILE = os.path.join(OUTPUT_DIR, "escalation_log.jsonl")
