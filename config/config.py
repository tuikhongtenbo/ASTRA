"""
configs.py — Configuration cho ASTRA.
Tái sử dụng và mở rộng từ CODA config.
"""

import os


# ======== PATHS ========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_dotenv_file(path: str) -> None:
    """Load simple KEY=VALUE pairs without requiring python-dotenv."""
    if not path or not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        return


def _load_dotenv() -> None:
    seen = set()
    for path in (os.path.join(BASE_DIR, ".env"), os.path.join(os.getcwd(), ".env")):
        if path not in seen:
            seen.add(path)
            _load_dotenv_file(path)


def _get_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default


_load_dotenv()


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

API_MODEL_NAME_2B = "qwen3-vl-2b-instruct"
API_MODEL_NAME_4B = "qwen3-vl-4b-instruct"
API_MODEL_NAME_8B = "qwen3-vl-8b-instruct"

MODEL_ALIASES = {
    "Qwen3-VL-2B": MODEL_NAME_2B,
    "Qwen3-VL-4B": MODEL_NAME_4B,
    "Qwen3-VL-8B": MODEL_NAME_8B,
    "2B": MODEL_NAME_2B,
    "4B": MODEL_NAME_4B,
    "8B": MODEL_NAME_8B,
}

API_MODEL_ALIASES = {
    "Qwen3-VL-2B": API_MODEL_NAME_2B,
    "Qwen3-VL-4B": API_MODEL_NAME_4B,
    "Qwen3-4B-VL": API_MODEL_NAME_4B,
    "Qwen3-VL-8B": API_MODEL_NAME_8B,
    "2B": API_MODEL_NAME_2B,
    "4B": API_MODEL_NAME_4B,
    "8B": API_MODEL_NAME_8B,
}

DEFAULT_MODEL = MODEL_NAME_4B

# ======== INFERENCE ========
MAX_NEW_TOKENS = 128
MAX_IMAGE_SIZE = 1280

# ======== MODULE PARAMETERS ========
# Module 1 - OGM (YOLOE-26X bbox detection)
CONFIDENCE_THRESHOLD = 0.0
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
DASHSCOPE_WORKSPACE_ID = _get_env("DASHSCOPE_WORKSPACE_ID", "WORKSPACE_ID")
DASHSCOPE_REGION = _get_env("DASHSCOPE_REGION", default="ap-southeast-1")


def _default_dashscope_base_url() -> str:
    if DASHSCOPE_WORKSPACE_ID:
        return f"https://{DASHSCOPE_WORKSPACE_ID}.{DASHSCOPE_REGION}.maas.aliyuncs.com/compatible-mode/v1"
    return "https://ws-vhe3s06410otzxtw.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"


DASHSCOPE_BASE_URL = _get_env("DASHSCOPE_BASE_URL", default=_default_dashscope_base_url())
DASHSCOPE_API_KEY = _get_env("DASHSCOPE_API_KEY", "QWEN_API_KEY", "EXTRACTOR_API_KEY")
EXTRACTOR_API_KEY = _get_env("EXTRACTOR_API_KEY", "QWEN_API_KEY", "DASHSCOPE_API_KEY")
EXTRACTOR_MAX_TOKENS = 64
EXTRACTOR_TEMPERATURE = 0.0
EXTRACTOR_MAX_RETRIES = 2

# VLM backend. "local" uses transformers; "dashscope" uses OpenAI-compatible API.
VLM_BACKEND = _get_env("ASTRA_VLM_BACKEND", "VLM_BACKEND", default="local")
VLM_API_KEY = _get_env("VLM_API_KEY", "DASHSCOPE_API_KEY", "QWEN_API_KEY", "EXTRACTOR_API_KEY")
VLM_API_MAX_RETRIES = int(_get_env("VLM_API_MAX_RETRIES", default="2"))
VLM_API_ENABLE_THINKING = _get_env("VLM_API_ENABLE_THINKING", default="false").lower() in (
    "1", "true", "yes", "on",
)

# Object extraction output
EXTRACTION_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "test_objects.json")

# Confidence thresholds
CONF_THRESHOLD_EXTRACT = 0.0
DET_CONF_THRESHOLD = 0.35
DEPTH_SAME_PLANE_EPS = 0.05

# Escalation
ESCALATION_LOG_FILE = os.path.join(OUTPUT_DIR, "escalation_log.jsonl")
