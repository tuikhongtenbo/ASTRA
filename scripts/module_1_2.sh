#!/usr/bin/env bash
set -euo pipefail

DATASET_URL="https://www.kaggle.com/api/v1/datasets/download/phmthn/qwen3-input"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOWNLOAD_DIR="${REPO_ROOT}/downloads"
ZIP_PATH="${DOWNLOAD_DIR}/qwen3-input.zip"
EXTRACT_DIR="${DOWNLOAD_DIR}/qwen3-input"
DATASET_DIR="${REPO_ROOT}/dataset"

mkdir -p "${DOWNLOAD_DIR}" "${DATASET_DIR}"

echo "[download] ${DATASET_URL}"
curl -L -o "${ZIP_PATH}" "${DATASET_URL}"

rm -rf "${EXTRACT_DIR}"
mkdir -p "${EXTRACT_DIR}"

echo "[unzip] ${ZIP_PATH} -> ${EXTRACT_DIR}"
unzip -q "${ZIP_PATH}" -d "${EXTRACT_DIR}"

copy_dir() {
  local src="$1"
  local dst="$2"

  if [[ ! -d "${src}" ]]; then
    echo "[error] Missing expected directory: ${src}" >&2
    echo "[hint] Check zip structure with: unzip -l ${ZIP_PATH}" >&2
    exit 1
  fi

  mkdir -p "${dst}"
  echo "[copy] ${src}/ -> ${dst}/"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "${src}/" "${dst}/"
  else
    cp -a "${src}/." "${dst}/"
  fi
}

copy_dir "${EXTRACT_DIR}/bbox_original_output/bbox_original_output"          "${DATASET_DIR}/bbox_original_output"
copy_dir "${EXTRACT_DIR}/depth_bbox_output/depth_bbox_output"          "${DATASET_DIR}/depth_bbox_output"

echo "[done] Files are ready:"
echo "  ${DATASET_DIR}/bbox_original_output"
echo "  ${DATASET_DIR}/depth_bbox_output"
