#!/usr/bin/env bash
set -euo pipefail

DATASET_URL="https://www.kaggle.com/api/v1/datasets/download/lamnmh05/coco2017test"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOWNLOAD_DIR="${REPO_ROOT}/downloads"
ZIP_PATH="${DOWNLOAD_DIR}/coco2017test.zip"
EXTRACT_DIR="${DOWNLOAD_DIR}/coco2017test"
DATASET_DIR="${REPO_ROOT}/dataset/data"

mkdir -p "${DOWNLOAD_DIR}" "${DATASET_DIR}"

echo "[download] ${DATASET_URL}"
curl -L -o "${ZIP_PATH}" "${DATASET_URL}"

rm -rf "${EXTRACT_DIR}"
mkdir -p "${EXTRACT_DIR}"

echo "[unzip] ${ZIP_PATH} -> ${EXTRACT_DIR}"
unzip -q "${ZIP_PATH}" -d "${EXTRACT_DIR}"

SRC_DIR="${EXTRACT_DIR}/test_images/test_images"
if [[ ! -d "${SRC_DIR}" ]]; then
  SRC_DIR="${EXTRACT_DIR}/test_images"
fi

if [[ ! -d "${SRC_DIR}" ]]; then
  echo "[error] Missing expected directory: test_images" >&2
  echo "[hint] Check zip structure with: unzip -l ${ZIP_PATH}" >&2
  exit 1
fi

DST_DIR="${DATASET_DIR}/test_images"
mkdir -p "${DST_DIR}"

echo "[copy] ${SRC_DIR}/ -> ${DST_DIR}/"
if command -v rsync >/dev/null 2>&1; then
  rsync -a "${SRC_DIR}/" "${DST_DIR}/"
else
  cp -a "${SRC_DIR}/." "${DST_DIR}/"
fi

echo "[done] Files are ready:"
echo "  ${DST_DIR}"
