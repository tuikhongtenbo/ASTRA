#!/bin/bash
# run_all_models.sh — Chạy baseline + full ASTRA trên TẤT CẢ 3 models (2B, 4B, 8B)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

SPLIT="${SPLIT:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/astra}"
DEVICE="${DEVICE:-cuda}"

echo "========================================"
echo "  ASTRA — Run ALL Models"
echo "========================================"
echo "Split:   $SPLIT"
echo "Output:  $OUTPUT_DIR"
echo "Device:  $DEVICE"
echo "========================================"

for size in 2B 4B 8B; do
    echo ""
    echo "########################################"
    echo "  Model: Qwen3-VL-$size"
    echo "########################################"
    bash "$SCRIPT_DIR/eval_${size}.sh"
    sleep 3
done

echo ""
echo "========================================"
echo "  Full Comparison"
echo "========================================"
python main.py compare --results-dir "$OUTPUT_DIR" --save "$OUTPUT_DIR/summary.json"

python -c "
import sys, json
sys.path.insert(0, '$PROJECT_DIR')
from evaluation.evaluator import build_ablation_summary, export_ablation_csv
s = build_ablation_summary('$OUTPUT_DIR')
if s:
    export_ablation_csv(s, '$OUTPUT_DIR/comparison.csv')
    print('CSV: $OUTPUT_DIR/comparison.csv')
"

echo ""
echo "========================================"
echo "  ALL DONE! Results in: $OUTPUT_DIR"
echo "========================================"
