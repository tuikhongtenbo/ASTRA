#!/bin/bash
# eval_4B.sh — Chạy Baseline và Full ASTRA trên Qwen3-VL-4B

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

MODEL="${MODEL:-Qwen3-VL-4B}"
SPLIT="${SPLIT:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/astra/4B}"
DEVICE="${DEVICE:-cuda}"
N_PERMS="${N_PERMS:-3}"
MAX_SAMPLES="${MAX:-}"

VARIANTS=(
    "baseline"
    "ASTRA_full"
)

echo "========================================"
echo "  ASTRA — $MODEL"
echo "========================================"
echo "Split:   $SPLIT"
echo "Output:  $OUTPUT_DIR"
echo "Device:  $DEVICE"
echo "========================================"

MAX_ARG=""
if [ -n "$MAX_SAMPLES" ]; then
    MAX_ARG="--max-samples $MAX_SAMPLES"
fi

for variant in "${VARIANTS[@]}"; do
    echo ""
    echo "[$variant]"

    out_dir="$OUTPUT_DIR/$variant"
    mkdir -p "$out_dir"

    if [ "$variant" = "baseline" ]; then
        python main.py eval \
            --model "$MODEL" \
            --baseline \
            --split "$SPLIT" \
            --output "$out_dir/results.jsonl" \
            --device "$DEVICE" \
            --n-perms "$N_PERMS" \
            $MAX_ARG
    else
        python main.py eval \
            --model "$MODEL" \
            --split "$SPLIT" \
            --output "$out_dir/results.jsonl" \
            --device "$DEVICE" \
            --n-perms "$N_PERMS" \
            $MAX_ARG
    fi

    if [ "$DEVICE" != "cpu" ]; then
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    fi
done

echo ""
echo "========================================"
echo "  Comparing..."
echo "========================================"
python main.py compare --results-dir "$OUTPUT_DIR" --save "$OUTPUT_DIR/summary.json"
echo "Done! Results in: $OUTPUT_DIR"
