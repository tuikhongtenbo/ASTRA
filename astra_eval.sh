#!/bin/bash
# astra_eval.sh — Quick launcher for ASTRA evaluation
# Calls scripts/eval_2B.sh by default (MODEL=2B/4B/8B)

MODEL="${MODEL:-2B}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$MODEL" = "2B" ]; then
    bash "$SCRIPT_DIR/scripts/eval_2B.sh"
elif [ "$MODEL" = "4B" ]; then
    bash "$SCRIPT_DIR/scripts/eval_4B.sh"
elif [ "$MODEL" = "8B" ]; then
    bash "$SCRIPT_DIR/scripts/eval_8B.sh"
else
    echo "Unknown MODEL: $MODEL. Use 2B, 4B, or 8B."
    exit 1
fi

