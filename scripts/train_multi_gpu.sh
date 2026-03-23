#!/usr/bin/env bash
# Launch distributed fine-tuning across multiple GPUs using accelerate.
#
# Usage:
#   ./scripts/train_multi_gpu.sh config/qlora_example.yaml [NUM_GPUS]

set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml> [num_gpus]}"
NUM_GPUS="${2:-$(nvidia-smi -L 2>/dev/null | wc -l)}"

echo "=== Multi-GPU Fine-tuning ==="
echo "Config: ${CONFIG}"
echo "GPUs: ${NUM_GPUS}"
echo ""

accelerate launch \
    --num_processes="${NUM_GPUS}" \
    --mixed_precision=bf16 \
    -m src.train \
    --config "${CONFIG}"

echo ""
echo "=== Fine-tuning complete ==="
