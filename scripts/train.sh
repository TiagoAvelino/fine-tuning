#!/usr/bin/env bash
# Launch fine-tuning with a YAML config file.
#
# Usage:
#   ./scripts/train.sh config/qlora_example.yaml
#
# For multi-GPU training:
#   ./scripts/train_multi_gpu.sh config/qlora_example.yaml

set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml>}"

echo "=== Fine-tuning started ==="
echo "Config: ${CONFIG}"
echo "GPUs available: $(nvidia-smi -L 2>/dev/null | wc -l || echo 'none detected')"
echo ""

python -m src.train --config "${CONFIG}"

echo ""
echo "=== Fine-tuning complete ==="
