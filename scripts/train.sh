#!/usr/bin/env bash
# Launch LLM fine-tuning with a YAML config file.
#
# Usage:
#   ./scripts/train.sh                                # default: TinyLlama QLoRA
#   ./scripts/train.sh config/ocp-instruct.yaml       # explicit config
#   ./scripts/train.sh config/ocp-instruct-mistral.yaml

set -euo pipefail

CONFIG="${1:-config/ocp-instruct.yaml}"

echo "=== LLM Fine-tuning started ==="
echo "Config: ${CONFIG}"
echo "GPUs available: $(nvidia-smi -L 2>/dev/null | wc -l || echo 'none detected')"
echo ""

python -m src.train --config "${CONFIG}"

echo ""
echo "=== Fine-tuning complete ==="
