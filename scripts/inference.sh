#!/usr/bin/env bash
# Run inference against a fine-tuned model.
#
# Usage:
#   ./scripts/inference.sh <base_model> <adapter_path> "Your prompt here"

set -euo pipefail

BASE_MODEL="${1:?Usage: $0 <base_model> <adapter_path> <prompt>}"
ADAPTER="${2:?Provide adapter path}"
PROMPT="${3:?Provide a prompt}"

python -m src.inference \
    --base-model "${BASE_MODEL}" \
    --adapter "${ADAPTER}" \
    --prompt "${PROMPT}"
