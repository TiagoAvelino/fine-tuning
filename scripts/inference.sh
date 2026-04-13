#!/usr/bin/env bash
# Run inference against a fine-tuned LLM.
#
# Usage:
#   ./scripts/inference.sh "Pod my-app in CrashLoopBackOff with OOMKilled"
#   ./scripts/inference.sh "Route returns 503" --adapter outputs/ocp-tinyllama-qlora/final
#   ./scripts/inference.sh "PVC stuck Pending" --base-model mistralai/Mistral-7B-Instruct-v0.3

set -euo pipefail

PROMPT="${1:?Usage: $0 <prompt> [--base-model <model>] [--adapter <path>]}"
shift

BASE_MODEL="${BASE_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
ADAPTER="${ADAPTER:-outputs/ocp-tinyllama-qlora/final}"

python -m src.inference \
    --base-model "${BASE_MODEL}" \
    --adapter "${ADAPTER}" \
    --prompt "${PROMPT}" \
    "$@"
