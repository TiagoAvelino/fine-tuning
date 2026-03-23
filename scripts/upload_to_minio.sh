#!/usr/bin/env bash
# Upload a fine-tuned model to MinIO S3 storage.
#
# Prerequisites: AWS_S3_ENDPOINT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
# must be set (automatically injected in OpenShift AI workbenches).
#
# Usage:
#   ./scripts/upload_to_minio.sh <model_dir> [bucket] [prefix]

set -euo pipefail

MODEL_DIR="${1:?Usage: $0 <model_dir> [bucket] [prefix]}"
BUCKET="${2:-fine-tuning}"
PREFIX="${3:-models/$(basename "${MODEL_DIR}")}"

echo "=== Uploading model to MinIO ==="
echo "Source: ${MODEL_DIR}"
echo "Dest:   s3://${BUCKET}/${PREFIX}"
echo ""

python -m src.upload \
    --model-dir "${MODEL_DIR}" \
    --bucket "${BUCKET}" \
    --prefix "${PREFIX}"
