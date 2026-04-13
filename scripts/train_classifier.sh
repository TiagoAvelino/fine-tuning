#!/usr/bin/env bash
# Train the OpenShift issue classifier.
#
# Usage:
#   ./scripts/train_classifier.sh
#
# Override any argument:
#   ./scripts/train_classifier.sh --model-name roberta-base --learning-rate 1e-5
#
# Upload to MinIO:
#   ./scripts/train_classifier.sh --upload-to-s3 --s3-prefix kcs-classifier/v3

set -euo pipefail

echo "=== KCS Classifier Training ==="
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l || echo 'none detected')"
echo ""

python -m src.train_classifier \
    --train-file data/ocp-issues-v2.csv \
    --model-name distilbert-base-uncased \
    --output-dir outputs/kcs-classifier \
    --num-epochs 10 \
    --train-batch-size 16 \
    --eval-batch-size 16 \
    --learning-rate 2e-5 \
    "$@"

echo ""
echo "=== Training complete ==="
echo "Artifacts: outputs/kcs-classifier/"
ls -lh outputs/kcs-classifier/
