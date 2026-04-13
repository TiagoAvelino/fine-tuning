#!/usr/bin/env bash
# Build and push all KFP component images.
#
# Usage:
#   ./scripts/build-images.sh                          # build only
#   ./scripts/build-images.sh --push                   # build + push
#   REGISTRY=quay.io/myuser ./scripts/build-images.sh  # custom registry
#
# Build only LLM images:
#   PROJECT=ocp-llm ./scripts/build-images.sh --push
#
# For OpenShift internal registry:
#   REGISTRY=image-registry.openshift-image-registry.svc:5000/fine-tuning \
#   ./scripts/build-images.sh --push

set -euo pipefail

REGISTRY="${REGISTRY:-quay.io/rh_ee_tavelino}"
TAG="${TAG:-latest}"
PUSH="${1:-}"
PROJECT_FILTER="${PROJECT:-all}"

CLASSIFIER_IMAGES=(
    "kcs-classifier:prepare-dataset:docker/Dockerfile.prepare-dataset"
    "kcs-classifier:train:docker/Dockerfile.train"
    "kcs-classifier:evaluate:docker/Dockerfile.evaluate"
    "kcs-classifier:upload-artifacts:docker/Dockerfile.upload-artifacts"
)

LLM_IMAGES=(
    "ocp-llm:train:docker/Dockerfile.llm-train"
    "ocp-llm:evaluate:docker/Dockerfile.llm-evaluate"
    "ocp-llm:upload:docker/Dockerfile.llm-upload"
)

build_images() {
    local -n images=$1
    for entry in "${images[@]}"; do
        IFS=':' read -r proj name dockerfile <<< "${entry}"
        FULL_IMAGE="${REGISTRY}/${proj}-${name}:${TAG}"

        echo "--- Building: ${FULL_IMAGE}"
        podman build -t "${FULL_IMAGE}" -f "${dockerfile}" .
        echo ""

        if [[ "${PUSH}" == "--push" ]]; then
            echo "--- Pushing: ${FULL_IMAGE}"
            podman push "${FULL_IMAGE}"
            echo ""
        fi
    done
}

echo "============================================"
echo "  KFP Pipeline Images — Build"
echo "============================================"
echo "Registry: ${REGISTRY}"
echo "Tag:      ${TAG}"
echo "Filter:   ${PROJECT_FILTER}"
echo ""

if [[ "${PROJECT_FILTER}" == "all" || "${PROJECT_FILTER}" == "kcs-classifier" ]]; then
    echo "=== Classifier images ==="
    build_images CLASSIFIER_IMAGES
fi

if [[ "${PROJECT_FILTER}" == "all" || "${PROJECT_FILTER}" == "ocp-llm" ]]; then
    echo "=== LLM images ==="
    build_images LLM_IMAGES
fi

echo "============================================"
echo "  Build complete"
echo "============================================"

if [[ "${PUSH}" != "--push" ]]; then
    echo ""
    echo "To push: $0 --push"
fi
