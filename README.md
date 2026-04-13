# LLM Fine-Tuning on OpenShift AI

Fine-tune causal large language models using QLoRA and OpenShift AI with NVIDIA GPUs. This repo contains dataset artifacts, configuration templates, training scripts, and utilities for uploading models to MinIO.

## What this repo contains

- `config/` — training YAML configurations
- `data/` — JSONL datasets and training examples
- `pipeline/` — pipeline manifest and component definitions
- `scripts/` — local training and upload helpers
- `src/` — training, model, dataset, inference, and upload logic

## Prerequisites

- Python 3.11+ or compatible Python 3 environment
- `pip` installed
- NVIDIA GPU with CUDA support for local training
- OpenShift AI workbench or MinIO access for S3 storage

## Quick start

Install dependencies:

```bash
cd /home/tavelino/ocp-ai/fine-tuning
pip install -r requirements.txt
```

Train with the default config:

```bash
./scripts/train.sh
```

Train with a specific config:

```bash
./scripts/train.sh config/ocp-instruct.yaml
```

Run multi-GPU training:

```bash
./scripts/train_multi_gpu.sh config/ocp-instruct.yaml
```

## Dataset and config files

The pipeline is designed to load these files from MinIO:

- `config/ocp-instruct.yaml` — fine-tuning config with model, dataset, and training settings
- `data/ocp-instructions.jsonl` — default dataset
- `data/ocp-agent-instructions-final.jsonl` — richer agent-style dataset (recommended)

If you want the pipeline to use the agent dataset, upload that file to MinIO and use `data/ocp-agent-instructions-final.jsonl` as `s3_dataset_key`.

## Upload model to MinIO

After training, upload a model directory to MinIO:

```bash
./scripts/upload_to_minio.sh ./outputs/ocp-tinyllama-qlora/final
```

By default this uploads to bucket `fine-tuning` with prefix `models/<model-dir>`.

## Pipeline integration

This repo includes a pipeline manifest at `pipeline/llm_training_pipeline.yaml` that downloads the config and dataset from MinIO, trains the model, and uploads artifacts back to S3.

Key default keys used by the pipeline:

- `config/ocp-instruct.yaml`
- `data/ocp-instructions.jsonl`
- `data/ocp-agent-instructions-final.jsonl`

## Project structure

```
fine-tuning/
├── config/                  # Training config YAML files
├── data/                    # Training datasets and examples
├── notebooks/               # Experimentation notebooks
├── pipeline/                # KFP pipeline and manifests
├── scripts/                 # Training and upload helper scripts
├── src/                     # Training, data, inference, and upload code
├── requirements.txt
└── README.md
```

## Recommended workflow

1. Prepare or review the dataset in `fine-tuning/data/`.
2. Choose a config file from `fine-tuning/config/`.
3. Train locally with `./scripts/train.sh`.
4. Upload the resulting `final` adapter directory with `./scripts/upload_to_minio.sh`.
5. Use the OpenShift AI workbench or pipeline to run model training and serving.

## Notes

- `config/ocp-instruct.yaml` points to `data/ocp-instructions.jsonl` by default.
- To use the better agent-style dataset, update the config or pipeline to reference `data/ocp-agent-instructions-final.jsonl`.
- The `pipeline/` files are intended for OpenShift AI pipeline execution and can be used with MinIO-backed storage.

## Useful commands

```bash
# Validate the dataset format
python -m src.data --check data/ocp-agent-instructions-final.jsonl

# Show available GPUs
nvidia-smi

# Train with a custom config
./scripts/train.sh config/ocp-instruct-mistral.yaml
```

## Resources

- `fine-tuning/config/ocp-instruct.yaml`
- `fine-tuning/config/ocp-instruct-mistral.yaml`
- `fine-tuning/data/ocp-agent-instructions-final.jsonl`
- `fine-tuning/pipeline/llm_training_pipeline.yaml`
