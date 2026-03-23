# LLM Fine-Tuning with LoRA/QLoRA

Fine-tune large language models (Llama, Mistral, etc.) using Hugging Face Transformers + PEFT on OpenShift AI with NVIDIA GPUs.

## Project Structure

```
fine-tuning/
├── config/                  # Training configurations (YAML)
│   ├── qlora_example.yaml   # QLoRA 4-bit example
│   └── full_finetune_example.yaml
├── data/                    # Training datasets (gitignored)
├── notebooks/               # Jupyter notebooks for experimentation
├── scripts/                 # Shell scripts for launching jobs
│   ├── train.sh             # Single-GPU training
│   ├── train_multi_gpu.sh   # Multi-GPU with accelerate
│   ├── inference.sh         # Run inference
│   └── upload_to_minio.sh   # Upload model to MinIO S3
├── src/                     # Source code
│   ├── data.py              # Dataset loading and preprocessing
│   ├── model.py             # Model loading, quantization, LoRA setup
│   ├── train.py             # Training entrypoint (SFTTrainer)
│   ├── inference.py         # Inference and model merging
│   └── upload.py            # S3/MinIO upload utility
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure training

Copy and edit a config file:

```bash
cp config/qlora_example.yaml config/my_training.yaml
# Edit model name, dataset, hyperparameters, etc.
```

### 3. Run fine-tuning

```bash
# Single GPU (QLoRA)
./scripts/train.sh config/qlora_example.yaml

# Multi-GPU
./scripts/train_multi_gpu.sh config/qlora_example.yaml
```

### 4. Run inference

```bash
python -m src.inference \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --adapter ./outputs/llama-3.1-8b-qlora/final \
    --prompt "Explain quantum computing in simple terms."
```

### 5. Upload to MinIO

The model is automatically uploadable to the MinIO S3 storage configured in the OpenShift AI workbench:

```bash
./scripts/upload_to_minio.sh ./outputs/llama-3.1-8b-qlora/final
```

## Configuration

All training parameters are controlled via YAML config files. Key sections:

| Section    | Description                                           |
|------------|-------------------------------------------------------|
| `model`    | Base model name, attention impl, quantization settings |
| `lora`     | LoRA rank, alpha, dropout, target modules             |
| `dataset`  | Data source (Hub or local), template, max samples     |
| `training` | Epochs, batch size, learning rate, scheduler, etc.    |

## Running on OpenShift AI

This project is designed to run inside the **PyTorch + CUDA** workbench in the `fine-tuning` data science project. The workbench comes pre-configured with:

- NVIDIA GPU access
- MinIO S3 connection (environment variables auto-injected)
- Persistent storage at `/opt/app-root/src/`

Clone this repo into your workbench and start training.
