"""LLM Fine-Tuning Pipeline — KFP v2.

Downloads the JSONL dataset from S3, normalizes it to the schema expected by the
trainer, validates GPU availability, trains a causal LLM with QLoRA, and uploads
the LoRA adapter to S3/MinIO.

Usage:
    pip install kfp kfp-kubernetes
    python pipeline/llm_training_pipeline.py

This produces pipeline/llm_training_pipeline.yaml which can be
imported into OpenShift AI Data Science Pipelines.
"""

from kfp import dsl, compiler, kubernetes

REGISTRY = "quay.io/rh_ee_tavelino"
PROJECT = "ocp-llm"
TAG = "latest"

S3_SECRET_NAME = "minio-s3-connection"

S3_ENV_MAPPING = {
    "AWS_S3_ENDPOINT": "AWS_S3_ENDPOINT",
    "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
    "AWS_DEFAULT_REGION": "AWS_DEFAULT_REGION",
    "AWS_S3_BUCKET": "AWS_S3_BUCKET",
}


def _image(name: str) -> str:
    return f"{REGISTRY}/{PROJECT}-{name}:{TAG}"


def _inject_s3_secret(task: dsl.PipelineTask) -> dsl.PipelineTask:
    return kubernetes.use_secret_as_env(
        task,
        secret_name=S3_SECRET_NAME,
        secret_key_to_env=S3_ENV_MAPPING,
    )


@dsl.container_component
def download_dataset(
    s3_key: str,
    dataset_file: dsl.Output[dsl.Artifact],
):
    """Download the JSONL dataset from S3."""
    return dsl.ContainerSpec(
        image=_image("upload"),
        command=["sh", "-c"],
        args=[
            "mkdir -p $(dirname \"$1\") && python3 -c '"
            "import boto3,os,sys;"
            "from botocore.client import Config;"
            "out=sys.argv[1];key=sys.argv[2];"
            "os.makedirs(os.path.dirname(out) or \".\",exist_ok=True);"
            "s3=boto3.client(\"s3\","
            "endpoint_url=os.environ[\"AWS_S3_ENDPOINT\"],"
            "aws_access_key_id=os.environ[\"AWS_ACCESS_KEY_ID\"],"
            "aws_secret_access_key=os.environ[\"AWS_SECRET_ACCESS_KEY\"],"
            "config=Config(signature_version=\"s3v4\"));"
            "bucket=os.environ.get(\"AWS_S3_BUCKET\",\"pipelines\");"
            "s3.download_file(bucket,key,out);"
            "print(f\"Downloaded s3://{bucket}/{key} ({os.path.getsize(out)} bytes)\")"
            "' \"$1\" \"$2\"",
            "--",
            dataset_file.path,
            s3_key,
        ],
    )


@dsl.container_component
def prepare_dataset(
    raw_dataset_file: dsl.Input[dsl.Artifact],
    normalized_dataset_file: dsl.Output[dsl.Artifact],
):
    """Normalize the new nested JSONL dataset to the flat instruction/response schema expected by src.train."""
    return dsl.ContainerSpec(
        image=_image("train"),
        command=["sh", "-c"],
        args=[
            r"""
set -e
python3 - <<'PY' "$1" "$2"
import json
import os
import sys

src_path = sys.argv[1]
dst_path = sys.argv[2]

os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)

def to_response_text(final_response: dict) -> str:
    category = final_response.get("category", "")
    sub_cause = final_response.get("sub_cause", "")
    confidence = final_response.get("confidence", "")
    explanation = final_response.get("explanation", "")

    fix_items = final_response.get("fix", []) or []
    commands = final_response.get("commands", []) or []
    verification = final_response.get("verification", "")

    parts = []

    if category:
        parts.append(f"Category: {category}")
    if sub_cause:
        parts.append(f"Sub-cause: {sub_cause}")
    if confidence:
        parts.append(f"Confidence: {confidence}")
    if explanation:
        parts.append(f"Explanation: {explanation}")

    if fix_items:
        parts.append("Fix:")
        parts.extend(f"- {item}" for item in fix_items)

    if commands:
        parts.append("Commands:")
        parts.extend(f"- {cmd}" for cmd in commands)

    if verification:
        parts.append(f"Verification: {verification}")

    return "\n".join(parts).strip()

count_in = 0
count_out = 0
skipped = 0

with open(src_path, "r", encoding="utf-8") as src, open(dst_path, "w", encoding="utf-8") as dst:
    for line_number, line in enumerate(src, start=1):
        line = line.strip()
        if not line:
            continue

        count_in += 1

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"Skipping line {line_number}: invalid JSON ({exc})")
            skipped += 1
            continue

        source_case = record.get("source_case", {}) or {}
        agent_example = record.get("agent_training_example", {}) or {}
        final_response = agent_example.get("final_response", {}) or {}

        instruction = (
            agent_example.get("user_request")
            or source_case.get("instruction")
            or ""
        ).strip()

        response = to_response_text(final_response)

        if not instruction or not response:
            print(
                f"Skipping line {line_number}: missing instruction or response "
                f"(instruction={bool(instruction)}, response={bool(response)})"
            )
            skipped += 1
            continue

        normalized = {
            "instruction": instruction,
            "response": response,
        }
        dst.write(json.dumps(normalized, ensure_ascii=False) + "\n")
        count_out += 1

print(f"Normalized dataset written to {dst_path}")
print(f"Input records: {count_in}")
print(f"Output records: {count_out}")
print(f"Skipped records: {skipped}")
PY
            """,
            "--",
            raw_dataset_file.path,
            normalized_dataset_file.path,
        ],
    )


@dsl.container_component
def download_config(
    s3_key: str,
    config_file: dsl.Output[dsl.Artifact],
):
    """Download the YAML training config from S3."""
    return dsl.ContainerSpec(
        image=_image("upload"),
        command=["sh", "-c"],
        args=[
            "mkdir -p $(dirname \"$1\") && python3 -c '"
            "import boto3,os,sys;"
            "from botocore.client import Config;"
            "out=sys.argv[1];key=sys.argv[2];"
            "os.makedirs(os.path.dirname(out) or \".\",exist_ok=True);"
            "s3=boto3.client(\"s3\","
            "endpoint_url=os.environ[\"AWS_S3_ENDPOINT\"],"
            "aws_access_key_id=os.environ[\"AWS_ACCESS_KEY_ID\"],"
            "aws_secret_access_key=os.environ[\"AWS_SECRET_ACCESS_KEY\"],"
            "config=Config(signature_version=\"s3v4\"));"
            "bucket=os.environ.get(\"AWS_S3_BUCKET\",\"pipelines\");"
            "s3.download_file(bucket,key,out);"
            "print(f\"Downloaded s3://{bucket}/{key} ({os.path.getsize(out)} bytes)\")"
            "' \"$1\" \"$2\"",
            "--",
            config_file.path,
            s3_key,
        ],
    )


@dsl.container_component
def validate_gpu():
    """Fail fast if CUDA/GPU is not available in the runtime."""
    return dsl.ContainerSpec(
        image=_image("train"),
        command=["sh", "-c"],
        args=[
            r"""
set -e
python3 - <<'PY'
import sys
import torch

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Training would run on CPU.")

idx = torch.cuda.current_device()
print("current device:", idx)
print("device name:", torch.cuda.get_device_name(idx))
print("GPU validation passed.")
PY
            """
        ],
    )


@dsl.container_component
def train_llm(
    dataset_file: dsl.Input[dsl.Artifact],
    config_file: dsl.Input[dsl.Artifact],
    base_model: str,
    num_epochs: str,
    batch_size: str,
    learning_rate: str,
    lora_r: str,
    max_seq_length: str,
    model_dir: dsl.Output[dsl.Artifact],
):
    """Train the LLM with QLoRA using the normalized dataset and config."""
    return dsl.ContainerSpec(
        image=_image("train"),
        command=["sh", "-c"],
        args=[
            r"""
set -e
DATASET_FILE="$1"
CONFIG_FILE="$2"
BASE_MODEL="$3"
OUTPUT_DIR="$4"
NUM_EPOCHS="$5"
BATCH_SIZE="$6"
LR="$7"
LORA_R="$8"
MAX_SEQ="$9"

python3 - <<'PY' "${CONFIG_FILE}" "${DATASET_FILE}" "${BASE_MODEL}" "${OUTPUT_DIR}/final" "${NUM_EPOCHS}" "${BATCH_SIZE}" "${LR}" "${LORA_R}" "${MAX_SEQ}"
import yaml
import sys

cfg_path, ds_path, model, out, ep, bs, lr, r, seq = sys.argv[1:]

with open(cfg_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

config["dataset"]["local_path"] = ds_path
config["dataset"]["instruction_field"] = "instruction"
config["dataset"]["response_field"] = "response"
config["model"]["name"] = model
config["training"]["output_dir"] = out
config["training"]["epochs"] = int(ep)
config["training"]["batch_size"] = int(bs)
config["training"]["learning_rate"] = float(lr)
config["lora"]["r"] = int(r)
config["training"]["max_seq_length"] = int(seq)

with open(cfg_path, "w", encoding="utf-8") as f:
    yaml.dump(config, f)

print("Config patched:\n" + yaml.dump(config, sort_keys=False))
PY

python3 - <<'PY'
import torch

print("Pre-training CUDA check")
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available inside training step.")

idx = torch.cuda.current_device()
print("current device:", idx)
print("device name:", torch.cuda.get_device_name(idx))
PY

python -m src.train --config "${CONFIG_FILE}"
            """,
            "--",
            dataset_file.path,
            config_file.path,
            base_model,
            model_dir.path,
            num_epochs,
            batch_size,
            learning_rate,
            lora_r,
            max_seq_length,
        ],
    )


@dsl.container_component
def merge_model(
    model_dir: dsl.Input[dsl.Artifact],
    base_model: str,
    merged_dir: dsl.Output[dsl.Artifact],
):
    """Merge the LoRA adapter into the base model to produce a full standalone model."""
    return dsl.ContainerSpec(
        image=_image("train"),
        command=["sh", "-c"],
        args=[
            "set -e;"
            "ADAPTER_DIR=\"$1/final/final\";"
            "BASE_MODEL=\"$2\";"
            "MERGED_DIR=\"$3\";"
            "python3 -c '"
            "from src.inference import merge_and_export;"
            "import sys;"
            "merge_and_export(sys.argv[1], sys.argv[2], sys.argv[3])"
            "' \"${BASE_MODEL}\" \"${ADAPTER_DIR}\" \"${MERGED_DIR}\"",
            "--",
            model_dir.path,
            base_model,
            merged_dir.path,
        ],
    )


@dsl.container_component
def upload_model(
    model_dir: dsl.Input[dsl.Artifact],
    prefix: str,
):
    """Upload model artifacts to S3/MinIO."""
    return dsl.ContainerSpec(
        image=_image("upload"),
        command=["sh", "-c"],
        args=[
            "python upload_artifacts.py"
            " --local-dir \"$1\""
            " --bucket \"${AWS_S3_BUCKET:-pipelines}\""
            " --prefix \"$2\"",
            "--",
            model_dir.path,
            prefix,
        ],
    )


@dsl.pipeline(
    name="OCP LLM Fine-Tuning",
    description="Fine-tune a causal LLM (TinyLlama/Mistral) with QLoRA on OpenShift troubleshooting data",
)
def llm_training_pipeline(
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    num_epochs: str = "3",
    batch_size: str = "4",
    learning_rate: str = "0.0002",
    lora_r: str = "32",
    max_seq_length: str = "1024",
    s3_dataset_key: str = "data/ocp-instructions.jsonl",
    s3_config_key: str = "config/ocp-instruct.yaml",
    s3_adapter_prefix: str = "ocp-llm/tinyllama-qlora/latest",
    s3_merged_prefix: str = "ocp-llm/tinyllama-merged/latest",
):
    dl_dataset = download_dataset(s3_key=s3_dataset_key)
    _inject_s3_secret(dl_dataset)

    normalized_dataset = prepare_dataset(
        raw_dataset_file=dl_dataset.outputs["dataset_file"],
    )

    dl_config = download_config(s3_key=s3_config_key)
    _inject_s3_secret(dl_config)

    validate_gpu_task = (
        validate_gpu()
        .set_accelerator_type("nvidia.com/gpu")
        .set_accelerator_limit("1")
    )

    train_task = (
        train_llm(
            dataset_file=normalized_dataset.outputs["normalized_dataset_file"],
            config_file=dl_config.outputs["config_file"],
            base_model=base_model,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            lora_r=lora_r,
            max_seq_length=max_seq_length,
        )
        .after(validate_gpu_task)
        .set_accelerator_type("nvidia.com/gpu")
        .set_accelerator_limit("1")
    )

    merge_task = (
        merge_model(
            model_dir=train_task.outputs["model_dir"],
            base_model=base_model,
        )
        .set_accelerator_type("nvidia.com/gpu")
        .set_accelerator_limit("1")
    )

    upload_adapter = upload_model(
        model_dir=train_task.outputs["model_dir"],
        prefix=s3_adapter_prefix,
    )
    _inject_s3_secret(upload_adapter)

    upload_merged = upload_model(
        model_dir=merge_task.outputs["merged_dir"],
        prefix=s3_merged_prefix,
    )
    _inject_s3_secret(upload_merged)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=llm_training_pipeline,
        package_path="llm_training_pipeline.yaml",
    )
    print("Pipeline compiled to llm_training_pipeline.yaml")