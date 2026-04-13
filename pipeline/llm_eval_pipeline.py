"""LLM Evaluation Pipeline — KFP v2.

Downloads the dataset and (optionally) a LoRA adapter from S3,
runs evaluation on both the base and fine-tuned model,
and uploads the comparison results to S3.

Usage:
    pip install kfp kfp-kubernetes
    python pipeline/llm_eval_pipeline.py

This produces pipeline/llm_eval_pipeline.yaml.
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
def download_file(
    s3_bucket: str,
    s3_key: str,
    output_file: dsl.Output[dsl.Artifact],
):
    """Download a file from S3."""
    return dsl.ContainerSpec(
        image=_image("upload"),
        command=["sh", "-c"],
        args=[
            "mkdir -p $(dirname \"$1\") && python3 -c '"
            "import boto3,os,sys;"
            "from botocore.client import Config;"
            "out=sys.argv[1];bucket=sys.argv[2];key=sys.argv[3];"
            "os.makedirs(os.path.dirname(out) or \".\",exist_ok=True);"
            "s3=boto3.client(\"s3\","
            "endpoint_url=os.environ[\"AWS_S3_ENDPOINT\"],"
            "aws_access_key_id=os.environ[\"AWS_ACCESS_KEY_ID\"],"
            "aws_secret_access_key=os.environ[\"AWS_SECRET_ACCESS_KEY\"],"
            "config=Config(signature_version=\"s3v4\"));"
            "s3.download_file(bucket,key,out);"
            "print(f\"Downloaded s3://{bucket}/{key} ({os.path.getsize(out)} bytes)\")"
            "' \"$1\" \"$2\" \"$3\"",
            "--",
            output_file.path,
            s3_bucket,
            s3_key,
        ],
    )


@dsl.container_component
def download_adapter(
    s3_bucket: str,
    s3_prefix: str,
    adapter_dir: dsl.Output[dsl.Artifact],
):
    """Download a LoRA adapter directory from S3."""
    return dsl.ContainerSpec(
        image=_image("upload"),
        command=["sh", "-c"],
        args=[
            "mkdir -p \"$1\" && python3 -c '"
            "import boto3,os,sys;"
            "from botocore.client import Config;"
            "out_dir=sys.argv[1];bucket=sys.argv[2];prefix=sys.argv[3];"
            "s3=boto3.client(\"s3\","
            "endpoint_url=os.environ[\"AWS_S3_ENDPOINT\"],"
            "aws_access_key_id=os.environ[\"AWS_ACCESS_KEY_ID\"],"
            "aws_secret_access_key=os.environ[\"AWS_SECRET_ACCESS_KEY\"],"
            "config=Config(signature_version=\"s3v4\"));"
            "resp=s3.list_objects_v2(Bucket=bucket,Prefix=prefix);"
            "files=resp.get(\"Contents\",[]);"
            "print(f\"Found {len(files)} files under s3://{bucket}/{prefix}\");"
            "[s3.download_file(bucket,f[\"Key\"],os.path.join(out_dir,os.path.relpath(f[\"Key\"],prefix))) or "
            "os.makedirs(os.path.dirname(os.path.join(out_dir,os.path.relpath(f[\"Key\"],prefix))),exist_ok=True) or None "
            "for f in files if not f[\"Key\"].endswith(\"/\")];"
            "print(f\"Downloaded {len(files)} files to {out_dir}\")"
            "' \"$1\" \"$2\" \"$3\"",
            "--",
            adapter_dir.path,
            s3_bucket,
            s3_prefix,
        ],
    )


@dsl.container_component
def evaluate_llm(
    dataset_file: dsl.Input[dsl.Artifact],
    adapter_dir: dsl.Input[dsl.Artifact],
    base_model: str,
    num_samples: str,
    eval_dir: dsl.Output[dsl.Artifact],
):
    """Run evaluation comparing base model vs fine-tuned model."""
    return dsl.ContainerSpec(
        image=_image("evaluate"),
        command=["sh", "-c"],
        args=[
            "python -m src.evaluate_llm"
            " --base-model \"$1\""
            " --adapter \"$2\""
            " --dataset \"$3\""
            " --output-dir \"$4\""
            " --num-samples \"$5\"",
            "--",
            base_model,
            adapter_dir.path,
            dataset_file.path,
            eval_dir.path,
            num_samples,
        ],
    )


@dsl.container_component
def evaluate_base_only(
    dataset_file: dsl.Input[dsl.Artifact],
    base_model: str,
    num_samples: str,
    eval_dir: dsl.Output[dsl.Artifact],
):
    """Run evaluation on the base model only (no adapter)."""
    return dsl.ContainerSpec(
        image=_image("evaluate"),
        command=["sh", "-c"],
        args=[
            "python -m src.evaluate_llm"
            " --base-model \"$1\""
            " --dataset \"$2\""
            " --output-dir \"$3\""
            " --num-samples \"$4\"",
            "--",
            base_model,
            dataset_file.path,
            eval_dir.path,
            num_samples,
        ],
    )


@dsl.container_component
def upload_results(
    eval_dir: dsl.Input[dsl.Artifact],
    s3_bucket: str,
    prefix: str,
):
    """Upload evaluation results to S3."""
    return dsl.ContainerSpec(
        image=_image("upload"),
        command=["sh", "-c"],
        args=[
            "python upload_artifacts.py"
            " --local-dir \"$1\""
            " --bucket \"$2\""
            " --prefix \"$3\"",
            "--",
            eval_dir.path,
            s3_bucket,
            prefix,
        ],
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dsl.pipeline(
    name="OCP LLM Evaluation",
    description="Evaluate base TinyLlama vs fine-tuned LoRA adapter on OpenShift troubleshooting",
)
def llm_eval_pipeline(
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    num_samples: str = "50",
    s3_bucket: str = "pipelines",
    s3_dataset_key: str = "data/ocp-instructions.jsonl",
    s3_adapter_prefix: str = "ocp-llm/tinyllama-qlora/latest/final",
    s3_results_prefix: str = "ocp-llm/eval/latest",
):
    # Download dataset
    dl_dataset = download_file(
        s3_bucket=s3_bucket,
        s3_key=s3_dataset_key,
    )
    _inject_s3_secret(dl_dataset)

    # Download LoRA adapter
    dl_adapter = download_adapter(
        s3_bucket=s3_bucket,
        s3_prefix=s3_adapter_prefix,
    )
    _inject_s3_secret(dl_adapter)

    # Evaluate both models
    eval_task = evaluate_llm(
        dataset_file=dl_dataset.outputs["output_file"],
        adapter_dir=dl_adapter.outputs["adapter_dir"],
        base_model=base_model,
        num_samples=num_samples,
    ).set_accelerator_type("nvidia.com/gpu").set_accelerator_limit("1")

    # Upload results
    upload_task = upload_results(
        eval_dir=eval_task.outputs["eval_dir"],
        s3_bucket=s3_bucket,
        prefix=s3_results_prefix,
    )
    _inject_s3_secret(upload_task)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=llm_eval_pipeline,
        package_path="pipeline/llm_eval_pipeline.yaml",
    )
    print("Pipeline compiled to pipeline/llm_eval_pipeline.yaml")
