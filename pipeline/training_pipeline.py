"""KCS Classifier Training Pipeline — KFP v2.

Compiles to YAML for import into OpenShift AI Data Science Pipelines.
S3 credentials are injected from a Data Connection secret, not hardcoded.

Usage:
    pip install kfp kfp-kubernetes
    python pipeline/training_pipeline.py

This produces pipeline/training_pipeline.yaml
"""

from kfp import dsl, compiler, kubernetes

REGISTRY = "quay.io/rh_ee_tavelino"
PROJECT = "kcs-classifier"
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
    """Mount the Data Connection secret as env vars on a task."""
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
    """Download dataset CSV from MinIO/S3 using env-injected credentials."""
    return dsl.ContainerSpec(
        image=_image("upload-artifacts"),
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
            "bucket=os.environ.get(\"AWS_S3_BUCKET\",\"fine-tuning\");"
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
    input_file: dsl.Input[dsl.Artifact],
    test_size: str,
    seed: str,
    output_dir: dsl.Output[dsl.Artifact],
):
    return dsl.ContainerSpec(
        image=_image("prepare-dataset"),
        command=["python", "prepare_dataset.py"],
        args=[
            "--input-file", input_file.path,
            "--output-dir", output_dir.path,
            "--test-size", test_size,
            "--seed", seed,
        ],
    )


@dsl.container_component
def train_model(
    splits_dir: dsl.Input[dsl.Artifact],
    model_name: str,
    num_epochs: str,
    train_batch_size: str,
    learning_rate: str,
    seed: str,
    model_dir: dsl.Output[dsl.Artifact],
):
    return dsl.ContainerSpec(
        image=_image("train"),
        command=["sh", "-c"],
        args=[
            "python train_classifier.py"
            " --train-file \"$1/train.csv\""
            " --eval-file \"$1/eval.csv\""
            " --model-name \"$2\""
            " --output-dir \"$3\""
            " --num-epochs \"$4\""
            " --train-batch-size \"$5\""
            " --learning-rate \"$6\""
            " --seed \"$7\"",
            "--",
            splits_dir.path,
            model_name,
            model_dir.path,
            num_epochs,
            train_batch_size,
            learning_rate,
            seed,
        ],
    )


@dsl.container_component
def evaluate_model(
    model_dir: dsl.Input[dsl.Artifact],
    splits_dir: dsl.Input[dsl.Artifact],
    eval_dir: dsl.Output[dsl.Artifact],
):
    return dsl.ContainerSpec(
        image=_image("evaluate"),
        command=["sh", "-c"],
        args=[
            "python evaluate_model.py"
            " --model-dir \"$1\""
            " --eval-file \"$2/eval.csv\""
            " --output-dir \"$3\"",
            "--",
            model_dir.path,
            splits_dir.path,
            eval_dir.path,
        ],
    )


@dsl.container_component
def upload_model(
    model_dir: dsl.Input[dsl.Artifact],
    prefix: str,
):
    """Upload model artifacts to S3/MinIO using env-injected credentials."""
    return dsl.ContainerSpec(
        image=_image("upload-artifacts"),
        command=["sh", "-c"],
        args=[
            "python upload_artifacts.py"
            " --local-dir \"$1\""
            " --bucket \"${AWS_S3_BUCKET:-fine-tuning}\""
            " --prefix \"$2\"",
            "--",
            model_dir.path,
            prefix,
        ],
    )


@dsl.pipeline(
    name="KCS Classifier Training",
    description="Prepare data, train, evaluate, and upload an OpenShift issue classifier",
)
def training_pipeline(
    model_name: str = "distilbert-base-uncased",
    num_epochs: str = "10",
    train_batch_size: str = "16",
    learning_rate: str = "0.00002",
    test_size: str = "0.2",
    seed: str = "42",
    s3_dataset_key: str = "data/ocp-issues-v2.csv",
    s3_model_prefix: str = "kcs-classifier/latest",
):
    download_task = download_dataset(s3_key=s3_dataset_key)
    _inject_s3_secret(download_task)

    prep_task = prepare_dataset(
        input_file=download_task.outputs["dataset_file"],
        test_size=test_size,
        seed=seed,
    )

    train_task = train_model(
        splits_dir=prep_task.outputs["output_dir"],
        model_name=model_name,
        num_epochs=num_epochs,
        train_batch_size=train_batch_size,
        learning_rate=learning_rate,
        seed=seed,
    ).set_accelerator_type("nvidia.com/gpu").set_accelerator_limit("1")

    eval_task = evaluate_model(
        model_dir=train_task.outputs["model_dir"],
        splits_dir=prep_task.outputs["output_dir"],
    )

    upload_task = upload_model(
        model_dir=train_task.outputs["model_dir"],
        prefix=s3_model_prefix,
    )
    _inject_s3_secret(upload_task)
    upload_task.after(eval_task)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path="pipeline/training_pipeline.yaml",
    )
    print("Pipeline compiled to pipeline/training_pipeline.yaml")
