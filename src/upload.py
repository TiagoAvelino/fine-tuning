"""Upload fine-tuned model artifacts to S3 (MinIO)."""

import argparse
import os
from pathlib import Path

import boto3


def get_s3_client():
    """Create an S3 client using environment variables (compatible with MinIO)."""
    return boto3.client(
        "s3",
        endpoint_url=os.environ["AWS_S3_ENDPOINT"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )


def upload_directory(local_dir: str, bucket: str, prefix: str = ""):
    """Recursively upload a local directory to an S3 bucket."""
    s3 = get_s3_client()
    local_path = Path(local_dir)

    files = [f for f in local_path.rglob("*") if f.is_file()]
    print(f"Uploading {len(files)} files from {local_dir} to s3://{bucket}/{prefix}")

    for file_path in files:
        relative = file_path.relative_to(local_path)
        s3_key = f"{prefix}/{relative}" if prefix else str(relative)
        print(f"  {relative} -> s3://{bucket}/{s3_key}")
        s3.upload_file(str(file_path), bucket, s3_key)

    print("Upload complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload model to S3/MinIO")
    parser.add_argument("--model-dir", required=True, help="Local model directory")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--prefix", default="", help="S3 key prefix")
    args = parser.parse_args()

    upload_directory(args.model_dir, args.bucket, args.prefix)
