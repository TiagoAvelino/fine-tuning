"""Upload model artifacts to S3/MinIO.

Standalone CLI tool for uploading a directory to an S3-compatible store.
Credentials are read from environment variables.

Required env vars:
    AWS_S3_ENDPOINT         MinIO/S3 endpoint URL
    AWS_ACCESS_KEY_ID       Access key
    AWS_SECRET_ACCESS_KEY   Secret key
    AWS_DEFAULT_REGION      Region (default: us-east-1)

Usage:
    python src/upload_artifacts.py \
        --local-dir outputs/run-001 \
        --bucket fine-tuning \
        --prefix kcs-classifier/v2
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import boto3
from botocore.client import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("upload_artifacts")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Upload artifacts to S3/MinIO")
    p.add_argument("--local-dir", required=True, help="Local directory to upload")
    p.add_argument("--bucket", required=True, help="S3 bucket name")
    p.add_argument("--prefix", required=True, help="S3 key prefix")
    return p.parse_args(argv)


def build_s3_client():
    """Build S3 client from environment variables."""
    required = ["AWS_S3_ENDPOINT", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    missing = [v for v in required if v not in os.environ]
    if missing:
        logger.error("Missing environment variables: %s", ", ".join(missing))
        sys.exit(1)

    return boto3.client(
        "s3",
        endpoint_url=os.environ["AWS_S3_ENDPOINT"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )


def upload_directory(local_dir, bucket, prefix):
    """Recursively upload all files in a directory to S3."""
    local_path = Path(local_dir)
    if not local_path.exists():
        logger.error("Directory not found: %s", local_path)
        sys.exit(1)

    files = [f for f in local_path.rglob("*") if f.is_file()]
    if not files:
        logger.warning("No files found in %s", local_path)
        return

    s3 = build_s3_client()
    logger.info("Uploading %d files to s3://%s/%s", len(files), bucket, prefix)

    total_bytes = 0
    for file_path in files:
        relative = file_path.relative_to(local_path).as_posix()
        s3_key = f"{prefix}/{relative}"
        size = file_path.stat().st_size
        total_bytes += size
        logger.info("  %s (%s) -> s3://%s/%s", relative, _human_size(size), bucket, s3_key)
        s3.upload_file(str(file_path), bucket, s3_key)

    logger.info("Upload complete: %d files, %s total", len(files), _human_size(total_bytes))


def _human_size(num_bytes):
    """Format bytes as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def main(argv=None):
    args = parse_args(argv)
    upload_directory(args.local_dir, args.bucket, args.prefix)


if __name__ == "__main__":
    main()
