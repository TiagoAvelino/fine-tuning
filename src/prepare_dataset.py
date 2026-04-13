"""Prepare and split a CSV dataset for classifier training.

Validates the schema, checks label balance, splits into train/eval,
and writes both files to the output directory.

Usage:
    python src/prepare_dataset.py \
        --input-file data/ocp-issues-v2.csv \
        --output-dir data/splits \
        --test-size 0.2 \
        --seed 42
"""

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("prepare_dataset")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Validate and split a CSV dataset")
    p.add_argument("--input-file", required=True, help="Raw CSV file (text,label,...)")
    p.add_argument("--output-dir", required=True, help="Directory for train.csv, eval.csv, dataset_info.json")
    p.add_argument("--text-column", default="text")
    p.add_argument("--label-column", default="label")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def validate_csv(file_path, text_col, label_col):
    """Read CSV, validate required columns exist, return rows."""
    path = Path(file_path)
    if not path.exists():
        logger.error("File not found: %s", path)
        sys.exit(1)

    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if text_col not in reader.fieldnames:
            logger.error("Missing column '%s' in %s (found: %s)", text_col, path, reader.fieldnames)
            sys.exit(1)
        if label_col not in reader.fieldnames:
            logger.error("Missing column '%s' in %s (found: %s)", label_col, path, reader.fieldnames)
            sys.exit(1)

        for i, row in enumerate(reader, start=2):
            if not row[text_col].strip():
                logger.warning("Empty text at line %d, skipping", i)
                continue
            if not row[label_col].strip():
                logger.warning("Empty label at line %d, skipping", i)
                continue
            rows.append(row)

    logger.info("Loaded %d valid rows from %s", len(rows), path)
    return rows, list(reader.fieldnames)


def split_data(rows, test_size, seed):
    """Deterministic train/eval split."""
    import random
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)

    n_eval = int(len(rows) * test_size)
    eval_indices = set(indices[:n_eval])

    train_rows = [rows[i] for i in range(len(rows)) if i not in eval_indices]
    eval_rows = [rows[i] for i in range(len(rows)) if i in eval_indices]

    return train_rows, eval_rows


def write_csv(rows, fieldnames, output_path):
    """Write rows to a CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %d rows to %s", len(rows), output_path)


def main(argv=None):
    args = parse_args(argv)

    rows, fieldnames = validate_csv(args.input_file, args.text_column, args.label_column)

    label_counts = Counter(row[args.label_column] for row in rows)
    logger.info("Label distribution:")
    for label, count in sorted(label_counts.items()):
        logger.info("  %s: %d", label, count)

    train_rows, eval_rows = split_data(rows, args.test_size, args.seed)

    out = Path(args.output_dir)
    write_csv(train_rows, fieldnames, out / "train.csv")
    write_csv(eval_rows, fieldnames, out / "eval.csv")

    label2id = {label: idx for idx, label in enumerate(sorted(label_counts.keys()))}
    id2label = {idx: label for label, idx in label2id.items()}

    info = {
        "source_file": args.input_file,
        "total_samples": len(rows),
        "train_samples": len(train_rows),
        "eval_samples": len(eval_rows),
        "test_size": args.test_size,
        "seed": args.seed,
        "num_labels": len(label2id),
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "label_distribution": dict(sorted(label_counts.items())),
    }

    info_path = out / "dataset_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    logger.info("Dataset info saved to %s", info_path)

    labels_path = out / "labels.json"
    with open(labels_path, "w") as f:
        json.dump({"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}, f, indent=2)
    logger.info("Labels saved to %s", labels_path)

    logger.info("Dataset preparation complete: %d train, %d eval, %d labels",
                len(train_rows), len(eval_rows), len(label2id))


if __name__ == "__main__":
    main()
