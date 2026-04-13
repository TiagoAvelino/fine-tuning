"""Train an OpenShift issue classifier.

Self-contained training script — no cross-module imports.
Designed to run as a standalone KFP component or CLI tool.

Inputs:  train CSV, eval CSV (or single CSV with auto-split), model name
Outputs: model weights, tokenizer, labels.json, eval_metrics.json, metadata.json

Usage:
    python src/train_classifier.py \
        --train-file data/splits/train.csv \
        --eval-file data/splits/eval.csv \
        --model-name distilbert-base-uncased \
        --output-dir outputs/run-001 \
        --num-epochs 10

    python src/train_classifier.py \
        --train-file data/ocp-issues-v2.csv \
        --model-name roberta-base \
        --output-dir outputs/run-002 \
        --learning-rate 1e-5
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_classifier")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train a text classifier for OpenShift issue triage",
    )
    p.add_argument("--train-file", required=True, help="CSV with text,label columns")
    p.add_argument("--eval-file", default=None, help="Separate eval CSV (if omitted, split from train-file)")
    p.add_argument("--test-size", type=float, default=0.2, help="Eval split fraction when no eval-file")
    p.add_argument("--text-column", default="text")
    p.add_argument("--label-column", default="label")
    p.add_argument("--model-name", default="distilbert-base-uncased")
    p.add_argument("--output-dir", required=True, help="Directory for all output artifacts")
    p.add_argument("--num-epochs", type=int, default=10)
    p.add_argument("--train-batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-length", type=int, default=512, help="Tokenizer max sequence length")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(args):
    """Load CSV dataset and return train/eval splits."""
    logger.info("Loading training data: %s", args.train_file)
    train_ds = load_dataset("csv", data_files=args.train_file, split="train")

    if args.eval_file:
        logger.info("Loading eval data: %s", args.eval_file)
        eval_ds = load_dataset("csv", data_files=args.eval_file, split="train")
    else:
        logger.info("Auto-splitting %.0f%% for eval (seed=%d)", args.test_size * 100, args.seed)
        split = train_ds.train_test_split(test_size=args.test_size, seed=args.seed)
        train_ds, eval_ds = split["train"], split["test"]

    logger.info("Train: %d samples — Eval: %d samples", len(train_ds), len(eval_ds))
    return train_ds, eval_ds


def build_label_mapping(dataset, label_column):
    """Discover labels from dataset and build id mappings."""
    labels = sorted(set(dataset[label_column]))
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    logger.info("Labels (%d): %s", len(label2id), list(label2id.keys()))
    return label2id, id2label


def encode_labels(dataset, label2id, label_column):
    """Map string labels to integer IDs."""
    def _encode(example):
        example["label"] = label2id[example[label_column]]
        return example
    dataset = dataset.map(_encode, desc="Encoding labels")
    from datasets import Value
    dataset = dataset.cast_column("label", Value("int64"))
    return dataset


def tokenize_dataset(dataset, tokenizer, text_column, max_length):
    """Tokenize a dataset split and remove raw text columns."""
    def _tok(examples):
        return tokenizer(examples[text_column], truncation=True, max_length=max_length)
    cols_to_remove = [c for c in dataset.column_names if c not in ("label",)]
    dataset = dataset.map(_tok, batched=True, remove_columns=cols_to_remove, desc="Tokenizing")
    return dataset


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(model_name, label2id, id2label):
    """Load pretrained model + tokenizer with a classification head."""
    logger.info("Loading model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )
    total = sum(p.numel() for p in model.parameters())
    logger.info("Parameters: %s", f"{total:,}")
    return model, tokenizer


def build_trainer(model, tokenizer, train_ds, eval_ds, compute_metrics_fn, args):
    """Create HF Trainer with TrainingArguments."""
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="no",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=10,
        report_to="none",
        seed=args.seed,
    )
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics_fn,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )


# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------

def export_onnx(model, tokenizer, output_dir, max_length=512):
    """Export model to ONNX format in OpenVINO-compatible directory structure."""
    import torch

    onnx_dir = Path(output_dir) / "onnx" / "1"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "model.onnx"

    model.eval()
    device = next(model.parameters()).device
    dummy = tokenizer("example text", return_tensors="pt", truncation=True, max_length=max_length)
    dummy = {k: v.to(device) for k, v in dummy.items()}

    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=14,
    )
    logger.info("ONNX model exported to %s (%.1f MB)",
                onnx_path, onnx_path.stat().st_size / 1e6)
    return onnx_dir.parent


def save_artifacts(trainer, tokenizer, label2id, id2label, args):
    """Save model, tokenizer, labels, eval metrics, and metadata."""
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))

    try:
        export_onnx(trainer.model, tokenizer, out, args.max_length)
    except Exception as e:
        logger.warning("ONNX export failed (non-fatal): %s", e)

    with open(out / "labels.json", "w") as f:
        json.dump({
            "label2id": label2id,
            "id2label": {str(k): v for k, v in id2label.items()},
        }, f, indent=2)

    eval_entries = [m for m in trainer.state.log_history if "eval_accuracy" in m]
    if eval_entries:
        best = max(eval_entries, key=lambda m: m["eval_accuracy"])
        metrics = {
            "best_accuracy": best["eval_accuracy"],
            "best_epoch": best.get("epoch"),
            "best_eval_loss": best.get("eval_loss"),
            "final_accuracy": eval_entries[-1]["eval_accuracy"],
            "final_eval_loss": eval_entries[-1].get("eval_loss"),
            "train_loss": next(
                (m["train_loss"] for m in reversed(trainer.state.log_history) if "train_loss" in m),
                None,
            ),
            "per_epoch": [
                {"epoch": m.get("epoch"), "accuracy": m["eval_accuracy"], "loss": m.get("eval_loss")}
                for m in eval_entries
            ],
        }
    else:
        metrics = {}

    with open(out / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    metadata = {
        "model_name": args.model_name,
        "train_file": args.train_file,
        "eval_file": args.eval_file,
        "test_size": args.test_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "train_batch_size": args.train_batch_size,
        "seed": args.seed,
        "max_length": args.max_length,
        "num_labels": len(label2id),
        "labels": label2id,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("All artifacts saved to %s", out)
    if eval_entries:
        logger.info("Best accuracy: %.4f (epoch %s)", metrics["best_accuracy"], metrics["best_epoch"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
        logger.info("GPU: %s (%.1f GB)", torch.cuda.get_device_name(0), mem / 1e9)

    train_ds, eval_ds = load_data(args)
    label2id, id2label = build_label_mapping(train_ds, args.label_column)
    train_ds = encode_labels(train_ds, label2id, args.label_column)
    eval_ds = encode_labels(eval_ds, label2id, args.label_column)

    model, tokenizer = build_model(args.model_name, label2id, id2label)
    train_tok = tokenize_dataset(train_ds, tokenizer, args.text_column, args.max_length)
    eval_tok = tokenize_dataset(eval_ds, tokenizer, args.text_column, args.max_length)

    accuracy_metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return accuracy_metric.compute(predictions=np.argmax(logits, axis=-1), references=labels)

    trainer = build_trainer(model, tokenizer, train_tok, eval_tok, compute_metrics, args)

    logger.info("Training: %d epochs, batch=%d, lr=%s, model=%s",
                args.num_epochs, args.train_batch_size, args.learning_rate, args.model_name)
    trainer.train()
    logger.info("Training complete")

    save_artifacts(trainer, tokenizer, label2id, id2label, args)
    logger.info("Done. Output: %s", args.output_dir)


if __name__ == "__main__":
    main()
