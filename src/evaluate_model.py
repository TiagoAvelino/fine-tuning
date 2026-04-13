"""Evaluate a trained classifier on a CSV dataset.

Loads a saved model directory and runs inference on every row in the eval file.
Produces eval_metrics.json and sample_predictions.json.

Usage:
    python src/evaluate_model.py \
        --model-dir outputs/run-001 \
        --eval-file data/splits/eval.csv \
        --output-dir outputs/run-001/evaluation
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate_model")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a trained classifier")
    p.add_argument("--model-dir", required=True, help="Directory with saved model + tokenizer + labels.json")
    p.add_argument("--eval-file", required=True, help="CSV file to evaluate on")
    p.add_argument("--output-dir", required=True, help="Directory for eval_metrics.json, sample_predictions.json")
    p.add_argument("--text-column", default="text")
    p.add_argument("--label-column", default="label")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-samples", type=int, default=0, help="Limit predictions saved (0 = all)")
    return p.parse_args(argv)


def load_labels(model_dir):
    """Load label mappings from labels.json in model directory."""
    labels_path = Path(model_dir) / "labels.json"
    if not labels_path.exists():
        logger.error("labels.json not found in %s", model_dir)
        sys.exit(1)

    with open(labels_path) as f:
        data = json.load(f)

    label2id = data["label2id"]
    id2label = {int(k): v for k, v in data["id2label"].items()}
    return label2id, id2label


def main(argv=None):
    args = parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    label2id, id2label = load_labels(args.model_dir)
    logger.info("Labels: %s", list(label2id.keys()))

    logger.info("Loading model from %s", args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    logger.info("Loading eval data from %s", args.eval_file)
    dataset = load_dataset("csv", data_files=args.eval_file, split="train")
    logger.info("Eval samples: %d", len(dataset))

    all_preds = []
    all_labels = []
    predictions_detail = []

    for i in range(len(dataset)):
        row = dataset[i]
        text = row[args.text_column]
        true_label_str = row[args.label_column]
        true_id = label2id.get(true_label_str, -1)

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
        pred_id = int(np.argmax(probs))

        all_preds.append(pred_id)
        all_labels.append(true_id)

        limit = args.max_samples if args.max_samples > 0 else len(dataset)
        if len(predictions_detail) < limit:
            predictions_detail.append({
                "text": text[:300],
                "true_label": true_label_str,
                "predicted_label": id2label.get(pred_id, str(pred_id)),
                "confidence": float(probs[pred_id]),
                "correct": pred_id == true_id,
            })

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = float(np.mean(all_preds == all_labels))

    per_class = {}
    for label_str, label_id in sorted(label2id.items(), key=lambda x: x[1]):
        mask = all_labels == label_id
        if mask.sum() == 0:
            per_class[label_str] = {"accuracy": 0.0, "count": 0, "correct": 0}
            continue
        class_correct = int((all_preds[mask] == label_id).sum())
        per_class[label_str] = {
            "accuracy": float(class_correct / mask.sum()),
            "count": int(mask.sum()),
            "correct": class_correct,
        }

    confusion = {}
    for true_id_val, pred_id_val in zip(all_labels, all_preds):
        true_str = id2label.get(int(true_id_val), str(true_id_val))
        pred_str = id2label.get(int(pred_id_val), str(pred_id_val))
        key = f"{true_str} -> {pred_str}"
        confusion[key] = confusion.get(key, 0) + 1

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics = {
        "accuracy": accuracy,
        "total_samples": len(dataset),
        "correct": int((all_preds == all_labels).sum()),
        "per_class": per_class,
        "confusion": confusion,
    }
    with open(out / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    preds_out = {
        "total_samples": len(dataset),
        "samples_shown": len(predictions_detail),
        "accuracy": accuracy,
        "predictions": predictions_detail,
    }
    with open(out / "sample_predictions.json", "w") as f:
        json.dump(preds_out, f, indent=2)

    logger.info("Accuracy: %.4f (%d/%d)", accuracy, metrics["correct"], len(dataset))
    for label_str, stats in per_class.items():
        logger.info("  %s: %.4f (%d/%d)", label_str, stats["accuracy"], stats["correct"], stats["count"])
    logger.info("Results saved to %s", out)


if __name__ == "__main__":
    main()
