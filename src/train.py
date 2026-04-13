"""Training entrypoint for supervised fine-tuning with LoRA/QLoRA.

Usage:
    python -m src.train --config config/ocp-instruct.yaml
"""

import argparse
import json
import logging
import yaml
from pathlib import Path

from transformers import TrainingArguments
from trl import SFTTrainer

from src.model import load_tokenizer, load_base_model, apply_lora
from src.data import (
    load_training_data,
    split_dataset,
    format_dataset_for_sft,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_training_args(config: dict, output_dir: str) -> TrainingArguments:
    t = config["training"]
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=t.get("epochs", 3),
        per_device_train_batch_size=t.get("batch_size", 4),
        per_device_eval_batch_size=t.get("eval_batch_size", 4),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 4),
        learning_rate=t.get("learning_rate", 2e-4),
        lr_scheduler_type=t.get("lr_scheduler", "cosine"),
        warmup_ratio=t.get("warmup_ratio", 0.03),
        weight_decay=t.get("weight_decay", 0.01),
        bf16=t.get("bf16", True),
        fp16=t.get("fp16", False),
        logging_steps=t.get("logging_steps", 10),
        save_strategy=t.get("save_strategy", "no"),
        eval_strategy=t.get("eval_strategy", "steps"),
        eval_steps=t.get("eval_steps", 200),
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=t.get("optimizer", "paged_adamw_8bit"),
        max_grad_norm=t.get("max_grad_norm", 0.3),
        report_to=t.get("report_to", "tensorboard"),
        seed=t.get("seed", 42),
        dataloader_pin_memory=True,
        remove_unused_columns=True,
        load_best_model_at_end=False,
    )


def save_training_metadata(config: dict, output_dir: str, train_result, eval_result=None):
    """Save training metadata and metrics as JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model": config["model"]["name"],
        "lora_r": config["lora"].get("r", 16),
        "lora_alpha": config["lora"].get("alpha", 32),
        "quantization": config["model"].get("quantization", {}).get("mode", "none"),
        "dataset": config["dataset"].get("local_path") or config["dataset"].get("hub_name"),
        "epochs": config["training"].get("epochs", 3),
        "batch_size": config["training"].get("batch_size", 4),
        "learning_rate": config["training"].get("learning_rate", 2e-4),
        "max_seq_length": config["training"].get("max_seq_length", 2048),
        "train_loss": train_result.metrics.get("train_loss"),
        "train_runtime_seconds": train_result.metrics.get("train_runtime"),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
    }

    if eval_result:
        metadata["eval_loss"] = eval_result.get("eval_loss")

    with open(out / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Training metadata saved to %s", out / "training_metadata.json")


def train(config_path: str):
    config = load_config(config_path)
    output_dir = config["training"].get("output_dir", "./outputs/llm-sft")
    max_seq_length = config["training"].get("max_seq_length", 2048)

    logger.info("Model: %s", config["model"]["name"])
    logger.info("Dataset: %s", config["dataset"].get("local_path") or config["dataset"].get("hub_name"))

    tokenizer = load_tokenizer(config["model"]["name"])
    model = load_base_model(config)
    model = apply_lora(model, config)

    raw_dataset = load_training_data(config)

    eval_ratio = config["dataset"].get("eval_ratio", 0.1)
    train_ds, eval_ds = split_dataset(raw_dataset, eval_ratio=eval_ratio)

    logger.info("Formatting dataset with chat template...")
    train_formatted = format_dataset_for_sft(train_ds, tokenizer, config)
    eval_formatted = format_dataset_for_sft(eval_ds, tokenizer, config) if eval_ds else None

    sample = train_formatted[0]["text"]
    logger.info("Sample formatted text (first 500 chars):\n%s", sample[:500])

    training_args = build_training_args(config, output_dir)

    if eval_formatted is None:
        training_args.eval_strategy = "no"

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_formatted,
        eval_dataset=eval_formatted,
        processing_class=tokenizer,
    )

    logger.info("Starting training...")
    train_result = trainer.train()
    logger.info("Training complete. Loss: %.4f", train_result.metrics.get("train_loss", 0))

    eval_metrics = None
    if eval_formatted:
        logger.info("Running final evaluation...")
        eval_metrics = trainer.evaluate()
        logger.info("Eval loss: %.4f", eval_metrics.get("eval_loss", 0))

    final_dir = Path(output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("Model + tokenizer saved to %s", final_dir)

    save_training_metadata(config, str(final_dir), train_result, eval_metrics)

    logger.info("Done. Artifacts in %s", final_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune an LLM with LoRA/QLoRA")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    train(args.config)
