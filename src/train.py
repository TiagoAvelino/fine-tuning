"""Training entrypoint for supervised fine-tuning with LoRA/QLoRA."""

import argparse
import yaml
from pathlib import Path

from transformers import TrainingArguments
from trl import SFTTrainer

from src.model import load_tokenizer, load_base_model, apply_lora
from src.data import load_training_data, preprocess_dataset


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_training_args(config: dict, output_dir: str) -> TrainingArguments:
    t = config["training"]
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=t.get("epochs", 3),
        per_device_train_batch_size=t.get("batch_size", 4),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 4),
        learning_rate=t.get("learning_rate", 2e-4),
        lr_scheduler_type=t.get("lr_scheduler", "cosine"),
        warmup_ratio=t.get("warmup_ratio", 0.03),
        weight_decay=t.get("weight_decay", 0.01),
        bf16=t.get("bf16", True),
        fp16=t.get("fp16", False),
        logging_steps=t.get("logging_steps", 10),
        save_strategy=t.get("save_strategy", "steps"),
        save_steps=t.get("save_steps", 100),
        save_total_limit=t.get("save_total_limit", 3),
        eval_strategy=t.get("eval_strategy", "no"),
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=t.get("optimizer", "paged_adamw_8bit"),
        max_grad_norm=t.get("max_grad_norm", 0.3),
        report_to=t.get("report_to", "tensorboard"),
        seed=t.get("seed", 42),
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )


def train(config_path: str):
    config = load_config(config_path)
    output_dir = config["training"].get("output_dir", "./outputs")

    tokenizer = load_tokenizer(config["model"]["name"])
    model = load_base_model(config)
    model = apply_lora(model, config)

    dataset = load_training_data(config)

    text_field = config["dataset"].get("text_field", "text")
    template = config["dataset"].get("template")

    training_args = build_training_args(config, output_dir)

    if template:
        from src.data import format_instruction

        def formatting_func(examples):
            return [format_instruction(ex, template) for ex in _unpack(examples)]

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            formatting_func=formatting_func,
            max_seq_length=config["training"].get("max_seq_length", 2048),
            packing=config["training"].get("packing", False),
        )
    else:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            max_seq_length=config["training"].get("max_seq_length", 2048),
            packing=config["training"].get("packing", False),
        )

    trainer.train()

    final_dir = Path(output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Model saved to {final_dir}")


def _unpack(batch: dict) -> list[dict]:
    keys = list(batch.keys())
    return [
        {k: batch[k][i] for k in keys}
        for i in range(len(batch[keys[0]]))
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune an LLM with LoRA/QLoRA")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    train(args.config)
