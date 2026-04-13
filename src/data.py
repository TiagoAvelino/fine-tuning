"""Dataset loading, chat formatting, and train/eval splitting for LLM fine-tuning."""

import json
import logging
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an OpenShift and Kubernetes troubleshooting expert. "
    "Given a problem description, diagnose the issue and respond with a JSON object "
    "containing: category, sub_cause, confidence, explanation, fix, commands, verification."
)


def load_training_data(config: dict) -> Dataset:
    """Load dataset from local JSONL/JSON file or HuggingFace Hub."""
    source = config["dataset"]

    if source.get("hub_name"):
        dataset = load_dataset(
            source["hub_name"],
            split=source.get("split", "train"),
        )
    elif source.get("local_path"):
        file_path = source["local_path"]
        ext = file_path.rsplit(".", 1)[-1]
        fmt = {"json": "json", "jsonl": "json", "csv": "csv", "parquet": "parquet"}
        dataset = load_dataset(
            fmt.get(ext, "json"),
            data_files=file_path,
            split="train",
        )
    else:
        raise ValueError("Provide either 'hub_name' or 'local_path' in dataset config")

    if source.get("max_samples"):
        dataset = dataset.select(range(min(source["max_samples"], len(dataset))))

    logger.info("Loaded dataset: %d samples", len(dataset))
    return dataset


def split_dataset(dataset: Dataset, eval_ratio: float = 0.1, seed: int = 42):
    """Split into train and eval datasets."""
    if eval_ratio <= 0:
        return dataset, None

    split = dataset.train_test_split(test_size=eval_ratio, seed=seed)
    logger.info("Split: %d train, %d eval", len(split["train"]), len(split["test"]))
    return split["train"], split["test"]


def build_chat_messages(instruction: str, response: str | None = None) -> list[dict]:
    """Build chat messages in the standard messages format.

    If response is provided (training), returns the full conversation.
    If response is None (inference), returns only system + user messages.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    return messages


def format_dataset_for_sft(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    config: dict,
) -> Dataset:
    """Format each example as a full chat conversation string for SFT.

    Uses the tokenizer's built-in chat template (Llama, Mistral, etc.)
    to produce the correctly formatted training text.
    """
    instruction_field = config["dataset"].get("instruction_field", "instruction")
    response_field = config["dataset"].get("response_field", "response")

    def apply_template(examples):
        texts = []
        for inst, resp in zip(examples[instruction_field], examples[response_field]):
            messages = build_chat_messages(inst, resp)
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    formatted = dataset.map(
        apply_template,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Formatting chat templates",
    )
    logger.info("Formatted %d examples with chat template", len(formatted))
    return formatted


def format_instruction(sample: dict, template: str) -> str:
    """Apply a plain-text template to a single sample (legacy support)."""
    return template.format_map(sample)
