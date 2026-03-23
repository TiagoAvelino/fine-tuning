"""Dataset loading and preprocessing for LLM fine-tuning."""

from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer


def load_training_data(config: dict) -> Dataset:
    """Load dataset from local files or HuggingFace Hub.

    Supports JSON/JSONL files from local path or S3-backed storage,
    and datasets from the HuggingFace Hub.
    """
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

    return dataset


def format_instruction(sample: dict, template: str) -> str:
    """Apply a chat/instruction template to a single sample.

    The template uses Python format-string syntax, e.g.:
      "### Instruction:\\n{instruction}\\n### Response:\\n{output}"
    """
    return template.format_map(sample)


def preprocess_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    config: dict,
) -> Dataset:
    """Tokenize and format the dataset for SFT training."""
    max_length = config["training"].get("max_seq_length", 2048)
    template = config["dataset"].get("template")

    def tokenize(examples):
        if template:
            texts = [format_instruction(ex, template) for ex in _iter_rows(examples)]
        else:
            text_field = config["dataset"].get("text_field", "text")
            texts = examples[text_field]

        return tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    return dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )


def _iter_rows(batch: dict) -> list[dict]:
    """Convert a columnar batch dict into a list of row dicts."""
    keys = list(batch.keys())
    return [
        {k: batch[k][i] for k in keys}
        for i in range(len(batch[keys[0]]))
    ]
