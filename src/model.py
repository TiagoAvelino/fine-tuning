"""Model loading with quantization and LoRA adapter configuration."""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)


def load_tokenizer(model_name: str) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_base_model(config: dict) -> PreTrainedModel:
    """Load the base model with optional quantization (4-bit or 8-bit)."""
    model_name = config["model"]["name"]
    quant_config = config["model"].get("quantization", {})
    quant_mode = quant_config.get("mode")

    bnb_config = None
    if quant_mode == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_config.get("quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(
                torch, quant_config.get("compute_dtype", "bfloat16")
            ),
            bnb_4bit_use_double_quant=quant_config.get("double_quant", True),
        )
    elif quant_mode == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if bnb_config is None else None,
        trust_remote_code=True,
        attn_implementation=config["model"].get("attn_implementation", "sdpa"),
    )

    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    return model


def apply_lora(model: PreTrainedModel, config: dict) -> PreTrainedModel:
    """Wrap the model with a LoRA adapter."""
    lora_cfg = config["lora"]

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get(
            "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
        bias=lora_cfg.get("bias", "none"),
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model
