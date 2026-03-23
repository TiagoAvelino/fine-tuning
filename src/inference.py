"""Inference utilities for loading and querying a fine-tuned model."""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(base_model: str, adapter_path: str | None = None):
    """Load a base model with an optional LoRA adapter merged in."""
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path or base_model, trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )
    return response.strip()


def merge_and_export(base_model: str, adapter_path: str, output_path: str):
    """Merge LoRA adapter into the base model and save the full model."""
    model, tokenizer = load_model(base_model, adapter_path)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Merged model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned model")
    parser.add_argument("--base-model", required=True, help="Base model name or path")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter")
    parser.add_argument("--prompt", required=True, help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    model, tokenizer = load_model(args.base_model, args.adapter)
    result = generate(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print(result)
