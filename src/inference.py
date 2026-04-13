"""Inference utilities for loading and querying a fine-tuned model.

Supports both:
  - LoRA adapter inference (base model + adapter path)
  - Merged model inference (single model path)

Usage:
    python -m src.inference \
        --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --adapter outputs/llm-sft/final \
        --prompt "Pod my-app in namespace production is in CrashLoopBackOff"

    python -m src.inference \
        --base-model outputs/merged-model \
        --prompt "Route returns 503 for service frontend"
"""

import argparse
import json
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.data import build_chat_messages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_model(base_model: str, adapter_path: str | None = None):
    """Load a base model with an optional LoRA adapter merged in."""
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path:
        logger.info("Loading LoRA adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded: %s params", f"{total_params:,}")
    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    """Generate a response for an OpenShift troubleshooting prompt.

    Formats the prompt as a chat conversation using the model's template
    before generating.
    """
    messages = build_chat_messages(prompt, response=None)
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

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


def parse_structured_response(raw_response: str) -> dict | None:
    """Attempt to parse the model's response as structured JSON."""
    text = raw_response.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


def diagnose(model, tokenizer, problem: str, **kwargs) -> dict:
    """High-level API: submit a problem, get a structured diagnosis."""
    raw = generate(model, tokenizer, problem, **kwargs)
    parsed = parse_structured_response(raw)
    return {
        "raw_response": raw,
        "structured": parsed,
        "parse_success": parsed is not None,
    }


def merge_and_export(base_model: str, adapter_path: str, output_path: str):
    """Merge LoRA adapter into the base model and save the full model."""
    model, tokenizer = load_model(base_model, adapter_path)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info("Merged model saved to %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned model")
    parser.add_argument("--base-model", required=True, help="Base model name or path")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter")
    parser.add_argument("--prompt", required=True, help="OpenShift problem description")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--merge-to", default=None, help="If set, merge adapter and save to this path")
    args = parser.parse_args()

    if args.merge_to:
        merge_and_export(args.base_model, args.adapter, args.merge_to)
    else:
        model, tokenizer = load_model(args.base_model, args.adapter)
        result = diagnose(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print("\n=== Raw Response ===")
        print(result["raw_response"])
        if result["parse_success"]:
            print("\n=== Parsed Diagnosis ===")
            print(json.dumps(result["structured"], indent=2))
        else:
            print("\n(Could not parse as JSON)")
