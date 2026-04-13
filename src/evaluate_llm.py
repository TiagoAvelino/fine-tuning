"""Evaluate a base LLM vs a fine-tuned (LoRA) LLM on OpenShift troubleshooting.

Loads N test samples from the JSONL dataset, runs inference on both
the base model and the fine-tuned model, and scores each response on:
  - json_valid:    did the model produce parseable JSON?
  - has_category:  does the JSON contain a valid category field?
  - has_sub_cause: does the JSON contain a sub_cause field?
  - has_fix:       does the JSON contain a non-empty fix list?
  - has_commands:  does the JSON contain a non-empty commands list?
  - category_match: does the category match the ground truth?
  - sub_cause_match: does the sub_cause match the ground truth?

Outputs eval_comparison.json with per-sample results and summary scores.

Usage:
    python -m src.evaluate_llm \
        --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --adapter outputs/ocp-tinyllama-qlora/final \
        --dataset data/ocp-instructions.jsonl \
        --output-dir outputs/eval \
        --num-samples 50
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an OpenShift and Kubernetes troubleshooting expert. "
    "Given a problem description, diagnose the issue and respond with a JSON object "
    "containing: category, sub_cause, confidence, explanation, fix, commands, verification."
)


def load_model(model_name, adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path or model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, instruction, max_new_tokens=512):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    elapsed = time.time() - start

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()

    return response, elapsed


def parse_json(text):
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


def score_response(parsed, ground_truth):
    """Score a parsed response against ground truth."""
    if parsed is None:
        return {
            "json_valid": False,
            "has_category": False,
            "has_sub_cause": False,
            "has_fix": False,
            "has_commands": False,
            "category_match": False,
            "sub_cause_match": False,
        }

    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth

    return {
        "json_valid": True,
        "has_category": bool(parsed.get("category")),
        "has_sub_cause": bool(parsed.get("sub_cause")),
        "has_fix": isinstance(parsed.get("fix"), list) and len(parsed["fix"]) > 0,
        "has_commands": isinstance(parsed.get("commands"), list) and len(parsed["commands"]) > 0,
        "category_match": (parsed.get("category") or "").lower() == (gt.get("category") or "").lower(),
        "sub_cause_match": (parsed.get("sub_cause") or "").lower() == (gt.get("sub_cause") or "").lower(),
    }


def load_test_samples(dataset_path, num_samples, seed=42):
    samples = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    rng = random.Random(seed)
    rng.shuffle(samples)
    return samples[:num_samples]


def evaluate_model(model, tokenizer, samples, label):
    """Run evaluation on a set of samples, return per-sample results."""
    results = []
    for i, sample in enumerate(samples):
        instruction = sample["instruction"]
        ground_truth = sample["response"]

        raw, elapsed = generate_response(model, tokenizer, instruction)
        parsed = parse_json(raw)
        scores = score_response(parsed, ground_truth)

        results.append({
            "index": i,
            "instruction": instruction[:200],
            "raw_response": raw[:500],
            "scores": scores,
            "latency_seconds": round(elapsed, 2),
        })

        logger.info(
            "[%s] %d/%d — json=%s cat_match=%s sub_match=%s (%.1fs)",
            label, i + 1, len(samples),
            scores["json_valid"], scores["category_match"],
            scores["sub_cause_match"], elapsed,
        )

    return results


def summarize(results):
    n = len(results)
    if n == 0:
        return {}
    metrics = {}
    for key in results[0]["scores"]:
        count = sum(1 for r in results if r["scores"][key])
        metrics[key] = {"count": count, "total": n, "rate": round(count / n, 4)}
    avg_latency = sum(r["latency_seconds"] for r in results) / n
    metrics["avg_latency_seconds"] = round(avg_latency, 2)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned LLM")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", default=None, help="LoRA adapter path (skip for base-only eval)")
    parser.add_argument("--dataset", required=True, help="JSONL dataset path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    samples = load_test_samples(args.dataset, args.num_samples, args.seed)
    logger.info("Loaded %d test samples", len(samples))

    # --- Base model ---
    logger.info("=== Evaluating BASE model: %s ===", args.base_model)
    base_model, base_tok = load_model(args.base_model)
    base_results = evaluate_model(base_model, base_tok, samples, "BASE")
    base_summary = summarize(base_results)
    del base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- Fine-tuned model ---
    ft_results = None
    ft_summary = None
    if args.adapter:
        logger.info("=== Evaluating FINE-TUNED model: %s + %s ===", args.base_model, args.adapter)
        ft_model, ft_tok = load_model(args.base_model, args.adapter)
        ft_results = evaluate_model(ft_model, ft_tok, samples, "FINE-TUNED")
        ft_summary = summarize(ft_results)
        del ft_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- Comparison ---
    comparison = {
        "num_samples": len(samples),
        "base_model": args.base_model,
        "adapter": args.adapter,
        "base_summary": base_summary,
    }
    if ft_summary:
        comparison["finetuned_summary"] = ft_summary
        comparison["improvement"] = {}
        for key in base_summary:
            if key == "avg_latency_seconds":
                continue
            base_rate = base_summary[key]["rate"]
            ft_rate = ft_summary[key]["rate"]
            comparison["improvement"][key] = {
                "base": base_rate,
                "finetuned": ft_rate,
                "delta": round(ft_rate - base_rate, 4),
            }

    with open(out / "eval_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    details = {"base": base_results}
    if ft_results:
        details["finetuned"] = ft_results
    with open(out / "eval_details.json", "w") as f:
        json.dump(details, f, indent=2)

    # --- Print summary ---
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY (%d samples)", len(samples))
    logger.info("=" * 60)
    header = f"{'Metric':<20} {'Base':>10}"
    if ft_summary:
        header += f" {'Fine-tuned':>12} {'Delta':>10}"
    logger.info(header)
    logger.info("-" * len(header))
    for key in base_summary:
        if key == "avg_latency_seconds":
            continue
        line = f"{key:<20} {base_summary[key]['rate']:>10.1%}"
        if ft_summary:
            line += f" {ft_summary[key]['rate']:>12.1%}"
            delta = ft_summary[key]["rate"] - base_summary[key]["rate"]
            sign = "+" if delta >= 0 else ""
            line += f" {sign}{delta:>9.1%}"
        logger.info(line)

    logger.info("-" * len(header))
    logger.info(f"{'avg_latency':<20} {base_summary['avg_latency_seconds']:>9.1f}s")
    if ft_summary:
        logger.info(f"{'':20} {ft_summary['avg_latency_seconds']:>12.1f}s")

    logger.info("\nResults saved to %s", out)


if __name__ == "__main__":
    main()
