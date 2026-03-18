"""
Pure HuggingFace fp16 baseline for NarrativeQA.
No KVQuant custom transformers - uses stock transformers.

Usage:
    CUDA_VISIBLE_DEVICES=0 python hf_baseline_eval.py \
        meta-llama/Meta-Llama-3.1-8B-Instruct \
        --task narrativeqa \
        --num-samples 5 \
        --output-path results/narrativeqa_hf_baseline.json
"""

import argparse
import json
import os
import re
import string
import time
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

import torch
import numpy as np


# ── Prompt templates (same as longbench_eval.py) ─────────────────────────────

TASK_PROMPTS = {
    "narrativeqa": (
        "You are given a story, which can be quite long, and a question. "
        "Answer the question as concisely as you can, using a single phrase if possible. "
        "Do not provide any explanation.\n\n"
        "Story: {context}\n\n"
        "Now, answer the question based on the story as concisely as you can, "
        "using a single phrase if possible. Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
}

TASK_METRICS = {
    "narrativeqa": "F1",
}


# ── Scoring ───────────────────────────────────────────────────────────────────

def normalize_answer(s):
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def score_sample(prediction, answers, metric):
    if metric == "F1":
        return max(f1_score(prediction, ans) for ans in answers)
    raise ValueError(f"Unknown metric: {metric}")


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_dataset(data_dir, task):
    path = os.path.join(data_dir, f"{task}.jsonl")
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples


def build_prompt(sample, task, tokenizer, max_input_tokens):
    template = TASK_PROMPTS[task]
    context = sample.get("context", "")
    inp = sample.get("input", "")

    ctx_tokens = tokenizer.encode(context, add_special_tokens=False)
    inp_tokens = tokenizer.encode(inp, add_special_tokens=False)
    max_ctx = max_input_tokens - len(inp_tokens) - 300
    if max_ctx < 0:
        max_ctx = 0
    if len(ctx_tokens) > max_ctx:
        ctx_tokens = ctx_tokens[:max_ctx]
        context = tokenizer.decode(ctx_tokens, skip_special_tokens=True)

    user_content = template.format(context=context, input=inp)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, tokenizer, input_ids, output_len, DEV):
    input_ids = input_ids.to(DEV)
    prompt_len = input_ids.shape[1]

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=output_len,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    torch.cuda.synchronize()
    elapsed_ms = (time.time() - t0) * 1000
    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    new_tokens = output[0][prompt_len:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return output_text, elapsed_ms, peak_mb


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("--task", type=str, default="narrativeqa")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "data", "longbench_v1"))
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--maxseqlen", type=int, default=32768)
    parser.add_argument("--n-warmup", type=int, default=2)
    args = parser.parse_args()

    DEV = torch.device("cuda:0")

    if args.output_path is None:
        os.makedirs("results", exist_ok=True)
        args.output_path = f"results/{args.task}_hf_baseline.json"

    # ── Load tokenizer & model ────────────────────────────────────────────────
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model (fp16, stock HuggingFace)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cuda:0",
    )
    model.eval()
    print(f"  Model loaded on {next(model.parameters()).device}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    data_dir = os.path.abspath(args.data_dir)
    samples = load_dataset(data_dir, args.task)
    if args.num_samples > 0:
        samples = samples[:args.num_samples]
    print(f"  {len(samples)} samples")

    metric_name = TASK_METRICS.get(args.task, "F1")
    max_input_tokens = args.maxseqlen - args.output_len - 10

    # ── Evaluate ──────────────────────────────────────────────────────────────
    details = []
    scores = []

    for idx, sample in enumerate(samples):
        prompt = build_prompt(sample, args.task, tokenizer, max_input_tokens)
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids

        answers = sample.get("answers", sample.get("answer", []))
        if isinstance(answers, str):
            answers = [answers]

        is_warmup = idx < args.n_warmup

        output_text, elapsed_ms, peak_mb = run_inference(
            model, tokenizer, input_ids, args.output_len, DEV
        )
        score = score_sample(output_text, answers, metric_name)

        print(
            f"[{'WARMUP ' if is_warmup else ''}{idx}] "
            f"score={score:.4f}  elapsed={elapsed_ms:.0f}ms  peak={peak_mb:.1f}MB"
        )
        print(f"  output   : {output_text[:120]}")
        print(f"  expected : {answers[0][:80]}")

        if not is_warmup:
            scores.append(score)
            details.append({
                "index": idx,
                "score": score,
                "metric": metric_name,
                "output": output_text,
                "ground_truth": answers[0],
                "peak_memory_mb": peak_mb,
                "elapsed_ms": elapsed_ms,
            })

    avg_score = float(np.mean(scores)) if scores else 0.0
    avg_elapsed = float(np.mean([d["elapsed_ms"] for d in details])) if details else 0.0
    max_peak = float(max((d["peak_memory_mb"] for d in details), default=0.0))

    result = {
        "task": args.task,
        "backend": "pure_hf_fp16",
        "model": args.model,
        "args": vars(args),
        "results": {
            "avg_score": avg_score,
            "avg_elapsed_ms": avg_elapsed,
            "max_peak_memory_mb": max_peak,
        },
        "details": details,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"\n=== {args.task} (pure HF fp16 baseline) ===")
    print(f"  avg {metric_name}: {avg_score:.4f}")
    print(f"  avg elapsed:  {avg_elapsed:.1f} ms")
    print(f"  max memory:   {max_peak:.1f} MB")
    print(f"  saved to:     {args.output_path}")


if __name__ == "__main__":
    main()
