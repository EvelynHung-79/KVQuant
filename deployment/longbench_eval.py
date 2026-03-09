"""
LongBench Evaluation for KVQuant

Data folder: ../data/longbench_v1/<task>.jsonl

Usage:
    CUDA_VISIBLE_DEVICES=0 python longbench_eval.py \
        meta-llama/Meta-Llama-3.1-8B-Instruct \
        --task narrativeqa \
        --bits 4 \
        --quantizer-path quantizers.pickle \
        --include_sparse \
        --sparsity-threshold 0.99 \
        --first_few_fp16 5 \
        --output-path results/narrativeqa.json
"""

import argparse
import json
import os
import pickle
import re
import string
import time
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import numpy as np


# ── Prompt templates (LongBench v1) ─────────────────────────────────────────

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
    "qasper": (
        "You are given a scientific article and a question. "
        "Answer the question as concisely as you can, using a single phrase or sentence if possible. "
        "If the question cannot be answered based on the information in the article, write \"unanswerable\". "
        "If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". "
        "Do not provide any explanation.\n\n"
        "Article: {context}\n\n"
        "Answer the question based on the above article as concisely as you can, "
        "using a single phrase or sentence if possible. "
        "If the question cannot be answered based on the information in the article, write \"unanswerable\". "
        "If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". "
        "Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n"
        "{context}\n\n"
        "Now, answer the following question based on the above text, "
        "only give me the answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. "
        "Write a one-page summary of the report.\n\n"
        "Report:\n{context}\n\n"
        "Now, write a one-page summary of the report.\n\nSummary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question or instruction. "
        "Answer the query in one or more sentences.\n\n"
        "Transcript:\n{context}\n\n"
        "Now, answer the query based on the above meeting transcript in one or more sentences.\n\n"
        "Query: {input}\nAnswer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news passages.\n\n"
        "News:\n{context}\n\n"
        "Now, write a one-page summary of all the news passages.\n\nSummary:"
    ),
    "trec": (
        "Please determine the type of the question below. Here are some examples of questions.\n\n"
        "{context}\n{input}"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. "
        "Only give me the answer and do not output any other words. "
        "The following are some examples.\n\n"
        "{context}\n\n{input}"
    ),
    "samsum": (
        "Summarize the dialogue into a few short sentences. The following are some examples.\n\n"
        "{context}\n\n{input}"
    ),
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. "
        "Some of them may be duplicates. "
        "Please carefully read these paragraphs and determine how many unique paragraphs there are after "
        "removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n"
        "{context}\n\n"
        "Please enter the final count of unique paragraphs after removing duplicates. "
        "The output format should only contain the final count, e.g., 1, 2, 3, ...\n\nThe number of unique paragraphs:"
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. "
        "Please determine which paragraph the abstract is from.\n\n"
        "{context}\n\n"
        "The following is an abstract.\n\n{input}\n\n"
        "Please enter the number of the paragraph that the abstract is from. "
        "The answer format must be like \"Paragraph 3\", \"Paragraph 1\", etc.\n\nThe answer is:"
    ),
    "lcc": (
        "Please complete the code given below.\n{context}Next line of code:\n"
    ),
    "repobench-p": (
        "Please complete the code given below.\n{context}{input}Next line of code:\n"
    ),
}

TASK_METRICS = {
    "narrativeqa": "F1",
    "qasper": "F1",
    "multifieldqa_en": "F1",
    "hotpotqa": "F1",
    "2wikimqa": "F1",
    "musique": "F1",
    "gov_report": "rouge-l",
    "qmsum": "rouge-l",
    "multi_news": "rouge-l",
    "trec": "accuracy",
    "triviaqa": "F1",
    "samsum": "rouge-l",
    "passage_count": "accuracy",
    "passage_retrieval_en": "accuracy",
    "lcc": "edit_sim",
    "repobench-p": "edit_sim",
}


# ── Scoring functions ────────────────────────────────────────────────────────

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


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


def rouge_l_score(prediction, ground_truth):
    """Sentence-level ROUGE-L (F1)."""
    def lcs_length(x, y):
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, gt_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def edit_sim_score(prediction, ground_truth):
    """Normalized edit similarity (1 - edit_distance / max_len)."""
    def edit_distance(s1, s2):
        m, n = len(s1), len(s2)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                if s1[i - 1] == s2[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = temp
        return dp[n]

    if not prediction and not ground_truth:
        return 1.0
    max_len = max(len(prediction), len(ground_truth))
    if max_len == 0:
        return 1.0
    return 1.0 - edit_distance(prediction, ground_truth) / max_len


def accuracy_score(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def score_sample(prediction, answers, metric):
    """Score one prediction against a list of ground truth answers."""
    if metric == "F1":
        return max(f1_score(prediction, ans) for ans in answers)
    elif metric == "rouge-l":
        return max(rouge_l_score(prediction, ans) for ans in answers)
    elif metric == "edit_sim":
        return max(edit_sim_score(prediction, ans) for ans in answers)
    elif metric == "accuracy":
        return max(accuracy_score(prediction, ans) for ans in answers)
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ── Model loading ────────────────────────────────────────────────────────────

def get_model(model_path, maxseqlen, bits, include_sparse, first_few_fp16):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(model_path)
    config.first_few_fp16 = first_few_fp16
    config.maxseqlen = maxseqlen
    config.abits = bits
    config.include_sparse = include_sparse
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, torch_dtype=torch.half,
        attn_implementation="sdpa", device_map="cpu"
    )
    return model


def load_quantizers(model, quantizers, bits, include_sparse, sparsity_threshold, norm):
    layers = model.model.layers
    for k in quantizers.keys():
        if '.lut' in k:
            continue
        ln = int(k.split('.')[-3])
        q = quantizers[k]
        if "k_proj" in k:
            layers[ln].self_attn.kcache.reset()
            layers[ln].self_attn.kcache.load_lookup_table(q, include_sparse, sparsity_threshold, norm)
        elif "v_proj" in k:
            layers[ln].self_attn.vcache.reset()
            layers[ln].self_attn.vcache.load_lookup_table(q, include_sparse, sparsity_threshold, norm)


def reset_kv_cache(model):
    for layer in model.model.layers:
        if layer.self_attn.kcache is not None:
            layer.self_attn.kcache.reset()
        if layer.self_attn.vcache is not None:
            layer.self_attn.vcache.reset()


# ── Inference ────────────────────────────────────────────────────────────────

def run_inference(model, tokenizer, input_ids, output_len, chunk_size, DEV):
    """
    Prefill the full prompt, then decode token-by-token using HF DynamicCache.
    Returns (output_text, prefill_ms, decode_ms, peak_memory_mb).
    """
    input_ids = input_ids.to(DEV)
    prompt_len = input_ids.shape[1]
    attention_mask = torch.ones((1, prompt_len), device=DEV)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # ── Prefill ──────────────────────────────────────────────────────────────
    t0 = time.time()
    with torch.no_grad():
        out = model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
    past_key_values = out.past_key_values
    torch.cuda.synchronize()
    prefill_ms = (time.time() - t0) * 1000

    # ── Decode ───────────────────────────────────────────────────────────────
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token.item()]

    t1 = time.time()
    with torch.no_grad():
        for step in range(1, output_len):
            if next_token.item() == tokenizer.eos_token_id:
                break
            cur_len = prompt_len + step
            attention_mask = torch.ones((1, cur_len), device=DEV)
            out = model(
                next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token.item())
    torch.cuda.synchronize()
    decode_ms = (time.time() - t1) * 1000

    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    output_text = tokenizer.decode(generated, skip_special_tokens=True)
    return output_text, prefill_ms, decode_ms, peak_mb


# ── Dataset loading ──────────────────────────────────────────────────────────

def load_dataset(data_dir, task):
    path = os.path.join(data_dir, f"{task}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Please put {task}.jsonl in {data_dir}"
        )
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples


def build_prompt(sample, task, tokenizer, max_input_tokens):
    """Build a chat-formatted prompt and truncate context if needed."""
    template = TASK_PROMPTS.get(task)
    if template is None:
        raise ValueError(f"Unsupported task: {task}. Supported: {list(TASK_PROMPTS)}")

    context = sample.get("context", "")
    inp = sample.get("input", "")

    # Truncate context to fit within max_input_tokens
    ctx_tokens = tokenizer.encode(context, add_special_tokens=False)
    inp_tokens = tokenizer.encode(inp, add_special_tokens=False)
    # Leave room for template overhead (~200 tokens) and input
    max_ctx = max_input_tokens - len(inp_tokens) - 300
    if max_ctx < 0:
        max_ctx = 0
    if len(ctx_tokens) > max_ctx:
        ctx_tokens = ctx_tokens[:max_ctx]
        context = tokenizer.decode(ctx_tokens, skip_special_tokens=True)

    user_content = template.format(context=context, input=inp)

    # LLaMA-3 chat format
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = f"[INST] {user_content} [/INST]"

    return prompt


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Model path or HuggingFace ID")
    parser.add_argument("--task", type=str, required=True,
                        help="LongBench task name (e.g. narrativeqa)")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "data", "longbench_v1"),
                        help="Directory containing <task>.jsonl files")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Path to save JSON result (default: results/<task>.json)")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 16],
                        help="KV cache quantization bits (16 = no quantization)")
    parser.add_argument("--quantizer-path", type=str, default=None,
                        help="Path to quantizers.pickle")
    parser.add_argument("--include_sparse", action="store_true",
                        help="Use dense-and-sparse quantization")
    parser.add_argument("--sparsity-threshold", type=float, default=0.99,
                        help="Outlier percentile threshold")
    parser.add_argument("--first_few_fp16", type=int, default=0,
                        help="Keep first N tokens in fp16")
    parser.add_argument("--norm", action="store_true",
                        help="Use q-norm")
    parser.add_argument("--num-samples", type=int, default=-1,
                        help="Number of samples to evaluate (-1 = all)")
    parser.add_argument("--output-len", type=int, default=64,
                        help="Max new tokens to generate")
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="Prefill chunk size in tokens")
    parser.add_argument("--maxseqlen", type=int, default=32768,
                        help="Max sequence length (KV cache size)")
    parser.add_argument("--n-warmup", type=int, default=2,
                        help="Number of warmup samples before timing")
    args = parser.parse_args()

    DEV = torch.device("cuda:0")

    # ── Output path ──────────────────────────────────────────────────────────
    if args.output_path is None:
        os.makedirs("results", exist_ok=True)
        args.output_path = f"results/{args.task}.json"

    # ── Load tokenizer ───────────────────────────────────────────────────────
    from transformers import AutoTokenizer
    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"Loading model from {args.model}...")
    model = get_model(
        args.model, args.maxseqlen, args.bits,
        args.include_sparse, args.first_few_fp16
    )
    model.eval()
    model.model.set_devices()
    model.lm_head = model.lm_head.to(DEV)

    # ── Load quantizers ──────────────────────────────────────────────────────
    if args.bits != 16:
        if args.quantizer_path is None:
            raise ValueError("--quantizer-path is required when --bits != 16")
        print(f"Loading quantizers from {args.quantizer_path}...")
        with open(args.quantizer_path, "rb") as f:
            quantizers = pickle.load(f)
        load_quantizers(model, quantizers, args.bits, args.include_sparse,
                        args.sparsity_threshold, args.norm)

    model = model.half()

    # ── Load dataset ─────────────────────────────────────────────────────────
    data_dir = os.path.abspath(args.data_dir)
    print(f"Loading dataset from {data_dir}/{args.task}.jsonl...")
    samples = load_dataset(data_dir, args.task)
    if args.num_samples > 0:
        samples = samples[: args.num_samples]
    print(f"  {len(samples)} samples")

    metric_name = TASK_METRICS.get(args.task, "F1")
    max_input_tokens = args.maxseqlen - args.output_len - 10

    # ── Evaluate ─────────────────────────────────────────────────────────────
    details = []
    scores = []

    for idx, sample in enumerate(samples):
        prompt = build_prompt(sample, args.task, tokenizer, max_input_tokens)
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids

        answers = sample.get("answers", sample.get("answer", []))
        if isinstance(answers, str):
            answers = [answers]

        is_warmup = idx < args.n_warmup

        output_text, prefill_ms, decode_ms, peak_mb = run_inference(
            model, tokenizer, input_ids, args.output_len, args.chunk_size, DEV
        )

        score = score_sample(output_text, answers, metric_name)

        print(
            f"[{'WARMUP ' if is_warmup else ''}{idx}] "
            f"score={score:.4f}  prefill={prefill_ms:.0f}ms  "
            f"decode={decode_ms:.0f}ms  peak={peak_mb:.1f}MB"
        )
        print(f"  output   : {output_text[:120]}")
        print(f"  expected : {answers[0][:120]}")

        if not is_warmup:
            scores.append(score)
            details.append({
                "index": idx,
                "score": score,
                "metric": metric_name,
                "output": output_text,
                "ground_truth": answers[0],
                "peak_memory_mb": peak_mb,
                "end_to_end_latency_ms": prefill_ms + decode_ms,
                "prefill_latency_ms": prefill_ms,
                "decode_latency_ms": decode_ms,
            })

    avg_score = float(np.mean(scores)) if scores else 0.0
    avg_e2e = float(np.mean([d["end_to_end_latency_ms"] for d in details])) if details else 0.0
    avg_prefill = float(np.mean([d["prefill_latency_ms"] for d in details])) if details else 0.0
    avg_decode = float(np.mean([d["decode_latency_ms"] for d in details])) if details else 0.0
    max_peak = float(max((d["peak_memory_mb"] for d in details), default=0.0))

    result = {
        "task": args.task,
        "version": "v1",
        "args": {
            "input_mode": "longbench",
            "model_id": args.model,
            "bench_version": "v1",
            "task_type": args.task,
            "num_samples": args.num_samples,
            "output_len": args.output_len,
            "chunk_size": args.chunk_size,
            "n_warmup": args.n_warmup,
            "bits": args.bits,
            "include_sparse": args.include_sparse,
            "sparsity_threshold": args.sparsity_threshold,
            "first_few_fp16": args.first_few_fp16,
            "maxseqlen": args.maxseqlen,
        },
        "avg_score": avg_score,
        "avg_end_to_end_latency_ms": avg_e2e,
        "avg_prefill_latency_ms": avg_prefill,
        "avg_decode_latency_ms": avg_decode,
        "max_peak_memory_mb": max_peak,
        "details": details,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"\n=== {args.task} ===")
    print(f"  avg {metric_name}: {avg_score:.4f}")
    print(f"  avg prefill:  {avg_prefill:.1f} ms")
    print(f"  avg decode:   {avg_decode:.1f} ms")
    print(f"  max memory:   {max_peak:.1f} MB")
    print(f"  saved to:     {args.output_path}")


if __name__ == "__main__":
    main()
