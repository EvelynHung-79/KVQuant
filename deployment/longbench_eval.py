"""
LongBench Evaluation for KVQuant

Data folder: ../data/longbench_v1/<task>.jsonl

Usage:
    CUDA_VISIBLE_DEVICES=0 python longbench_eval.py \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --task narrativeqa \
        --bits 4 \
        --quantizer-path quantizers.pickle \
        --include_sparse \
        --first_few_fp16 1 \
        --output-path results/narrativeqa.json
"""

import argparse
import json
import os
import pickle
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Load custom KVQuant transformers and quant_cuda before any `from transformers` import
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "transformers", "src"))

# quant_cuda: add build directory to sys.path so `import quant_cuda` works
import glob as _glob
_cuda_builds = _glob.glob(os.path.join(_here, "kvquant", "build", "lib.*", "quant_cuda*.so"))
if _cuda_builds:
    sys.path.insert(0, os.path.dirname(_cuda_builds[0]))

import numpy as np
import torch
from tqdm import tqdm

from kvquant_model import get_model, load_quantizers, run_inference
from longbench_scoring import score_sample

import importlib.util as _ilu

def _load_data_module(name):
    path = os.path.join(_here, "..", "data", f"{name}.py")
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_lb_data = _load_data_module("longbench_data")
build_prompt = _lb_data.build_prompt
load_dataset = _lb_data.load_dataset
COMPLETION_TASKS = _lb_data.COMPLETION_TASKS

_lb_const = _load_data_module("longbench_constants")
TASK_METRICS = _lb_const.TASK_METRICS
TASK_OUTPUT_LEN = _lb_const.TASK_OUTPUT_LEN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Model path or HuggingFace ID")
    parser.add_argument("--task", type=str, required=True,
                        help="LongBench task name (e.g. narrativeqa)")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "data", "longbench_v1"),
                        help="Directory containing <task>.jsonl files")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Path to save JSON result (default: results/<task>.json)")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 16],
                        help="KV cache quantization bits (16 = no quantization)")
    parser.add_argument("--quantizer-path", type=str, default="quant/quantizers.pickle",
                        help="Path to quantizers.pickle")
    parser.add_argument("--include_sparse", action="store_true", default=True,
                        help="Use dense-and-sparse quantization")
    parser.add_argument("--sparsity-threshold", type=float, default=0.99,
                        help="Outlier percentile threshold")
    parser.add_argument("--first_few_fp16", type=int, default=1,
                        help="Keep first N tokens in fp16")
    parser.add_argument("--norm", action="store_true",
                        help="Use q-norm")
    parser.add_argument("--num-samples", type=int, default=-1,
                        help="Number of samples to evaluate (-1 = all)")
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="Prefill chunk size in tokens")
    parser.add_argument("--maxseqlen", type=int, default=32768,
                        help="Max sequence length (KV cache size)")
    args = parser.parse_args()

    DEV = torch.device("cuda:0")

    # ── Output path ──────────────────────────────────────────────────────────
    if args.output_path is None:
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_path = f"results/{timestamp}_{args.task}.json"

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

    output_len = TASK_OUTPUT_LEN.get(args.task, 64)
    print(f"  output_len: {output_len} tokens")

    metric_name = TASK_METRICS.get(args.task, "F1")
    max_input_tokens = args.maxseqlen - output_len - 10

    # ── Evaluate ─────────────────────────────────────────────────────────────
    details = []
    scores = []

    for idx, sample in enumerate(tqdm(samples, desc="Processing samples")):
        prompt = build_prompt(sample, args.task, tokenizer, max_input_tokens)
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids

        answers = sample.get("answers", sample.get("answer", []))
        if isinstance(answers, str):
            answers = [answers]

        output_text, prefill_ms, decode_ms, peak_mb = run_inference(
            model, tokenizer, input_ids, output_len, args.chunk_size, DEV,
            stop_on_newline=(args.task in COMPLETION_TASKS)
        )

        if args.task in {"trec", "triviaqa", "samsum"}:
            output_text = output_text.lstrip('\n').split('\n')[0]

        score = score_sample(output_text, answers, metric_name)

        # print(
        #     f"[{idx}] "
        #     f"score={score:.4f}  prefill={prefill_ms:.0f}ms  "
        #     f"decode={decode_ms:.0f}ms  peak={peak_mb:.1f}MB"
        # )
        # print(f"  output   : {output_text[:120]}")
        # print(f"  expected : {answers[0][:120]}")

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
            "output_len": output_len,
            "chunk_size": args.chunk_size,
            "bits": args.bits,
            "include_sparse": args.include_sparse,
            "sparsity_threshold": args.sparsity_threshold,
            "first_few_fp16": args.first_few_fp16,
            "maxseqlen": args.maxseqlen,
        },
        "results": {
            "avg_score": avg_score,
            "avg_end_to_end_latency_ms": avg_e2e,
            "avg_prefill_latency_ms": avg_prefill,
            "avg_decode_latency_ms": avg_decode,
            "max_peak_memory_mb": max_peak,
        },
        "details": details
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"\n=== {args.task} ===")
    print(f"  avg {metric_name}: {avg_score:.4f}")
    print(f"  avg prefill:  {avg_prefill:.1f} ms")
    print(f"  avg decode:   {avg_decode:.1f} ms")
    print(f"  max memory:   {max_peak:.1f} MB")
    print(f"  saved to:     {args.output_path}")


if __name__ == "__main__":
    main()
