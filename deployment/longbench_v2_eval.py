"""
LongBench v2 Evaluation for KVQuant

Data folder: ../data/longbench_v2/<task>.jsonl
Tasks (domain names): single-doc, multi-doc, long-context, dialogue, code, structured

Usage:
    CUDA_VISIBLE_DEVICES=0 python longbench_v2_eval.py \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --task single-doc \
        --bits 4 \
        --quantizer-path quantizers.pickle
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

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "transformers", "src"))

import glob as _glob
_cuda_builds = _glob.glob(os.path.join(_here, "kvquant", "build", "lib.*", "quant_cuda*.so"))
if _cuda_builds:
    sys.path.insert(0, os.path.dirname(_cuda_builds[0]))

import numpy as np
import torch
from tqdm import tqdm

from kvquant_model import get_model, load_quantizers, run_inference

import importlib.util as _ilu


def _load_data_module(name):
    path = os.path.join(_here, "..", "data", f"{name}.py")
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_lb_v2_data = _load_data_module("longbench_v2_data")
build_prompt_v2 = _lb_v2_data.build_prompt_v2
load_dataset_v2 = _lb_v2_data.load_dataset_v2

_lb_v2_const = _load_data_module("longbench_v2_constants")
TASK_METRICS_V2 = _lb_v2_const.TASK_METRICS_V2
TASK_OUTPUT_LEN_V2 = _lb_v2_const.TASK_OUTPUT_LEN_V2
V2_DOMAIN_MAP = _lb_v2_const.V2_DOMAIN_MAP


def score_mcq(output_text, answer):
    """Return 1.0 if the first letter of output matches the expected answer letter."""
    pred = output_text.strip().upper()
    if pred:
        return 1.0 if pred[0] == answer.strip().upper() else 0.0
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--task", type=str, required=True,
                        help="LongBench v2 domain task (e.g. single-doc, multi-doc)")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "data", "longbench_v2"),
                        help="Directory containing <task>.jsonl files")
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 16])
    parser.add_argument("--quantizer-path", type=str, default="quant/quantizers.pickle")
    parser.add_argument("--include_sparse", action="store_true", default=True)
    parser.add_argument("--sparsity-threshold", type=float, default=0.99)
    parser.add_argument("--first_few_fp16", type=int, default=1)
    parser.add_argument("--norm", action="store_true")
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--maxseqlen", type=int, default=131072)
    args = parser.parse_args()

    DEV = torch.device("cuda:0")

    if args.output_path is None:
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_path = f"results/{timestamp}_v2_{args.task}.json"

    from transformers import AutoTokenizer
    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {args.model}...")
    model = get_model(
        args.model, args.maxseqlen, args.bits,
        args.include_sparse, args.first_few_fp16
    )
    model.eval()
    model.model.set_devices()
    model.lm_head = model.lm_head.to(DEV)

    if args.bits != 16:
        if args.quantizer_path is None:
            raise ValueError("--quantizer-path is required when --bits != 16")
        print(f"Loading quantizers from {args.quantizer_path}...")
        with open(args.quantizer_path, "rb") as f:
            quantizers = pickle.load(f)
        load_quantizers(model, quantizers, args.bits, args.include_sparse,
                        args.sparsity_threshold, args.norm)

    model = model.half()

    data_dir = os.path.abspath(args.data_dir)
    print(f"Loading dataset from {data_dir}/{args.task}.jsonl...")
    samples = load_dataset_v2(data_dir, args.task)
    if args.num_samples > 0:
        samples = samples[: args.num_samples]
    print(f"  {len(samples)} samples")

    output_len = TASK_OUTPUT_LEN_V2.get(args.task, 4)
    max_input_tokens = args.maxseqlen - output_len - 10

    details = []
    scores = []

    for idx, sample in enumerate(tqdm(samples, desc="Processing samples")):
        prompt = build_prompt_v2(sample, tokenizer, max_input_tokens)
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        if input_ids.shape[1] > max_input_tokens:
            half = max_input_tokens // 2
            input_ids = torch.cat([input_ids[:, :half], input_ids[:, input_ids.shape[1] - (max_input_tokens - half):]], dim=1)

        answer = sample.get("answer", sample.get("answers", ""))
        if isinstance(answer, list):
            answer = answer[0]

        output_text, prefill_ms, decode_ms, peak_mb = run_inference(
            model, tokenizer, input_ids, output_len, args.chunk_size, DEV,
            stop_on_newline=False
        )
        if output_text .startswith("OOM"):
            tqdm.write(f"{output_text} at sample {idx}, with {input_ids.shape[1]} tokens.")

        score = score_mcq(output_text, answer)
        is_oom = output_text .startswith("OOM")
        scores.append(score)
        details.append({
            "index": idx,
            "score": score,
            "metric": "accuracy",
            "output": output_text,
            "ground_truth": answer,
            "peak_memory_mb": peak_mb,
            "end_to_end_latency_ms": (prefill_ms + decode_ms) if (prefill_ms is not None and decode_ms is not None) else None,
            "prefill_latency_ms": prefill_ms,
            "decode_latency_ms": decode_ms,
        })

    num_oom = sum(1 for d in details if d["output"].startswith("OOM"))
    valid_scores = [d["score"] for d in details if not d["output"].startswith("OOM")]
    avg_score = float(np.mean(valid_scores)) if valid_scores else 0.0
    _e2e = [d["end_to_end_latency_ms"] for d in details if d["end_to_end_latency_ms"] is not None]
    _prefill = [d["prefill_latency_ms"] for d in details if d["prefill_latency_ms"] is not None]
    _decode = [d["decode_latency_ms"] for d in details if d["decode_latency_ms"] is not None]
    avg_e2e = float(np.mean(_e2e)) if _e2e else 0.0
    avg_prefill = float(np.mean(_prefill)) if _prefill else 0.0
    avg_decode = float(np.mean(_decode)) if _decode else 0.0
    max_peak = float(max((d["peak_memory_mb"] for d in details), default=0.0))

    domain_label = V2_DOMAIN_MAP.get(args.task, args.task)
    result = {
        "task": args.task,
        "version": "v2",
        "domain": domain_label,
        "args": {
            "input_mode": "longbench_v2",
            "model_id": args.model,
            "bench_version": "v2",
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
            "avg_accuracy": avg_score,
            "num_oom": num_oom,
            "avg_end_to_end_latency_ms": avg_e2e,
            "avg_prefill_latency_ms": avg_prefill,
            "avg_decode_latency_ms": avg_decode,
            "max_peak_memory_mb": max_peak,
        },
        "details": details,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"\n=== {args.task} ({domain_label}) ===")
    print(f"  avg accuracy: {avg_score:.4f}")
    print(f"  avg prefill:  {avg_prefill:.1f} ms")
    print(f"  avg decode:   {avg_decode:.1f} ms")
    print(f"  max memory:   {max_peak:.1f} MB")
    print(f"  saved to:     {args.output_path}")


if __name__ == "__main__":
    main()
