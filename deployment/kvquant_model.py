import pickle
import time

import torch


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


def run_inference(model, tokenizer, input_ids, output_len, chunk_size, DEV, stop_on_newline=False):
    """
    Prefill the full prompt, then decode token-by-token using HF DynamicCache.
    Returns (output_text, prefill_ms, decode_ms, peak_memory_mb).

    stop_on_newline: if True, stop generation at the first newline token
                     (used for completion-style tasks: trec, samsum, lcc, repobench-p).
    """
    reset_kv_cache(model)
    input_ids = input_ids.to(DEV)
    prompt_len = input_ids.shape[1]
    attention_mask = torch.ones((1, prompt_len), device=DEV)

    # Find newline token IDs once
    if stop_on_newline:
        probe = tokenizer.encode(".\n.", add_special_tokens=False)
        newline_ids = set(t for t in probe if '\n' in tokenizer.decode([t]))
    else:
        newline_ids = set()

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
            tok = next_token.item()
            if tok == tokenizer.eos_token_id:
                break
            if stop_on_newline and tok in newline_ids:
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
