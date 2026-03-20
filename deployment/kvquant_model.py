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
    # dynamicrope=True is only valid when rope_scaling is None (uses
    # LlamaRotaryEmbeddingDynamic with 3-arg forward).  Models with
    # rope_scaling (e.g. Llama-3.1 "llama3" type) use the standard
    # 2-arg forward and must have dynamicrope=False.
    config.dynamicrope = config.rope_scaling is None
    attn_impl = "eager" if bits < 16 else "sdpa"
    # Must set _attn_implementation on config BEFORE from_pretrained so that
    # _attn_implementation_internal is non-None.  The HF auto-dispatch logic in
    # _autoset_attn_implementation only respects a user-specified implementation
    # when _attn_implementation_internal is not None; if it is None the getter
    # silently returns "eager" as a fallback, the condition in from_pretrained
    # (config._attn_implementation != kwarg_attn_imp) evaluates to False, and
    # the internal field is never set – causing auto-selection of "sdpa".
    config._attn_implementation = attn_impl
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, torch_dtype=torch.half,
        attn_implementation=attn_impl, device_map="cpu"
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
    Run prefill + decode via a single model.generate() call.
    Returns (output_text, total_ms, 0.0, peak_memory_mb).

    stop_on_newline: if True, stop generation at the first newline token
                     (used for completion-style tasks: trec, samsum, lcc, repobench-p).
    """
    from transformers import StoppingCriteria, StoppingCriteriaList

    reset_kv_cache(model)
    input_ids = input_ids.to(DEV)
    prompt_len = input_ids.shape[1]
    attention_mask = torch.ones((1, prompt_len), device=DEV)

    extra_kwargs = {}
    if stop_on_newline:
        probe = tokenizer.encode(".\n.", add_special_tokens=False)
        newline_ids = frozenset(t for t in probe if '\n' in tokenizer.decode([t]))
        if newline_ids:
            class _NewlineStopper(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    return input_ids[0, -1].item() in newline_ids
            extra_kwargs["stopping_criteria"] = StoppingCriteriaList([_NewlineStopper()])

    # Hook to capture the moment prefill ends (first forward pass through all layers)
    prefill_done = [False]
    t_prefill_end = [None]

    def _mark_prefill(module, inp, out):
        if not prefill_done[0]:
            torch.cuda.synchronize()
            t_prefill_end[0] = time.time()
            prefill_done[0] = True

    hook = model.model.register_forward_hook(_mark_prefill)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    with torch.no_grad():
        gen_out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=output_len,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **extra_kwargs,
        )

    torch.cuda.synchronize()
    t1 = time.time()
    hook.remove()

    prefill_ms = (t_prefill_end[0] - t0) * 1000 if t_prefill_end[0] else 0.0
    decode_ms = (t1 - t_prefill_end[0]) * 1000 if t_prefill_end[0] else (t1 - t0) * 1000
    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    output_text = tokenizer.decode(gen_out[0][prompt_len:].tolist(), skip_special_tokens=True)
    return output_text, prefill_ms, decode_ms, peak_mb
