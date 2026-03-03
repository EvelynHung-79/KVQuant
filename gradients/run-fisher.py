import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from tqdm import tqdm


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    load: Optional[str] = field(default="")


@dataclass
class DataArguments:
    dataset: str = field(default="c4")
    num_examples: int = field(default=16, metadata={"help": "Number of calibration examples"})
    seqlen: int = field(default=2048)
    maxseqlen: int = field(default=32768)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length."},
    )


def get_modules_kv(layer):
    # NOTE: This is llama-specific.
    # For other models, replace with the appropriate k/v projection names.
    return layer.self_attn.k_proj, layer.self_attn.v_proj


def make_retain_grad_hook():
    """Forward hook that retains the output gradient and saves the activation."""
    def hook(module, inp, out):
        out.retain_grad()
        module.act = out
    return hook


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.dataset in ("c4", "wikitext2"):
        from datautils import get_loaders
        print(f"Calibration with {data_args.dataset}")
        dataloader, _ = get_loaders(
            data_args.dataset,
            model=model_args.model_name_or_path,
            seqlen=data_args.seqlen,
            seed=0,
        )
    else:
        raise NotImplementedError("Please define your own dataset here")

    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)

    # Extend RoPE if the requested context is longer than the model's default
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and data_args.maxseqlen > orig_ctx_len:
        scaling_factor = float(math.ceil(data_args.maxseqlen / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    config._flash_attn_2_enabled = True

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )

    if config.vocab_size == 32001:
        model.resize_token_embeddings(32001)

    model = model.bfloat16()

    if model_args.load:
        model.load_state_dict(torch.load(model_args.load), strict=False)
        model.eval()

    _layers = model.model.layers
    grads = {}

    # Register hooks once — they persist across all calibration samples
    handles = []
    for layer in _layers:
        k_proj, v_proj = get_modules_kv(layer)
        handles.append(k_proj.register_forward_hook(make_retain_grad_hook()))
        handles.append(v_proj.register_forward_hook(make_retain_grad_hook()))

    for sample_idx, data in tqdm(enumerate(dataloader[:data_args.num_examples])):
        x = data[0].cuda()
        loss = model(input_ids=x, labels=x).loss
        loss.backward()

        for layer_idx, layer in enumerate(_layers):
            print(f"weight layer {layer_idx}")
            k_proj, v_proj = get_modules_kv(layer)
            kgrad = (k_proj.act.grad ** 2).float().cpu()
            vgrad = (v_proj.act.grad ** 2).float().cpu()

            if f"k_proj{layer_idx}" not in grads:
                grads[f"k_proj{layer_idx}"] = kgrad
            else:
                grads[f"k_proj{layer_idx}"] = torch.cat((grads[f"k_proj{layer_idx}"], kgrad), dim=1)

            if f"v_proj{layer_idx}" not in grads:
                grads[f"v_proj{layer_idx}"] = vgrad
            else:
                grads[f"v_proj{layer_idx}"] = torch.cat((grads[f"v_proj{layer_idx}"], vgrad), dim=1)

    for h in handles:
        h.remove()

    # Overwrite model weights with accumulated gradients, then use save_pretrained
    # to serialise them in the standard HF format.
    for layer_idx, layer in enumerate(_layers):
        k_proj, v_proj = get_modules_kv(layer)
        k_proj.weight.data = grads[f"k_proj{layer_idx}"]
        v_proj.weight.data = grads[f"v_proj{layer_idx}"]

    print(f"Saving model gradients to {training_args.output_dir}")
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
