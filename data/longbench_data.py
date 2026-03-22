import json
import os
import importlib.util as _ilu

_here = os.path.dirname(os.path.abspath(__file__))
_lb_const_path = os.path.join(_here, "longbench_constants.py")
_lb_const_spec = _ilu.spec_from_file_location("longbench_constants", _lb_const_path)
_lb_const = _ilu.module_from_spec(_lb_const_spec)
_lb_const_spec.loader.exec_module(_lb_const)
TASK_PROMPTS = _lb_const.TASK_PROMPTS


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


COMPLETION_TASKS = {"trec", "samsum", "lcc", "repobench-p"}


def build_prompt(sample, task, tokenizer, max_input_tokens):
    """Build a prompt and truncate context if needed.

    Completion-style tasks (trec, samsum, lcc, repobench-p) use raw text
    completion and must NOT use chat template.
    """
    template = TASK_PROMPTS.get(task)
    if template is None:
        raise ValueError(f"Unsupported task: {task}. Supported: {list(TASK_PROMPTS)}")

    context = sample.get("context", "")
    inp = sample.get("input", "")

    # Truncate context to fit within max_input_tokens (head + tail)
    ctx_tokens = tokenizer.encode(context, add_special_tokens=False)
    inp_tokens = tokenizer.encode(inp, add_special_tokens=False)
    max_ctx = max_input_tokens - len(inp_tokens) - 300
    if max_ctx < 0:
        max_ctx = 0
    if len(ctx_tokens) > max_ctx:
        half = max_ctx // 2
        ctx_tokens = ctx_tokens[:half] + ctx_tokens[len(ctx_tokens) - (max_ctx - half):]
        context = tokenizer.decode(ctx_tokens, skip_special_tokens=True)

    user_content = template.format(context=context, input=inp)

    if task in COMPLETION_TASKS:
        return user_content

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
