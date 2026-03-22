import json
import os
import importlib.util as _ilu

def _load_constants():
    path = os.path.join(os.path.dirname(__file__), "longbench_v2_constants.py")
    spec = _ilu.spec_from_file_location("longbench_v2_constants", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_const = _load_constants()
_DOMAIN_MAP = {k: v.replace(" ", "_").replace("-", "_") for k, v in _const.V2_DOMAIN_MAP.items()}


def load_dataset_v2(data_dir, task):
    stem = _DOMAIN_MAP.get(task, task)
    # Try .json (list) first, then .jsonl (line-delimited)
    for ext in (".json", ".jsonl"):
        path = os.path.join(data_dir, f"{stem}{ext}")
        if os.path.exists(path):
            with open(path) as f:
                if ext == ".json":
                    return json.load(f)
                return [json.loads(line) for line in f if line.strip()]
    raise FileNotFoundError(
        f"Dataset not found for task '{task}' in {data_dir}\n"
        f"Expected: {stem}.json or {stem}.jsonl"
    )


def build_prompt_v2(sample, tokenizer, max_input_tokens):
    """Build an MCQ prompt for LongBench v2.

    Expected sample fields: context, question, choice_A/B/C/D (or nested choices dict).
    """
    context = sample.get("context", "")
    question = sample.get("question", "")

    # Support both flat keys and a nested 'choices' dict
    choices_dict = sample.get("choices", {})
    choice_A = choices_dict.get("choice_A", sample.get("choice_A", ""))
    choice_B = choices_dict.get("choice_B", sample.get("choice_B", ""))
    choice_C = choices_dict.get("choice_C", sample.get("choice_C", ""))
    choice_D = choices_dict.get("choice_D", sample.get("choice_D", ""))

    # Truncate context to fit within max_input_tokens (head + tail)
    ctx_tokens = tokenizer.encode(context, add_special_tokens=False)
    question_tokens = tokenizer.encode(question, add_special_tokens=False)
    choices_text = f"A. {choice_A}\nB. {choice_B}\nC. {choice_C}\nD. {choice_D}"
    choices_tokens = tokenizer.encode(choices_text, add_special_tokens=False)
    max_ctx = max_input_tokens - len(question_tokens) - len(choices_tokens) - 100
    if max_ctx < 0:
        max_ctx = 0
    if len(ctx_tokens) > max_ctx:
        half = max_ctx // 2
        ctx_tokens = ctx_tokens[:half] + ctx_tokens[len(ctx_tokens) - (max_ctx - half):]
        context = tokenizer.decode(ctx_tokens, skip_special_tokens=True)

    choices = {"choice_A": choice_A, "choice_B": choice_B,
               "choice_C": choice_C, "choice_D": choice_D}

    prompt = (
        f"Read the following context and answer the question by choosing the correct option (A, B, C, or D).\n\n"
        f"Please output ONLY the single letter (A, B, C, or D) of the correct option and nothing else.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n"
        f"A. {choices.get('choice_A', '')}\n"
        f"B. {choices.get('choice_B', '')}\n"
        f"C. {choices.get('choice_C', '')}\n"
        f"D. {choices.get('choice_D', '')}\n\n"
        f"Answer:"
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"[INST] {prompt} [/INST]"
