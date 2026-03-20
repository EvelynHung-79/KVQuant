import re
import string
from collections import Counter

from rouge import Rouge as _Rouge
from fuzzywuzzy import fuzz as _fuzz

_rouge_scorer = _Rouge()


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
    """ROUGE-L F1 via py-rouge, consistent with LongBench official evaluation."""
    if not prediction.strip() or not ground_truth.strip():
        return 0.0
    try:
        scores = _rouge_scorer.get_scores(prediction, ground_truth)
        return scores[0]["rouge-l"]["f"]
    except Exception:
        return 0.0


def edit_sim_score(prediction, ground_truth):
    """Code edit similarity via fuzz.ratio, consistent with LongBench/FastKV official evaluation.

    Post-processing: extract the first valid code line (skip backtick/comment lines),
    then compute fuzz.ratio against ground truth.
    """
    all_lines = prediction.lstrip('\n').split('\n')
    prediction = next(
        (l for l in all_lines if '`' not in l and '#' not in l and '//' not in l),
        ""
    )
    return _fuzz.ratio(prediction, ground_truth) / 100


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
