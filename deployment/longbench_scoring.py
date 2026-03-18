import re
import string
from collections import Counter


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
