"""SQuAD 风格的 EM / F1 评估。

参考自 SQuAD v1.1 / HotpotQA 官方脚本：
- 小写化
- 去除冠词 (a, an, the)
- 去除标点
- 折叠多余空白
F1 基于 token bag 计算，多答案取 max。
"""
from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pred_toks)
    r = num_same / len(gold_toks)
    return 2 * p * r / (p + r)


def metric_max_over_ground_truths(metric_fn, pred: str, golds: list[str]) -> float:
    return max(metric_fn(pred, g) for g in golds)


def evaluate(predictions: list[str], references: list[list[str]]) -> dict:
    assert len(predictions) == len(references)
    em_total = 0.0
    f1_total = 0.0
    for pred, golds in zip(predictions, references):
        em_total += metric_max_over_ground_truths(exact_match_score, pred, golds)
        f1_total += metric_max_over_ground_truths(f1_score, pred, golds)
    n = len(predictions)
    return {
        "EM": em_total / n if n else 0.0,
        "F1": f1_total / n if n else 0.0,
        "N":  n,
    }
