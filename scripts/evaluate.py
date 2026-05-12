"""把 dev.txt 与 dev.jsonl 对齐，算 EM / F1。

Usage:
    python -m scripts.evaluate --pred outputs/dev.txt --gold dev.jsonl
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import evaluate
from src.utils.io import iter_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="预测答案文件，每行一个答案")
    parser.add_argument("--gold", required=True, help="带 answers 字段的 jsonl 文件")
    args = parser.parse_args()

    with open(args.pred, "r", encoding="utf-8") as f:
        preds = [line.rstrip("\n") for line in f]

    refs = []
    for row in iter_jsonl(args.gold):
        golds = row.get("answers")
        if not golds:
            raise ValueError(f"{args.gold} 中存在没有 answers 字段的行，无法评估")
        refs.append(golds)

    if len(preds) != len(refs):
        raise ValueError(
            f"行数不一致: pred={len(preds)} vs gold={len(refs)}"
        )

    metrics = evaluate(preds, refs)
    print(f"N   = {metrics['N']}")
    print(f"EM  = {metrics['EM']*100:.2f}")
    print(f"F1  = {metrics['F1']*100:.2f}")


if __name__ == "__main__":
    main()
