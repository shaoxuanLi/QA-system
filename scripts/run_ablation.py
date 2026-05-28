"""一键消融实验：在 dev.jsonl 上跑多组配置，输出 EM/F1 对比表。

支持的轴：
  - rerank   : on / off
  - top_k_retrieve : BM25 召回数
  - top_k_final    : 进入 LLM 的段落数

默认网格 (8 组)：
  rerank ∈ {off, on} × top_k_final ∈ {1, 3, 5} (rerank=off 时 top_k_retrieve 也等于 top_k_final)
  外加 retrieve∈{50, 100} × rerank=on, final=5 看召回数影响

Usage:
    python -m scripts.run_ablation --config config.yaml --out ablation.md
可加 --limit 200 先用 200 条 dev 调通。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import evaluate
from src.generator import SiliconFlowLLM
from src.rag import RAGPipeline
from src.reranker import CrossEncoderReranker
from src.retriever import BM25Retriever
from src.utils.io import CorpusReader, iter_jsonl, load_config


@dataclass
class Run:
    name: str
    rerank: bool
    top_k_retrieve: int
    top_k_final: int


DEFAULT_GRID: list[Run] = [
    # 主对比：BM25 only vs +Reranker（同 final）
    Run("bm25-only-top1",      rerank=False, top_k_retrieve=1,   top_k_final=1),
    Run("bm25-only-top3",      rerank=False, top_k_retrieve=3,   top_k_final=3),
    Run("bm25-only-top5",      rerank=False, top_k_retrieve=5,   top_k_final=5),
    Run("bm25+rerank-top1",    rerank=True,  top_k_retrieve=20,  top_k_final=1),
    Run("bm25+rerank-top3",    rerank=True,  top_k_retrieve=20,  top_k_final=3),
    Run("bm25+rerank-top5",    rerank=True,  top_k_retrieve=20,  top_k_final=5),
    # 召回数影响（reranker on, final=5）
    Run("bm25+rerank-recall50",  rerank=True, top_k_retrieve=50,  top_k_final=5),
    Run("bm25+rerank-recall100", rerank=True, top_k_retrieve=100, top_k_final=5),
]


def _build_llm(cfg: dict) -> SiliconFlowLLM:
    g = cfg["generator"]
    return SiliconFlowLLM(
        api_url=g["api_url"], model=g["model"],
        api_key=g.get("api_key", ""), api_key_env=g.get("api_key_env", "SILICONFLOW_API_KEY"),
        max_tokens=g.get("max_tokens", 64), temperature=g.get("temperature", 0.0),
        frequency_penalty=g.get("frequency_penalty", 0.0),
        timeout=g.get("timeout", 60), max_retries=g.get("max_retries", 5),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="ablation.md", help="Markdown 结果表输出路径")
    parser.add_argument("--predictions-dir", default="outputs/ablation",
                        help="把每组的预测落盘到这里，方便复现 / case study")
    parser.add_argument("--limit", type=int, default=None,
                        help="只用前 N 条 dev 调通（建议先 50/200 跑一轮）")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 加载 dev
    rows = list(iter_jsonl(cfg["data"]["dev"]))
    if args.limit:
        rows = rows[: args.limit]
    questions = [r["question"] for r in rows]
    references = [r["answers"] for r in rows]
    print(f"[ablation] N={len(questions)}  runs={len(DEFAULT_GRID)}")

    # 共用：retriever / corpus / reranker / llm 全部跨 run 共用，避免重复加载 9.4GB BM25 索引
    print("[ablation] loading shared retriever / corpus / reranker / llm ...")
    shared_retriever = BM25Retriever(cfg["index"]["dir"])
    shared_corpus = CorpusReader(cfg["data"]["corpus"], cfg["index"]["offsets"])
    r_cfg = cfg["reranker"]
    shared_reranker = CrossEncoderReranker(
        model_name=r_cfg["model"],
        device=r_cfg.get("device", "cpu"),
        batch_size=r_cfg.get("batch_size", 32),
    )
    shared_llm = _build_llm(cfg)
    num_workers = cfg["inference"].get("num_workers", 4)

    Path(args.predictions_dir).mkdir(parents=True, exist_ok=True)
    rows_out: list[dict] = []

    for run in DEFAULT_GRID:
        print(f"\n=== {run.name}  rerank={run.rerank}  "
              f"retrieve={run.top_k_retrieve}  final={run.top_k_final} ===")

        pipe = RAGPipeline(
            retriever=shared_retriever,
            corpus=shared_corpus,
            llm=shared_llm,
            reranker=shared_reranker if run.rerank else None,
            retrieve_top_k=run.top_k_retrieve,
            final_top_k=run.top_k_final,
        )

        t0 = time.time()
        results = pipe.answer_batch(questions, num_workers=num_workers)
        elapsed = time.time() - t0
        preds = [r.answer for r in results]
        metrics = evaluate(preds, references)

        # 落盘
        pred_path = Path(args.predictions_dir) / f"{run.name}.txt"
        pred_path.write_text(
            "\n".join(p.replace("\n", " ").strip() for p in preds) + "\n",
            encoding="utf-8",
        )

        rows_out.append({
            "name": run.name,
            "rerank": run.rerank,
            "top_k_retrieve": run.top_k_retrieve,
            "top_k_final": run.top_k_final,
            "EM": metrics["EM"] * 100,
            "F1": metrics["F1"] * 100,
            "N":  metrics["N"],
            "seconds": elapsed,
        })
        print(f"  -> EM={metrics['EM']*100:.2f}  F1={metrics['F1']*100:.2f}  "
              f"(took {elapsed:.1f}s)")

    shared_corpus.close()

    # 输出 markdown 表
    md = ["# Ablation on dev.jsonl\n",
          f"_N = {rows_out[0]['N']}_\n",
          "| Run | Reranker | top_k_retrieve | top_k_final | EM | F1 | Time(s) |",
          "|---|---|---|---|---|---|---|"]
    for r in rows_out:
        md.append(
            f"| {r['name']} | {'on' if r['rerank'] else 'off'} "
            f"| {r['top_k_retrieve']} | {r['top_k_final']} "
            f"| {r['EM']:.2f} | {r['F1']:.2f} | {r['seconds']:.1f} |"
        )
    md_str = "\n".join(md) + "\n"
    Path(args.out).write_text(md_str, encoding="utf-8")
    print(f"\n[ablation] table written to {args.out}")

    # 同时 dump JSON 方便后续画图
    json_path = Path(args.out).with_suffix(".json")
    json_path.write_text(json.dumps(rows_out, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"[ablation] json written to {json_path}")


if __name__ == "__main__":
    main()
