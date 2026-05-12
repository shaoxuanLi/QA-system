"""跑 dev / test 集，输出 dev.txt / test.txt（每行一个答案）。

Usage:
    python -m scripts.run_inference --split dev  --config config.yaml
    python -m scripts.run_inference --split test --config config.yaml

可选开关:
    --no-rerank           关闭重排序器，做消融
    --top-k-retrieve N    覆盖 retriever.top_k
    --top-k-final    N    覆盖 reranker.top_k (=进入 LLM 的段落数)
    --limit          N    只跑前 N 条 (调试用)
    --dump-jsonl PATH     额外把 (q, pred, retrieved) 保存为 jsonl 用于报告
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generator import SiliconFlowLLM
from src.rag import RAGPipeline
from src.reranker import CrossEncoderReranker
from src.retriever import BM25Retriever
from src.utils.io import CorpusReader, iter_jsonl, load_config, write_lines


def build_pipeline(cfg: dict, use_rerank: bool, top_k_retrieve: int, top_k_final: int) -> RAGPipeline:
    retriever = BM25Retriever(cfg["index"]["dir"])
    corpus = CorpusReader(cfg["data"]["corpus"], cfg["index"]["offsets"])

    g = cfg["generator"]
    llm = SiliconFlowLLM(
        api_url=g["api_url"],
        model=g["model"],
        api_key=g.get("api_key", ""),
        api_key_env=g.get("api_key_env", "SILICONFLOW_API_KEY"),
        max_tokens=g.get("max_tokens", 256),
        temperature=g.get("temperature", 0.0),
        timeout=g.get("timeout", 60),
        max_retries=g.get("max_retries", 3),
    )

    reranker = None
    if use_rerank:
        r = cfg["reranker"]
        reranker = CrossEncoderReranker(
            model_name=r["model"],
            device=r.get("device", "cpu"),
            batch_size=r.get("batch_size", 32),
        )

    return RAGPipeline(
        retriever=retriever,
        corpus=corpus,
        llm=llm,
        reranker=reranker,
        retrieve_top_k=top_k_retrieve,
        final_top_k=top_k_final,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--top-k-retrieve", type=int, default=None)
    parser.add_argument("--top-k-final", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dump-jsonl", default=None)
    parser.add_argument("--out", default=None, help="覆盖默认输出路径")
    args = parser.parse_args()

    cfg = load_config(args.config)

    top_k_retrieve = args.top_k_retrieve or cfg["retriever"]["top_k"]
    top_k_final = args.top_k_final or cfg["reranker"]["top_k"]
    use_rerank = (not args.no_rerank) and cfg["reranker"].get("enabled", True)

    # 加载数据
    data_path = cfg["data"][args.split]
    rows = list(iter_jsonl(data_path))
    if args.limit:
        rows = rows[: args.limit]
    questions = [r["question"] for r in rows]

    print(f"[infer] split={args.split}  n={len(questions)}  rerank={use_rerank}  "
          f"top_k_retrieve={top_k_retrieve}  top_k_final={top_k_final}")

    pipe = build_pipeline(cfg, use_rerank, top_k_retrieve, top_k_final)
    results = pipe.answer_batch(
        questions, num_workers=cfg["inference"].get("num_workers", 8)
    )

    answers = [r.answer for r in results]

    out_path = args.out or cfg["output"][args.split]
    write_lines(out_path, answers)
    print(f"[infer] answers written to {out_path}")

    if args.dump_jsonl:
        Path(args.dump_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(args.dump_jsonl, "w", encoding="utf-8") as f:
            for row, res in zip(rows, results):
                obj = {
                    "question": res.question,
                    "prediction": res.answer,
                    "answers": row.get("answers"),
                    "retrieved": [
                        {"id": d["id"], "score": d["score"]}
                        for d in res.retrieved
                    ],
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"[infer] details dumped to {args.dump_jsonl}")


if __name__ == "__main__":
    main()
