"""一次性脚本：构建 BM25 索引 + 文档行偏移表。

Usage:
    python -m scripts.build_index --config config.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retriever import BM25Retriever
from src.utils.io import CorpusReader, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--skip-bm25", action="store_true", help="只生成行偏移，不重建 BM25 索引"
    )
    parser.add_argument(
        "--skip-offsets", action="store_true", help="只重建 BM25 索引，不重建行偏移"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    corpus_path = cfg["data"]["corpus"]
    index_dir = cfg["index"]["dir"]
    offsets_path = cfg["index"]["offsets"]

    # 1) 行偏移
    if not args.skip_offsets:
        print(f"[offsets] building offsets for {corpus_path} -> {offsets_path}")
        reader = CorpusReader(corpus_path, offsets_path)
        reader.build_offsets()
        print(f"[offsets] done, {len(reader)} documents")
        reader.close()

    # 2) BM25 索引
    if not args.skip_bm25:
        print(f"[bm25] building index from {corpus_path} -> {index_dir}")
        BM25Retriever.build(corpus_path=corpus_path, index_dir=index_dir)
        print(f"[bm25] done.")


if __name__ == "__main__":
    main()
