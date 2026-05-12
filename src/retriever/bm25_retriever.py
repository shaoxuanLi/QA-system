"""基于 bm25s 的 BM25 检索器。

- 索引构建：流式读取语料 -> 分词 -> bm25s.BM25 -> save 到目录
- 查询：bm25s 自带 retrieve，输入 token 数组，返回 (doc_id, score)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import bm25s
import numpy as np
from tqdm import tqdm

from .base import BaseRetriever


def _stream_corpus_text(corpus_path: str) -> Iterator[str]:
    """从 wiki18_100w.jsonl 中流式拿出每篇文档的可索引文本。"""
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # 字段名为 'contents'，包含标题与正文
            yield obj.get("contents", "")


class BM25Retriever(BaseRetriever):
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self._bm25: bm25s.BM25 | None = None

    # ---------- 构建 ----------
    @classmethod
    def build(
        cls,
        corpus_path: str,
        index_dir: str,
        stopwords: str = "en",
        stemmer: str | None = "english",
    ) -> "BM25Retriever":
        """读取语料 -> 分词 -> 建索引 -> 落盘。"""
        Path(index_dir).mkdir(parents=True, exist_ok=True)

        # 1) 收集文本（流式，避免一次性吃满内存里的列表）
        texts: list[str] = []
        for txt in tqdm(_stream_corpus_text(corpus_path), desc="loading corpus"):
            texts.append(txt)

        # 2) 分词
        print(f"[bm25] tokenizing {len(texts)} documents ...")
        stemmer_obj = None
        if stemmer:
            import Stemmer
            stemmer_obj = Stemmer.Stemmer(stemmer)
        tokens = bm25s.tokenize(
            texts,
            stopwords=stopwords,
            stemmer=stemmer_obj,
            show_progress=True,
        )

        # 3) 建索引
        print("[bm25] indexing ...")
        bm25 = bm25s.BM25()
        bm25.index(tokens, show_progress=True)
        bm25.save(index_dir)

        # 保存 stemmer / stopwords 选择，方便检索时复用
        meta = {"stopwords": stopwords, "stemmer": stemmer}
        with open(Path(index_dir) / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        obj = cls(index_dir)
        obj._bm25 = bm25
        obj._stopwords = stopwords
        obj._stemmer_name = stemmer
        return obj

    # ---------- 加载 ----------
    def _ensure_loaded(self) -> None:
        if self._bm25 is not None:
            return
        self._bm25 = bm25s.BM25.load(self.index_dir, load_corpus=False)
        meta_path = Path(self.index_dir) / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self._stopwords = meta.get("stopwords", "en")
            self._stemmer_name = meta.get("stemmer", "english")
        else:
            self._stopwords = "en"
            self._stemmer_name = "english"

    # ---------- 检索 ----------
    def _tokenize(self, queries: list[str]):
        stemmer_obj = None
        if self._stemmer_name:
            import Stemmer
            stemmer_obj = Stemmer.Stemmer(self._stemmer_name)
        return bm25s.tokenize(
            queries,
            stopwords=self._stopwords,
            stemmer=stemmer_obj,
            show_progress=False,
        )

    def retrieve(self, query: str, top_k: int) -> list[tuple[int, float]]:
        return self.retrieve_batch([query], top_k)[0]

    def retrieve_batch(
        self, queries: list[str], top_k: int
    ) -> list[list[tuple[int, float]]]:
        self._ensure_loaded()
        q_tokens = self._tokenize(queries)
        # bm25s.retrieve 返回 (doc_ids[n, k], scores[n, k])
        doc_ids, scores = self._bm25.retrieve(q_tokens, k=top_k, show_progress=False)
        results: list[list[tuple[int, float]]] = []
        for ids_row, sc_row in zip(np.asarray(doc_ids), np.asarray(scores)):
            results.append([(int(i), float(s)) for i, s in zip(ids_row, sc_row)])
        return results
