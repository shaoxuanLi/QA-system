"""通用 IO 工具：jsonl 读写、按行偏移读取大语料。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

import numpy as np
import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_jsonl(path: str) -> list[dict]:
    return list(iter_jsonl(path))


def write_lines(path: str, lines: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.replace("\n", " ").strip() + "\n")


class CorpusReader:
    """按文档 id (= 行号) 读取大语料，避免一次性载入 14GB。

    第一次使用前需调用 build_offsets() 生成行偏移表 (~8MB / 1M docs)。
    """

    def __init__(self, corpus_path: str, offsets_path: str):
        self.corpus_path = corpus_path
        self.offsets_path = offsets_path
        self._offsets: np.ndarray | None = None
        self._fh = None

    # ---------- 索引行偏移 ----------
    def build_offsets(self) -> None:
        offsets: list[int] = []
        with open(self.corpus_path, "rb") as f:
            pos = f.tell()
            line = f.readline()
            while line:
                offsets.append(pos)
                pos = f.tell()
                line = f.readline()
        arr = np.asarray(offsets, dtype=np.int64)
        Path(self.offsets_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(self.offsets_path, arr)
        self._offsets = arr

    def _ensure_loaded(self) -> None:
        if self._offsets is None:
            if not os.path.exists(self.offsets_path):
                raise FileNotFoundError(
                    f"未找到 offsets 文件 {self.offsets_path}，请先运行 build_index.py"
                )
            self._offsets = np.load(self.offsets_path)
        if self._fh is None:
            self._fh = open(self.corpus_path, "rb")

    # ---------- 取文档 ----------
    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._offsets)

    def get(self, doc_id: int) -> dict:
        self._ensure_loaded()
        self._fh.seek(int(self._offsets[doc_id]))
        line = self._fh.readline().decode("utf-8")
        return json.loads(line)

    def get_many(self, doc_ids: list[int]) -> list[dict]:
        return [self.get(i) for i in doc_ids]

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
