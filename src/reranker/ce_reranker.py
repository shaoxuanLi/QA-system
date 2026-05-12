"""Cross-Encoder 重排序器。

使用 sentence-transformers 的 CrossEncoder 接口加载 bge-reranker-base/-v2-m3。
输入：query + 候选文档文本列表
输出：按相关性降序的索引和分数
"""
from __future__ import annotations

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._model: CrossEncoder | None = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self.max_length,
            )

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """对 documents 重排，返回 [(原索引, 分数), ...]（降序）。"""
        if not documents:
            return []
        self._ensure_loaded()
        pairs = [(query, d) for d in documents]
        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        ranked = sorted(
            enumerate(scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        if top_k is not None:
            ranked = ranked[:top_k]
        return [(int(i), float(s)) for i, s in ranked]
