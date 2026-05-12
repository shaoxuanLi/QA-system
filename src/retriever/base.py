"""检索器基类。所有检索器实现 retrieve(query, top_k) -> List[Tuple[doc_id, score]]"""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[tuple[int, float]]:
        ...

    def retrieve_batch(
        self, queries: list[str], top_k: int
    ) -> list[list[tuple[int, float]]]:
        return [self.retrieve(q, top_k) for q in queries]
