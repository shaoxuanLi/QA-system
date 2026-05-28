"""RAG 主流程：retrieve -> (rerank) -> generate。

可通过 enable_reranker / top_k 参数做消融。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..generator import SiliconFlowLLM
from ..reranker import CrossEncoderReranker
from ..retriever import BaseRetriever
from ..utils.io import CorpusReader
from .prompts import build_messages


def _post_process_answer(text: str, max_chars: int = 150) -> str:
    """规范化 LLM 输出，避免 token 循环 / 多行答案污染 dev.txt。
    - 取第一行非空内容
    - 去掉首尾引号 / 多余空白
    - 限长 (硬截断)
    """
    if not text:
        return ""
    for line in text.splitlines():
        line = line.strip()
        if line:
            text = line
            break
    else:
        return ""
    # 去掉常见前缀 "Answer:" "答案：" "Final answer:"
    for prefix in ("Answer:", "answer:", "ANSWER:", "Final answer:", "答案：", "答："):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    # 去引号
    text = text.strip().strip('"').strip("'").strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0]
    return text


@dataclass
class RAGResult:
    question: str
    answer: str
    retrieved: list[dict] = field(default_factory=list)   # 进入 LLM 上下文的最终段落
    candidates: list[dict] = field(default_factory=list)  # BM25 召回的全部候选 (含分数)


class RAGPipeline:
    def __init__(
        self,
        retriever: BaseRetriever,
        corpus: CorpusReader,
        llm: SiliconFlowLLM,
        reranker: CrossEncoderReranker | None = None,
        retrieve_top_k: int = 20,
        final_top_k: int = 5,
    ):
        self.retriever = retriever
        self.corpus = corpus
        self.llm = llm
        self.reranker = reranker
        self.retrieve_top_k = retrieve_top_k
        self.final_top_k = final_top_k

    # ---------- 单条 ----------
    def answer(self, question: str) -> RAGResult:
        # 1) 检索
        hits = self.retriever.retrieve(question, self.retrieve_top_k)
        doc_ids = [d for d, _ in hits]
        docs = self.corpus.get_many(doc_ids)
        texts = [d.get("contents", "") for d in docs]

        candidates = [
            {"id": did, "score": sc, "text": txt}
            for (did, sc), txt in zip(hits, texts)
        ]

        # 2) 重排（可选）
        if self.reranker is not None and texts:
            ranked = self.reranker.rerank(question, texts, top_k=self.final_top_k)
            final_idx = [i for i, _ in ranked]
            final_texts = [texts[i] for i in final_idx]
            final_meta = [
                {"id": doc_ids[i], "score": s, "text": texts[i]}
                for (i, s) in ranked
            ]
        else:
            final_texts = texts[: self.final_top_k]
            final_meta = candidates[: self.final_top_k]

        # 3) 生成 + 后处理
        messages = build_messages(question, final_texts)
        raw = self.llm.chat(messages)
        answer = _post_process_answer(raw)

        return RAGResult(
            question=question,
            answer=answer,
            retrieved=final_meta,
            candidates=candidates,
        )

    # ---------- 批量 ----------
    def answer_batch(
        self,
        questions: list[str],
        num_workers: int = 8,
    ) -> list[RAGResult]:
        """先批量检索+重排（CPU/本地），再多线程并发请求 LLM。"""
        # 1) 检索
        all_hits = self.retriever.retrieve_batch(questions, self.retrieve_top_k)

        # 2) 取文本 + 重排，准备 prompts
        prompts: list[list[dict]] = []
        finals: list[list[dict]] = []
        cands_all: list[list[dict]] = []
        for q, hits in zip(questions, all_hits):
            doc_ids = [d for d, _ in hits]
            docs = self.corpus.get_many(doc_ids)
            texts = [d.get("contents", "") for d in docs]
            cands = [
                {"id": did, "score": sc, "text": txt}
                for (did, sc), txt in zip(hits, texts)
            ]
            cands_all.append(cands)

            if self.reranker is not None and texts:
                ranked = self.reranker.rerank(q, texts, top_k=self.final_top_k)
                final_texts = [texts[i] for i, _ in ranked]
                final_meta = [
                    {"id": doc_ids[i], "score": s, "text": texts[i]}
                    for (i, s) in ranked
                ]
            else:
                final_texts = texts[: self.final_top_k]
                final_meta = cands[: self.final_top_k]

            finals.append(final_meta)
            prompts.append(build_messages(q, final_texts))

        # 3) 并发调用 LLM + 后处理
        raw_answers = self.llm.chat_batch(prompts, num_workers=num_workers, desc="generate")
        answers = [_post_process_answer(a) for a in raw_answers]

        return [
            RAGResult(
                question=q, answer=a, retrieved=fin, candidates=cands,
            )
            for q, a, fin, cands in zip(questions, answers, finals, cands_all)
        ]
