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


def _truncate_repetition(text: str, max_repeats: int = 3) -> str:
    """Qwen2.5-7B 在低温下陷入 token 循环 (eg. 'on on on on...').
    检测连续重复的 1-3 元 token 序列，从第一次出现循环处截断。
    """
    toks = text.split()
    if len(toks) < max_repeats:
        return text
    for ngram_n in (1, 2, 3):
        i = 0
        while i + ngram_n <= len(toks):
            ngram = toks[i : i + ngram_n]
            repeats = 1
            j = i + ngram_n
            while j + ngram_n <= len(toks) and toks[j : j + ngram_n] == ngram:
                repeats += 1
                j += ngram_n
            if repeats >= max_repeats:
                # 保留第一次出现，砍掉重复尾巴
                return " ".join(toks[: i + ngram_n])
            i += 1
    return text


import re


def _post_process_cot(text: str, max_chars: int = 150) -> str:
    """CoT 模式后处理：从 "Final Answer: X" 中抽取 X。

    LLM 可能输出：
      Reasoning: <...>
      Final Answer: Ernest Hemingway
    或者：
      <推理...>
      Final Answer: 1945

    抽取规则：
    1. 找最后一次出现的 "Final Answer:" / "final answer:"，取之后内容到行尾/句号
    2. 退化兜底：找 "Answer:" 前缀
    3. 再退化：取最后一个非空行（推理之后通常就是答案）
    4. 套用通用清理 (去引号、去冠词、限长、砍循环)
    """
    if not text:
        return ""

    # 1) Final Answer: <X>
    m = re.search(r"final\s*answer\s*[:：]\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    else:
        # 2) Answer: <X>
        m = re.search(r"\banswer\s*[:：]\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        if m:
            text = m.group(1).strip()
        else:
            # 3) 退化：最后一个非空行
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            text = lines[-1] if lines else ""

    return _clean_short_answer(text, max_chars=max_chars)


def _clean_short_answer(text: str, max_chars: int = 150) -> str:
    """通用短答案清理：取首行、去常见前缀、去引号、砍循环、限长。
    注意：不要去冠词，因为 SQuAD 评估的归一化函数会处理；这里去会改变原始答案
    的字面，反而可能让某些题答错（实测 EM -1.5）。
    """
    if not text:
        return ""
    text = text.strip()
    # 去常见前缀
    for prefix in (
        "Answer:", "answer:", "ANSWER:",
        "Final answer:", "Final Answer:", "final answer:",
        "答案：", "答：",
    ):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    # 去引号
    text = text.strip().strip('"').strip("'").strip()
    # 砍 token 循环退化（之前 dev EM 15.5 已经包含这个）
    text = _truncate_repetition(text)
    # 限长
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0]
    return text


def _post_process_answer(text: str, max_chars: int = 150) -> str:
    """严格模式 (EM/F1 评测) 后处理：取首行 + 通用短答案清理。"""
    if not text:
        return ""
    for line in text.splitlines():
        line = line.strip()
        if line:
            text = line
            break
    else:
        return ""
    return _clean_short_answer(text, max_chars=max_chars)


def _post_process_friendly(text: str, max_chars: int = 400) -> str:
    """UI 模式后处理：允许多句完整答案，但同样要截掉退化尾巴。

    - 先 _truncate_repetition 砍掉 token 循环
    - 取第一段（遇空行截）
    - 限长 400 字符
    """
    if not text:
        return ""
    text = text.strip()
    # 取到第一个空行之前的内容
    paragraphs = text.split("\n\n")
    text = paragraphs[0].strip()
    # 去掉常见前缀
    for prefix in ("Answer:", "answer:", "ANSWER:", "答案：", "答："):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    # 砍掉 token 循环 (Qwen 退化)
    text = _truncate_repetition(text)
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "…"
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
        prompt_mode: str = "strict",   # 'strict' / 'cot' / 'friendly'
        postprocess: bool = True,
        n_samples: int = 1,            # self-consistency 采样数 (1=贪婪, 3-5=majority vote)
        sample_temperature: float = 0.4,  # n_samples>1 时用的温度
    ):
        self.retriever = retriever
        self.corpus = corpus
        self.llm = llm
        self.reranker = reranker
        self.retrieve_top_k = retrieve_top_k
        self.final_top_k = final_top_k
        self.prompt_mode = prompt_mode
        self.postprocess = postprocess
        self.n_samples = n_samples
        self.sample_temperature = sample_temperature

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

        # 3) 生成 + 后处理（strict / cot / friendly）
        messages = build_messages(question, final_texts, mode=self.prompt_mode)
        raw = self.llm.chat(messages)
        if not self.postprocess:
            answer = raw.strip()
        elif self.prompt_mode == "cot":
            answer = _post_process_cot(raw)
        elif self.prompt_mode == "friendly":
            answer = _post_process_friendly(raw)
        else:
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
            prompts.append(build_messages(q, final_texts, mode=self.prompt_mode))

        # 3) 并发调用 LLM
        if self.n_samples <= 1:
            raw_answers = self.llm.chat_batch(prompts, num_workers=num_workers, desc="generate")
        else:
            # Self-consistency: 每条问题采样 n 次（temperature>0），取多数投票答案
            raw_answers = self._self_consistency_batch(prompts, num_workers=num_workers)

        # 4) 后处理
        if not self.postprocess:
            answers = [a.strip() for a in raw_answers]
        elif self.prompt_mode == "cot":
            answers = [_post_process_cot(a) for a in raw_answers]
        elif self.prompt_mode == "friendly":
            answers = [_post_process_friendly(a) for a in raw_answers]
        else:
            answers = [_post_process_answer(a) for a in raw_answers]

        return [
            RAGResult(
                question=q, answer=a, retrieved=fin, candidates=cands,
            )
            for q, a, fin, cands in zip(questions, answers, finals, cands_all)
        ]

    def _self_consistency_batch(
        self,
        prompts: list[list[dict]],
        num_workers: int = 4,
    ) -> list[str]:
        """对每条 prompt 用非零温度采样 n_samples 次，取多数投票答案。

        - 把 n_samples 份 prompt 平铺成 n_samples × N 个调用
        - 并发跑完，按 prompt 分组
        - 每组用归一化后的答案做 majority vote
        """
        from collections import Counter
        from ..evaluation.metrics import normalize_answer

        n = len(prompts)
        k = self.n_samples
        flat: list[list[dict]] = []
        for i in range(n):
            for _ in range(k):
                flat.append(prompts[i])

        # 复用 chat_batch，但每次都用 sample_temperature 而不是默认 0
        original_temp = self.llm.temperature
        self.llm.temperature = self.sample_temperature
        try:
            flat_raw = self.llm.chat_batch(
                flat, num_workers=num_workers,
                desc=f"generate ×{k}",
            )
        finally:
            self.llm.temperature = original_temp

        # 按 prompt 分组 + 投票
        results: list[str] = []
        for i in range(n):
            group_raws = flat_raw[i * k : (i + 1) * k]
            # 先 post-process 得到候选短答案
            if self.prompt_mode == "cot":
                cands = [_post_process_cot(r) for r in group_raws]
            elif self.prompt_mode == "friendly":
                cands = [_post_process_friendly(r) for r in group_raws]
            else:
                cands = [_post_process_answer(r) for r in group_raws]
            # 归一化后投票（"Canberra" 与 "the canberra" 算同一票）
            norm_to_raw: dict[str, str] = {}
            counts = Counter()
            for c in cands:
                if not c.strip():
                    continue
                key = normalize_answer(c)
                counts[key] += 1
                norm_to_raw.setdefault(key, c)
            if not counts:
                results.append("")
                continue
            # 取票数最高的；票数相同时按出现顺序（Counter 自带）
            winner_norm = counts.most_common(1)[0][0]
            results.append(norm_to_raw[winner_norm])
        return results
