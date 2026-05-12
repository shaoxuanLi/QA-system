"""Gradio Web UI：输入问题 -> 显示检索到的段落 + 最终答案。

启动：
    python -m ui.app --config config.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr

from src.generator import SiliconFlowLLM
from src.rag import RAGPipeline
from src.reranker import CrossEncoderReranker
from src.retriever import BM25Retriever
from src.utils.io import CorpusReader, load_config


def _build_pipeline(cfg: dict) -> RAGPipeline:
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

    r = cfg["reranker"]
    reranker = CrossEncoderReranker(
        model_name=r["model"],
        device=r.get("device", "cpu"),
        batch_size=r.get("batch_size", 32),
    ) if r.get("enabled", True) else None

    return RAGPipeline(
        retriever=retriever,
        corpus=corpus,
        llm=llm,
        reranker=reranker,
        retrieve_top_k=cfg["retriever"]["top_k"],
        final_top_k=cfg["reranker"]["top_k"],
    )


def _format_passages(passages: list[dict]) -> str:
    lines = []
    for i, p in enumerate(passages, 1):
        text = p.get("text", "").replace("\n", " ")
        lines.append(
            f"### Passage {i}  (doc_id={p['id']}, score={p['score']:.3f})\n\n{text}"
        )
    return "\n\n---\n\n".join(lines) if lines else "_no passages_"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pipeline = _build_pipeline(cfg)

    def respond(question: str, top_k_retrieve: int, top_k_final: int, use_rerank: bool):
        if not question.strip():
            return "（请输入问题）", ""
        pipeline.retrieve_top_k = int(top_k_retrieve)
        pipeline.final_top_k = int(top_k_final)
        original_reranker = pipeline.reranker
        if not use_rerank:
            pipeline.reranker = None
        try:
            res = pipeline.answer(question)
        finally:
            pipeline.reranker = original_reranker
        return res.answer, _format_passages(res.retrieved)

    with gr.Blocks(title="Wiki RAG QA") as demo:
        gr.Markdown("# Wiki RAG QA\nBM25 + BGE Reranker + Qwen2.5-7B-Instruct")
        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(
                    label="问题",
                    placeholder="例：Who was the first president of the United States?",
                    lines=2,
                )
                with gr.Row():
                    top_k_retrieve = gr.Slider(5, 100, value=cfg["retriever"]["top_k"], step=1, label="BM25 召回 top_k")
                    top_k_final = gr.Slider(1, 20, value=cfg["reranker"]["top_k"], step=1, label="进入 LLM 的段落数")
                    use_rerank = gr.Checkbox(value=cfg["reranker"].get("enabled", True), label="启用 Reranker")
                submit = gr.Button("提问", variant="primary")
                answer = gr.Textbox(label="答案", lines=2, interactive=False)
            with gr.Column(scale=4):
                passages = gr.Markdown(label="检索到的段落")
        submit.click(
            respond,
            inputs=[question, top_k_retrieve, top_k_final, use_rerank],
            outputs=[answer, passages],
        )
        question.submit(
            respond,
            inputs=[question, top_k_retrieve, top_k_final, use_rerank],
            outputs=[answer, passages],
        )

    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
