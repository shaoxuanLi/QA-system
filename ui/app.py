"""Gradio Web UI：BM25 + Reranker + LLM 的在线问答演示。

启动：
    python -m ui.app --config config.yaml

特性：
- 启动时一次性预热 BM25 索引、reranker 模型、LLM 客户端，运行时秒级响应
- 支持 Dropdown 切换语言模型 (Qwen2.5-7B / DeepSeek-V4-Pro / GLM-5.1 / Qwen3.6-Plus)
- 显示 BM25 召回 + Reranker 重排的段落，便于理解 RAG 内部
- 可调 top_k_retrieve / top_k_final / 开关 Reranker，演示消融效果
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr

from src.generator import SiliconFlowLLM
from src.rag import RAGPipeline
from src.reranker import CrossEncoderReranker
from src.retriever import BM25Retriever
from src.utils.io import CorpusReader, load_config


# 全局共享资源：UI 启动时加载一次，所有请求共用
_RETRIEVER: BM25Retriever | None = None
_CORPUS: CorpusReader | None = None
_RERANKER: CrossEncoderReranker | None = None
_LLM_CACHE: dict[str, SiliconFlowLLM] = {}
_MODEL_PRESETS: dict[str, dict] = {}
_CFG: dict = {}


# ----------------- 资源构造 -----------------

def _make_llm_from_preset(preset: dict) -> SiliconFlowLLM:
    return SiliconFlowLLM(
        api_url=preset["api_url"],
        model=preset["model"],
        api_key=preset.get("api_key", ""),
        api_key_env=preset.get("api_key_env", "SILICONFLOW_API_KEY"),
        max_tokens=preset.get("max_tokens", 256),
        temperature=preset.get("temperature", 0.0),
        frequency_penalty=preset.get("frequency_penalty", 0.0),
        timeout=preset.get("timeout", 60),
        max_retries=preset.get("max_retries", 5),
    )


def _get_llm(model_name: str) -> tuple[SiliconFlowLLM, str]:
    """返回 (LLM, prompt_mode)。首次访问时实例化并缓存。"""
    if model_name not in _LLM_CACHE:
        preset = _MODEL_PRESETS[model_name]
        _LLM_CACHE[model_name] = _make_llm_from_preset(preset)
    return _LLM_CACHE[model_name], _MODEL_PRESETS[model_name].get("prompt_mode", "friendly")


# ----------------- 启动预热 -----------------

def _warmup(default_model_name: str) -> None:
    """启动时预加载 BM25 索引 (9.4GB) + reranker 模型 (~1GB) + 默认 LLM。
    用户首次提问无需等待。
    """
    print("=" * 60)
    print("[warmup] preloading heavy resources …")

    t0 = time.time()
    _RETRIEVER._ensure_loaded()
    print(f"[warmup] BM25 index loaded in {time.time() - t0:.1f}s")

    if _RERANKER is not None:
        t0 = time.time()
        _RERANKER._ensure_loaded()
        _RERANKER.rerank("warmup", ["dummy passage"], top_k=1)
        print(f"[warmup] reranker loaded + warmed in {time.time() - t0:.1f}s")

    t0 = time.time()
    llm, _ = _get_llm(default_model_name)
    llm.chat([{"role": "user", "content": "Say hi"}])
    print(f"[warmup] default LLM ({default_model_name}) handshake in {time.time() - t0:.1f}s")

    print("[warmup] ready. UI 现在每条提问只需 1-5 秒.")
    print("=" * 60)


# ----------------- 推理 -----------------

def respond(
    question: str,
    model_name: str,
    top_k_retrieve: int,
    top_k_final: int,
    use_rerank: bool,
):
    if not question.strip():
        return "_(请输入问题)_", "", ""

    t0 = time.time()
    llm, prompt_mode = _get_llm(model_name)

    pipe = RAGPipeline(
        retriever=_RETRIEVER,
        corpus=_CORPUS,
        llm=llm,
        reranker=_RERANKER if use_rerank else None,
        retrieve_top_k=int(top_k_retrieve),
        final_top_k=int(top_k_final),
        prompt_mode=prompt_mode,
        postprocess=True,
    )
    res = pipe.answer(question.strip())
    elapsed = time.time() - t0

    answer_md = f"### 💬 {res.answer}\n\n*<small>耗时 {elapsed:.2f}s · 模型 {model_name}</small>*"
    passages_md = _format_passages(res.retrieved)
    meta_md = _format_meta(model_name, top_k_retrieve, top_k_final, use_rerank, len(res.retrieved))
    return answer_md, passages_md, meta_md


def _format_passages(passages: list[dict]) -> str:
    if not passages:
        return "_(没有检索到段落)_"
    parts = []
    for i, p in enumerate(passages, 1):
        text = p.get("text", "").replace("\n", " ").strip()
        if len(text) > 600:
            text = text[:600] + "…"
        parts.append(
            f"#### 📄 段落 {i}\n"
            f"<small>doc_id=`{p['id']}` · rerank score=`{p['score']:.3f}`</small>\n\n"
            f"> {text}"
        )
    return "\n\n---\n\n".join(parts)


def _format_meta(model_name: str, top_k_retrieve: int, top_k_final: int,
                 use_rerank: bool, n_returned: int) -> str:
    note = _MODEL_PRESETS.get(model_name, {}).get("note", "")
    rerank_str = "✅ 启用" if use_rerank else "❌ 关闭"
    return (
        f"| 字段 | 值 |\n|---|---|\n"
        f"| 模型 | `{model_name}` |\n"
        f"| 说明 | {note} |\n"
        f"| BM25 召回 | top_k_retrieve = {top_k_retrieve} |\n"
        f"| Reranker | {rerank_str}，输入 LLM = {n_returned} 篇 |\n"
    )


# ----------------- UI -----------------

EXAMPLE_QUESTIONS = [
    "Who painted the Mona Lisa?",
    "What is the capital of Australia?",
    "Who wrote The Old Man and the Sea?",
    "In what year was Albert Einstein born?",
    "Who is the founder of Apple Inc.?",
    "What is the tallest mountain in the world?",
    "When did World War II end?",
    "Who invented the telephone?",
]


def build_demo(cfg: dict) -> gr.Blocks:
    default_model = cfg["ui_models"]["default"]
    model_names = [p["name"] for p in cfg["ui_models"]["presets"]]
    default_top_k_retrieve = cfg["retriever"]["top_k"]
    default_top_k_final = cfg["reranker"]["top_k"]
    default_rerank = cfg["reranker"].get("enabled", True)

    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate",
        radius_size=gr.themes.sizes.radius_lg,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ).set(
        button_primary_background_fill="*primary_500",
        button_primary_background_fill_hover="*primary_600",
        block_label_text_weight="600",
    )

    custom_css = """
    .gradio-container { max-width: 1400px !important; margin: 0 auto !important; }
    #header { padding: 1rem 1.5rem; margin-bottom: 0.5rem;
              background: linear-gradient(135deg,#4f46e5 0%,#06b6d4 100%);
              border-radius: 12px; color: white; }
    #header h1 { margin: 0; font-size: 1.6rem; font-weight: 700; }
    #header p  { margin: 0.3rem 0 0 0; opacity: .9; font-size: .95rem; }
    .answer-box { background: #f8fafc; border-left: 4px solid #4f46e5;
                  padding: 0.8rem 1.2rem; border-radius: 8px; }
    .passage-box { max-height: 600px; overflow-y: auto; padding: 0.5rem; }
    """

    with gr.Blocks(
        title="Wiki RAG QA · 智能问答系统",
        theme=theme,
        css=custom_css,
    ) as demo:
        gr.HTML("""
        <div id="header">
            <h1>🔍 Wiki RAG QA · 智能问答系统</h1>
            <p>BM25 (21M Wikipedia 段落) → Cross-Encoder Reranker → LLM &nbsp;|&nbsp; 清华大学 搜索与推荐 大作业</p>
        </div>
        """)

        with gr.Row():
            # 左侧：输入 + 控制
            with gr.Column(scale=5):
                question = gr.Textbox(
                    label="❓ 你的问题",
                    placeholder="例如：Who painted the Mona Lisa?",
                    lines=2,
                    autofocus=True,
                )

                with gr.Row():
                    submit = gr.Button("🚀 提问", variant="primary", scale=2)
                    clear  = gr.Button("🧹 清空", scale=1)

                gr.Examples(
                    examples=[[q] for q in EXAMPLE_QUESTIONS],
                    inputs=[question],
                    label="💡 示例问题（点击直接填入）",
                )

                with gr.Accordion("⚙️ 高级设置 (model / top_k / reranker)", open=False):
                    model_selector = gr.Dropdown(
                        choices=model_names,
                        value=default_model,
                        label="🤖 语言模型",
                        info="评分用 Qwen2.5-7B-Instruct (课程要求)；DeepSeek-V4-Pro 演示更稳",
                    )
                    with gr.Row():
                        top_k_retrieve = gr.Slider(
                            5, 100, value=default_top_k_retrieve, step=1,
                            label="BM25 召回 top_k",
                        )
                        top_k_final = gr.Slider(
                            1, 10, value=default_top_k_final, step=1,
                            label="进入 LLM 的段落数",
                        )
                    use_rerank = gr.Checkbox(
                        value=default_rerank, label="启用 Cross-Encoder Reranker"
                    )

                gr.Markdown("### 📊 本次请求信息")
                meta = gr.Markdown("_（提问后显示）_")

            # 右侧：答案 + 检索证据
            with gr.Column(scale=7):
                gr.Markdown("### 💬 答案")
                answer = gr.Markdown(
                    "_（输入问题后点击「提问」）_",
                    elem_classes=["answer-box"],
                )

                gr.Markdown("### 📚 检索到的支撑段落（按 reranker 分数降序）")
                passages = gr.Markdown(
                    "_（提问后显示）_",
                    elem_classes=["passage-box"],
                )

        # 事件绑定
        inputs = [question, model_selector, top_k_retrieve, top_k_final, use_rerank]
        outputs = [answer, passages, meta]
        submit.click(respond, inputs=inputs, outputs=outputs)
        question.submit(respond, inputs=inputs, outputs=outputs)
        clear.click(
            lambda: ("", "_（输入问题后点击「提问」）_", "_（提问后显示）_", "_（提问后显示）_"),
            outputs=[question, answer, passages, meta],
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    global _CFG, _RETRIEVER, _CORPUS, _RERANKER, _MODEL_PRESETS
    _CFG = load_config(args.config)

    # 装载共享资源
    _RETRIEVER = BM25Retriever(_CFG["index"]["dir"])
    _CORPUS = CorpusReader(_CFG["data"]["corpus"], _CFG["index"]["offsets"])
    r = _CFG["reranker"]
    _RERANKER = CrossEncoderReranker(
        model_name=r["model"],
        device=r.get("device", "cpu"),
        batch_size=r.get("batch_size", 32),
    ) if r.get("enabled", True) else None

    # 模型注册表
    _MODEL_PRESETS = {p["name"]: p for p in _CFG["ui_models"]["presets"]}

    # 启动预热
    _warmup(_CFG["ui_models"]["default"])

    demo = build_demo(_CFG)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
