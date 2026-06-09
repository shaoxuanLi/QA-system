"""为汇报生成 4 张图表。运行后图片落在 presentation/ 目录。

usage:
    python -m scripts.make_presentation_figures
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


OUT_DIR = Path("presentation")
OUT_DIR.mkdir(exist_ok=True)

# 配色：indigo / cyan / amber, 与 UI 主题一致
COLOR_BM25 = "#4f46e5"     # indigo-600
COLOR_RER = "#06b6d4"      # cyan-500
COLOR_LLM = "#f59e0b"      # amber-500
COLOR_BG = "#f8fafc"       # slate-50
COLOR_TXT = "#1e293b"      # slate-800
COLOR_OK = "#10b981"       # emerald-500
COLOR_BAD = "#ef4444"      # rose-500

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
})


# ============================================================
# Figure 1: 系统架构图
# ============================================================
def make_architecture():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    boxes = [
        (0.4, 1.5, 1.8, 2.0, "Question", "#e0e7ff", "#312e81"),
        (2.6, 1.5, 2.0, 2.0, "BM25\nRetriever\n(21M docs)", COLOR_BM25, "white"),
        (5.0, 1.5, 2.0, 2.0, "Cross-Encoder\nReranker\n(bge-base)", COLOR_RER, "white"),
        (7.4, 1.5, 2.0, 2.0, "LLM\nQwen2.5-7B\n(SiliconFlow)", COLOR_LLM, "white"),
        (9.8, 1.5, 1.8, 2.0, "Short\nAnswer", "#dcfce7", "#14532d"),
    ]
    for x, y, w, h, txt, fc, tc in boxes:
        box = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
            linewidth=1.5, edgecolor=COLOR_TXT, facecolor=fc,
        )
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center",
                fontsize=11, color=tc, fontweight="bold")

    for x_start, x_end in [(2.2, 2.6), (4.6, 5.0), (7.0, 7.4), (9.4, 9.8)]:
        arrow = FancyArrowPatch(
            (x_start, 2.5), (x_end, 2.5),
            arrowstyle="-|>", mutation_scale=20, linewidth=2, color=COLOR_TXT,
        )
        ax.add_patch(arrow)

    # 上方标注
    ax.text(3.6, 4.0, "top_k_retrieve = 20", ha="center", fontsize=10,
            style="italic", color=COLOR_TXT, alpha=0.7)
    ax.text(6.0, 4.0, "top_k_final = 3", ha="center", fontsize=10,
            style="italic", color=COLOR_TXT, alpha=0.7)
    ax.text(8.4, 4.0, "max_tokens = 64", ha="center", fontsize=10,
            style="italic", color=COLOR_TXT, alpha=0.7)

    # 底部细节
    ax.text(3.6, 0.8, "9.4 GB index\nCorpus seek by byte offset", ha="center",
            fontsize=9, color=COLOR_TXT, alpha=0.6)
    ax.text(6.0, 0.8, "MPS GPU\n1.9× faster", ha="center",
            fontsize=9, color=COLOR_TXT, alpha=0.6)
    ax.text(8.4, 0.8, "Stop seq + post-process\nHandle Qwen degradation",
            ha="center", fontsize=9, color=COLOR_TXT, alpha=0.6)

    fig.suptitle("Wiki RAG QA — System Architecture",
                 fontsize=16, fontweight="bold", y=0.96)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(OUT_DIR / "fig1_architecture.png", dpi=180, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"  wrote {OUT_DIR / 'fig1_architecture.png'}")


# ============================================================
# Figure 2: 消融实验柱状图 (从 ablation.json 读数据)
# ============================================================
def make_ablation():
    if not Path("ablation.json").exists():
        print("  WARN: ablation.json 不存在，跳过图 2")
        return
    rows = json.load(open("ablation.json"))

    labels = [r["name"].replace("bm25-only", "BM25").replace("bm25+rerank", "BM25+R")
                       .replace("recall", "r")
              for r in rows]
    ems = [r["EM"] for r in rows]
    f1s = [r["F1"] for r in rows]

    colors = [COLOR_BM25 if not r["rerank"] else COLOR_RER for r in rows]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(labels))
    width = 0.38

    b1 = ax.bar(x - width/2, ems, width, label="EM", color=colors,
                edgecolor=COLOR_TXT, linewidth=0.8)
    b2 = ax.bar(x + width/2, f1s, width, label="F1", color=colors,
                edgecolor=COLOR_TXT, linewidth=0.8, alpha=0.55)

    # 高亮最优
    best_idx = np.argmax(ems)
    ax.bar(x[best_idx] - width/2, ems[best_idx], width, color=COLOR_OK,
           edgecolor=COLOR_TXT, linewidth=2)
    ax.bar(x[best_idx] + width/2, f1s[best_idx], width, color=COLOR_OK,
           alpha=0.7, edgecolor=COLOR_TXT, linewidth=2)

    for i, (e, f) in enumerate(zip(ems, f1s)):
        ax.text(i - width/2, e + 0.3, f"{e:.1f}", ha="center", fontsize=8.5,
                color=COLOR_TXT)
        ax.text(i + width/2, f + 0.3, f"{f:.1f}", ha="center", fontsize=8.5,
                color=COLOR_TXT, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=10)
    ax.set_ylabel("Score (%)")
    ax.set_title(
        "Ablation on dev.jsonl (N=1000) — BM25-only (indigo) vs +Reranker (cyan), "
        f"best ⇒ {rows[best_idx]['name']}",
        loc="left",
    )

    # legend
    legend_handles = [
        mpatches.Patch(color=COLOR_BM25, label="BM25 only"),
        mpatches.Patch(color=COLOR_RER,  label="BM25 + Reranker"),
        mpatches.Patch(color=COLOR_OK,   label="Best"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_ylim(0, max(f1s) * 1.18)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig2_ablation.png", dpi=180, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"  wrote {OUT_DIR / 'fig2_ablation.png'}")


# ============================================================
# Figure 3: LLM 选型对比 (前 200 条 dev)
# ============================================================
def make_llm_comparison():
    data = {
        "Qwen2.5-7B-Instruct\n(course required)": (13.50, 19.49),
        "DeepSeek-V4-Pro\n(modern LLM)":           (18.50, 19.42),
    }
    labels = list(data.keys())
    ems = [data[k][0] for k in labels]
    f1s = [data[k][1] for k in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    width = 0.32

    bar_em = ax.bar(x - width/2, ems, width, label="EM",
                    color=[COLOR_LLM, COLOR_RER], edgecolor=COLOR_TXT, linewidth=1.5)
    bar_f1 = ax.bar(x + width/2, f1s, width, label="F1",
                    color=[COLOR_LLM, COLOR_RER], alpha=0.55,
                    edgecolor=COLOR_TXT, linewidth=1.5)

    for i, (e, f) in enumerate(zip(ems, f1s)):
        ax.text(i - width/2, e + 0.3, f"{e:.2f}", ha="center", fontsize=11,
                fontweight="bold")
        ax.text(i + width/2, f + 0.3, f"{f:.2f}", ha="center", fontsize=11,
                fontweight="bold", alpha=0.7)

    # 标注 +5 EM 差异
    ax.annotate("", xy=(1 - width/2, 18.5), xytext=(0 - width/2, 13.5),
                arrowprops=dict(arrowstyle="->", color=COLOR_OK, lw=2.5))
    ax.text(0.5, 16.5, "+5.00 EM\nsame retrieval,\nstronger LLM",
            ha="center", color=COLOR_OK, fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score (%)")
    ax.set_title("LLM Comparison — identical retrieval pipeline (200 dev questions)",
                 loc="left")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_ylim(0, 25)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig3_llm_compare.png", dpi=180, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"  wrote {OUT_DIR / 'fig3_llm_compare.png'}")


# ============================================================
# Figure 4: 工程优化时间线（gantt-like）
# ============================================================
def make_optimization_timeline():
    items = [
        ("Build index (21M docs)",              0, 26,  COLOR_BM25),
        ("Initial run @ 8 workers",             26, 5,  COLOR_BAD),
        ("→ 358/1000 fail (rate limit)",        31, 0,  None),
        ("Fix: 4 workers + exp backoff",        31, 1,  COLOR_OK),
        ("Retry empties",                       32, 3,  COLOR_RER),
        ("→ 16/1000 empty, EM 11.10",           35, 0,  None),
        ("Ablation 8 configs",                  36, 180, COLOR_BM25),
        ("→ Best: top_k_final = 3 (not 5!)",    216, 0, None),
        ("Switch to top3 → EM 15.50",           216, 5,  COLOR_OK),
        ("MPS GPU on reranker (1.9×)",          221, 2,  COLOR_RER),
        ("LLM compare: DeepSeek + 5 EM",        223, 3,  COLOR_LLM),
    ]
    labels = [t[0] for t in items if t[3] is not None]
    starts = [t[1] for t in items if t[3] is not None]
    durs   = [max(t[2], 1) for t in items if t[3] is not None]
    colors = [t[3] for t in items if t[3] is not None]

    annotations = [t for t in items if t[3] is None]

    fig, ax = plt.subplots(figsize=(11, 5.5))

    y = np.arange(len(labels))
    ax.barh(y, durs, left=starts, color=colors, edgecolor=COLOR_TXT, linewidth=0.8,
            height=0.6)
    for i, lbl in enumerate(labels):
        ax.text(starts[i] + durs[i] + 1, i, lbl, va="center", fontsize=10,
                color=COLOR_TXT)

    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xlabel("Cumulative Wall-clock Time (minutes)")
    ax.set_title("Engineering Journey — discoveries that drove the score upward",
                 loc="left")
    ax.set_xlim(0, 280)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.spines["left"].set_visible(False)

    # 添加注释
    for note, x_pos, _, _ in annotations:
        # 找到时间最接近的 item 位置作为 y
        idx = min(range(len(starts)),
                  key=lambda i: abs(starts[i] - x_pos))
        ax.annotate(note, xy=(x_pos, idx + 0.4),
                    xytext=(x_pos + 5, idx + 1.3),
                    fontsize=9, color="#475569", style="italic",
                    arrowprops=dict(arrowstyle="-", color="#94a3b8",
                                    lw=0.8, alpha=0.6))

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig4_journey.png", dpi=180, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"  wrote {OUT_DIR / 'fig4_journey.png'}")


if __name__ == "__main__":
    print("Generating presentation figures...")
    make_architecture()
    make_ablation()
    make_llm_comparison()
    make_optimization_timeline()
    print("Done.")
