# Wiki RAG QA System

清华大学《搜索与推荐》智能问答系统大作业。基于 21M 段维基百科语料的检索增强生成 (RAG) 问答。

| 项 | 选型 / 数值 |
|---|---|
| 检索器 | **BM25** (`bm25s`) — 21M 文档，9.4 GB 索引 |
| 额外功能 | **Cross-Encoder 重排序** (`BAAI/bge-reranker-base`)，**MPS GPU** 加速 |
| 生成器 | **Qwen/Qwen2.5-7B-Instruct**（硅基流动 API，作业指定） |
| UI | **Gradio** + Soft 主题 + 4 模型 dropdown |
| 评估 | **EM / F1**（SQuAD 风格归一化） |
| 最终成绩 | dev (N=1000): **EM 15.50 / F1 20.69** |

代码仓库：https://github.com/shaoxuanLi/QA-system

---

## 目录结构

```
QA-sys/
├── config.yaml                    # 全部参数集中
├── requirements.txt
├── README.md                      # 本文件
├── report.md / report.pdf         # 实验报告 (13 章 / 4 张图)
├── handoff.md                     # 队友交接文档（内部材料，提交可删）
│
├── wiki18_100w.jsonl              # 语料 (~14GB, 21M 段落)，不在提交包中
├── dev.jsonl / test.jsonl         # 验证 / 测试集，不在提交包中
├── dev.txt / test.txt             # 答案输出（提交内容）
├── ablation.md / ablation.json    # 消融实验结果
│
├── src/
│   ├── retriever/                 # BaseRetriever + BM25Retriever
│   ├── reranker/                  # CrossEncoderReranker (MPS GPU)
│   ├── generator/                 # SiliconFlowLLM (chat / chat_batch / 退避重试)
│   ├── rag/                       # RAGPipeline + 3 套 prompts + 后处理
│   ├── evaluation/                # EM / F1
│   └── utils/                     # config / jsonl / CorpusReader(字节偏移读语料)
│
├── scripts/
│   ├── build_index.py             # 一次性建 BM25 索引 + 行偏移表
│   ├── run_inference.py           # 跑 dev/test，含 --retry-empty / --prompt-mode
│   ├── run_ablation.py            # 一键 8 组消融 → ablation.md
│   ├── evaluate.py                # EM/F1
│   └── make_presentation_figures.py  # 生成 4 张汇报图（matplotlib）
│
├── ui/app.py                      # Gradio Web UI
│
├── presentation/                  # 汇报用图与大纲
│   ├── fig1_architecture.png      # 系统架构
│   ├── fig2_ablation.png          # 消融柱状图
│   ├── fig3_llm_compare.png       # LLM 对比
│   ├── fig4_journey.png           # 工程历程时间线
│   └── 3min_outline.md            # 3 分钟汇报大纲
│
└── indices/                       # BM25 索引 + offsets（运行后生成，不提交）
```

---

## 环境

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

依赖包含 PyTorch（约 2GB）、sentence-transformers、bm25s、Gradio。

---

## 配置 API Key

**评分用模型（必须）**：硅基流动的 Qwen2.5-7B-Instruct
```bash
export SILICONFLOW_API_KEY="sk-xxxx"
# 或写到 config.yaml 的 generator.api_key
```

**UI 演示可选模型**（Paratera 平台的 DeepSeek-V4-Pro / GLM-5.1 / Qwen3.6-Plus）：
```bash
export PARATERA_KEY="sk-xxxx"
```

> ⚠️ 提交前确保 `config.yaml` 里的 `api_key` 字段是空字符串。

---

## 一次性建索引

```bash
python -m scripts.build_index --config config.yaml
```

生成：
- `indices/bm25/` — BM25 索引（9.4 GB）
- `indices/offsets.npy` — 文档 id → 文件字节偏移表（168 MB），按需读取避免 13GB 语料整体进内存

支持 `--skip-bm25` / `--skip-offsets` 单独重建。

21M 文档全量首次构建约 **26 分钟**（M-series Mac, 32GB RAM，峰值占用 9.6GB）。

---

## 跑验证集 / 测试集

```bash
# 验证集 (使用 commit 的最终配置: top_k_retrieve=20, top_k_final=3, strict prompt)
python -m scripts.run_inference --split dev  --config config.yaml

# 测试集
python -m scripts.run_inference --split test --config config.yaml
```

输出根目录下的 `dev.txt` / `test.txt`（每行一个答案，命名严格遵循作业要求）。

### 高级选项

| 选项 | 作用 |
|---|---|
| `--no-rerank` | 关闭重排序器（消融用） |
| `--top-k-retrieve N` | 覆盖 BM25 召回数 |
| `--top-k-final N` | 覆盖进入 LLM 的段落数 |
| `--prompt-mode {strict\|cot\|friendly}` | 切换 prompt 模板 |
| `--retry-empty` | 只重跑当前输出中**空答案**的那些行（补 API 失败） |
| `--limit N` | 只跑前 N 条（调试用） |
| `--dump-jsonl PATH` | 把 `(question, prediction, retrieved)` 详情存 jsonl |

---

## 评估

```bash
python -m scripts.evaluate --pred dev.txt --gold dev.jsonl
```

输出形如：
```
N   = 1000
EM  = 15.50
F1  = 20.69
```

---

## Web UI

```bash
python -m ui.app --config config.yaml
```

启动时**预热**所有重对象（BM25 索引 + reranker 模型 + LLM 握手），约 30-90 秒。
预热完成后每条提问 1-5 秒。打开 http://127.0.0.1:7860。

特性：
- **4 模型 dropdown**：Qwen2.5-7B（默认）/ DeepSeek-V4-Pro / GLM-5.1 / Qwen3.6-Plus
- 实时调 `top_k_retrieve` / `top_k_final`、开关 Reranker
- 展示检索到的段落 + reranker 分数
- 8 个示例问题点击直接填入

---

## 消融实验

```bash
# 全量 1000 条（约 3 小时，含 8 组配置）
python -m scripts.run_ablation --config config.yaml --out ablation.md

# 调通用前 200 条（约 30 分钟）
python -m scripts.run_ablation --limit 200
```

默认网格：
- BM25 only vs BM25 + Reranker × `top_k_final` ∈ {1, 3, 5}
- Reranker 开启时 `top_k_retrieve` ∈ {20, 50, 100}

输出：
- `ablation.md` — Markdown 对比表
- `ablation.json` — 结构化数据，方便后续画图
- `outputs/ablation/<run_name>.txt` — 每组的预测，方便 case study

**关键发现**：top_k_final = 3 在 reranker 开启下达到 EM 15.50（最优），比 top5 高 +4.40 EM —— 详见 [report.md](report.md) §7。

---

## 生成汇报图表

```bash
python -m scripts.make_presentation_figures
```

生成 4 张 PNG 到 `presentation/`：架构图 / 消融柱状图 / LLM 对比 / 工程历程时间线。

---

## 提交前清理

```bash
# 1. 确认 api_key 已清空
grep -E "sk-" config.yaml

# 2. grep 全项目（无真实 key）
grep -rE "sk-[A-Za-z0-9]{30,}" . \
    --exclude-dir=.venv --exclude-dir=.git --exclude-dir=indices

# 3. 打包（替换 <学号>）
cd ..
zip -r "<学号>.zip" QA-sys \
    -x "QA-sys/.venv/*" "QA-sys/indices/*" "QA-sys/outputs/*" \
    -x "QA-sys/smoke/*" "QA-sys/__MACOSX/*" \
    -x "QA-sys/*.jsonl" "QA-sys/*.bak.txt" \
    -x "QA-sys/handoff.md" "QA-sys/.git/*" \
    -x "*.DS_Store" "*__pycache__*" "*.pyc"

# 4. 检查 zip 内确实没 key
unzip -p "<学号>.zip" "QA-sys/config.yaml" | grep -i "sk-" \
    && echo "⚠️ 还有 key" || echo "✅ 干净"
```

---

## 关键实验结果速览

| 数据集 | EM | F1 | 空答案 |
|---|---|---|---|
| dev (N=1000)  | **15.50** | **20.69** | 10 |
| test (N=1000) | 助教离线评测 | 助教离线评测 | 13 |

| 关键消融发现 | EM | 备注 |
|---|---|---|
| BM25-only top1 | 11.30 | baseline |
| BM25 + Reranker top1 | 13.70 | +2.40 EM (额外功能价值) |
| BM25 + Reranker **top3** ⭐ | **15.50** | 最优配置（"少而精 > 多而全"） |
| BM25 + Reranker top5 | 10.80 | top5 反而劣于 top3 |

| 对照实验 | EM | F1 |
|---|---|---|
| Qwen2.5-7B (作业指定) | 13.50 | 19.49 |
| DeepSeek-V4-Pro (同检索) | **18.50** | 19.42 |

**结论**：F1 几乎一样说明检索召回足够，瓶颈在于 Qwen2.5-7B 的"短答案抽取精度"。

---

## 参考 / 致谢

| 资源 | 用途 |
|---|---|
| [`bm25s`](https://github.com/xhluca/bm25s) | 高性能纯 Python BM25 实现 |
| [`sentence-transformers`](https://github.com/UKPLab/sentence-transformers) | CrossEncoder 接口 |
| [`BAAI/bge-reranker-base`](https://huggingface.co/BAAI/bge-reranker-base) | 重排序模型 |
| [硅基流动](https://siliconflow.cn) | Qwen2.5-7B-Instruct API |
| [Gradio](https://gradio.app) | Web UI |
| SQuAD v1.1 evaluate-v1.1.py | EM/F1 归一化逻辑参考 |

本作业未直接复用他人完整代码；所有模块均自实现，仅 import 上述开源库的标准 API。开发期使用 Claude Code 辅助代码编写与调试。
