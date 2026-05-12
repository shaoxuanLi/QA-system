# Wiki RAG QA System

智能问答系统大作业实现：基于维基百科语料的检索增强生成 (RAG) 问答。

- **检索器**：BM25 (`bm25s`)
- **额外功能**：Cross-Encoder 重排序器 (`BAAI/bge-reranker-base`)
- **生成器**：硅基流动 API · `Qwen/Qwen2.5-7B-Instruct`
- **UI**：Gradio
- **评估**：EM / F1（SQuAD 风格归一化）

---

## 目录结构

```
QA-sys/
├── config.yaml                # 全部可调参数
├── requirements.txt
├── README.md
├── wiki18_100w.jsonl          # 语料 (~14GB, 1M 段落)，不在提交包中
├── dev.jsonl / test.jsonl     # 验证 / 测试集
├── src/
│   ├── retriever/             # BaseRetriever + BM25Retriever
│   ├── reranker/              # CrossEncoderReranker
│   ├── generator/             # SiliconFlowLLM (chat / chat_batch)
│   ├── rag/                   # RAGPipeline + prompts
│   ├── evaluation/            # EM / F1
│   └── utils/                 # config / jsonl / CorpusReader(按行偏移读语料)
├── scripts/
│   ├── build_index.py         # 建 BM25 索引 + 行偏移
│   ├── run_inference.py       # 跑 dev / test，生成 outputs/{dev,test}.txt
│   └── evaluate.py            # 与 dev.jsonl 对齐算 EM / F1
├── ui/app.py                  # Gradio Web 界面
├── indices/                   # BM25 索引 + offsets（运行后生成）
└── outputs/                   # dev.txt / test.txt（运行后生成）
```

---

## 环境

```bash
pip install -r requirements.txt
```

## 配置 API Key

打开 `config.yaml`，把 `generator.api_key` 填上你的硅基流动 key；或者：

```bash
export SILICONFLOW_API_KEY="sk-xxxx"
```

## 一次性建索引

```bash
python -m scripts.build_index --config config.yaml
```

会生成：
- `indices/bm25/` — BM25 索引
- `indices/offsets.npy` — 文档 id → 文件字节偏移，避免把 14GB 语料全部读入内存

支持 `--skip-bm25` / `--skip-offsets` 只重建其中一项。

## 跑验证集 / 测试集

```bash
# 验证集
python -m scripts.run_inference --split dev  --config config.yaml
# 测试集
python -m scripts.run_inference --split test --config config.yaml
```

输出 `outputs/dev.txt` 与 `outputs/test.txt`，每行一个答案。

常用开关：
- `--no-rerank`：关闭重排序器（用于消融）
- `--top-k-retrieve N` / `--top-k-final N`：覆盖 BM25 召回数 / 进入 LLM 的段落数
- `--limit N`：只跑前 N 条调试
- `--dump-jsonl path`：把 `(question, prediction, retrieved)` 写到 jsonl，方便报告里做 case study

## 评估

```bash
python -m scripts.evaluate --pred outputs/dev.txt --gold dev.jsonl
```

输出 EM / F1。

## Web UI

```bash
python -m ui.app --config config.yaml
```

打开 http://127.0.0.1:7860，可以实时调 `top_k`、开关 reranker、查看检索到的段落和最终答案。

---

## 消融建议

复用 `--no-rerank` 与不同 `--top-k-final` 跑同一份 `dev.jsonl`，对比 EM/F1：

| 配置 | 命令 |
|---|---|
| BM25 only, top1 | `--no-rerank --top-k-final 1` |
| BM25 only, top5 | `--no-rerank --top-k-final 5` |
| BM25 + Reranker, top5 | (默认) |
| BM25 + Reranker, top1 | `--top-k-final 1` |

---

## 参考 / 致谢

- `bm25s`: 高性能纯 Python BM25 实现
- `sentence-transformers`: CrossEncoder 接口
- `BAAI/bge-reranker-base`: 重排序模型
- 评估脚本归一化逻辑参考 SQuAD v1.1 官方实现
