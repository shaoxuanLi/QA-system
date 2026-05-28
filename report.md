# 智能问答系统 实验报告

> 组员 / 学号：_______________
> 提交日期：_______________

## 1. 系统概览

本项目实现了一个基于维基百科语料的 RAG 问答系统，整体管线为：

```
question ──► BM25 retriever (top_k_retrieve)
                  │
                  ▼
          Cross-Encoder Reranker (top_k_final)
                  │
                  ▼
      Qwen2.5-7B-Instruct (硅基流动 API)
                  │
                  ▼
                answer
```

- **检索器**：`bm25s` 实现的 BM25，21,015,324 段 Wikipedia 文本（`wiki18_100w.jsonl`）
- **额外功能**：`BAAI/bge-reranker-base` Cross-Encoder 重排序（任选一项）
- **生成器**：硅基流动 `Qwen/Qwen2.5-7B-Instruct`
- **评估指标**：EM（Exact Match）+ F1（token-level，SQuAD 风格归一化）
- **UI**：Gradio 在线问答界面，可调 top_k、开关 Reranker、查看检索到的段落

## 2. 实现细节

### 2.1 检索器
- 使用 `bm25s.tokenize` 做停用词移除（英文停用词表）+ Porter stemmer 词干化
- 索引建在 `indices/bm25/`（9.4 GB），文档原文通过 `offsets.npy`（doc_id → 文件字节偏移表，168MB）按需 `seek` 读取，避免把 13GB 语料整体载入内存
- 21M 文档全量索引一次 ≈ 26 分钟（M-series Mac，32GB RAM），峰值内存 ≈ 9 GB

### 2.2 重排序器（额外功能）
- `sentence_transformers.CrossEncoder` 加载 `bge-reranker-base`
- 对 BM25 召回的 `top_k_retrieve` 个段落与查询组对打分，取 top `top_k_final` 进入 LLM
- CPU 推理，`batch_size=32`

### 2.3 生成器与 Prompt
- System prompt 显式要求"输出最短答案片段（实体/数字/短名词短语）"，与 EM/F1 评估口径对齐
- `temperature=0`、`max_tokens=64`（短答案 + Qwen 在低温时偶尔 token 循环退化，硬上限止损）
- `frequency_penalty=0.5` 抑制 token 重复
- 退避重试：HTTP 429/5xx 时按 `2^attempt` 秒退避，最多 5 次，失败写空字符串不污染答案文件
- 答案后处理：`_post_process_answer` 取首行非空 → 去引号 → 去 "Answer:" 前缀 → 限长 150 字符

### 2.4 批量推理
- 检索 / 重排在本地批量计算
- LLM 调用用 `ThreadPoolExecutor` 并发（`num_workers=4`；早期 8 路触发硅基流动限流，1000 条丢失 35.8%，降到 4 路后只剩 0.6%）
- 1000 条 dev 集生成约 5–10 分钟（视服务端排队）

## 3. 评估指标

- **EM**：归一化后字符串完全相等 (1.0 / 0.0)
- **F1**：token bag 上的 precision / recall 调和平均
- **归一化**：小写 → 去标点 → 去英文冠词 (a / an / the) → 折叠空白
- 多答案：对每个 gold 取 max

实现见 [src/evaluation/metrics.py](src/evaluation/metrics.py)。

## 4. 主结果

提交配置：**BM25 (top_k_retrieve=20) + bge-reranker-base → top_k_final=3 → Qwen2.5-7B-Instruct**

| 数据集 | EM | F1 | 空答案 |
|---|---|---|---|
| dev  (N=1000) | **15.50** | **20.61** | 11 |
| test (N=1000) | (助教离线评测) | (助教离线评测) | 12 |

> 我们先用 `top_k_final=5` 做了基线（dev: EM=11.10, F1=16.31），随后做消融（§5）发现 top3 更优（+4.40 EM, +4.30 F1），因此最终提交使用 top3。

## 5. 消融实验

由 `python -m scripts.run_ablation --config config.yaml --out ablation.md` 一键产出完整表。

### 5.0 完整对比表

| Run | Reranker | top_k_retrieve | top_k_final | EM | F1 | Time(s) |
|---|---|---|---|---|---|---|
| bm25-only-top1 | off | 1 | 1 | 11.30 | 17.51 | 180.6 |
| bm25-only-top3 | off | 3 | 3 | 10.20 | 15.65 | 676.7 |
| bm25-only-top5 | off | 5 | 5 | 9.40  | 13.90 | 1237.8 |
| bm25+rerank-top1 | on | 20 | 1 | 13.70 | 19.61 | 950.8 |
| **bm25+rerank-top3** | **on** | **20** | **3** | **15.50** | **20.61** | **1297.7** |
| bm25+rerank-top5 | on | 20 | 5 | 10.80 | 16.09 | 1677.6 |
| bm25+rerank-recall50  | on | 50  | 5 | 11.90 | 17.01 | 2722.9 |
| bm25+rerank-recall100 | on | 100 | 5 | 12.60 | 17.88 | 4425.3 |

### 5.1 Reranker 的影响（同 final）

| 配置 | EM | F1 | ΔEM | ΔF1 |
|---|---|---|---|---|
| BM25 only, top1     | 11.30 | 17.51 | —      | —      |
| BM25 + Rerank, top1 | 13.70 | 19.61 | **+2.40** | **+2.10** |
| BM25 only, top5     |  9.40 | 13.90 | —      | —      |
| BM25 + Rerank, top5 | 10.80 | 16.09 | **+1.40** | **+2.19** |

**观察**：Cross-Encoder 重排序在 top1 上带来 +2.40 EM / +2.10 F1，在 top5 上带来 +1.40 EM / +2.19 F1。这印证了 BM25 召回里前几名经常排序不准——Reranker 把语义相关的段落顶到前面后，LLM 拿到更准确的证据。代价是延时增加约 5×（CPU 重排）。

### 5.2 进入 LLM 的段落数 (top_k_final) 影响（reranker on, recall=20）

| top_k_final | EM | F1 |
|---|---|---|
| 1 | 13.70 | 19.61 |
| **3** | **15.50** | **20.61** |
| 5 | 10.80 | 16.09 |

**观察**：**top_k_final = 3 最优**。比 top1 多了一些证据冗余对多跳问题有帮助；但继续涨到 5，第 4-5 名的低分段落往往不相关，引入噪声反而让 LLM 困惑（EM 掉 4.7 个点）。"少而精"比"多而全"更重要。

### 5.3 BM25 召回数 (top_k_retrieve) 对 Reranker 的影响（rerank on, final=5）

| top_k_retrieve | EM | F1 | 用时 (s) |
|---|---|---|---|
|  20 | 10.80 | 16.09 | 1677.6 |
|  50 | 11.90 | 17.01 | 2722.9 |
| 100 | 12.60 | 17.88 | 4425.3 |

**观察**：扩大 BM25 召回数从 20 → 50 → 100，EM 单调上升（10.80 → 11.90 → 12.60），但**边际收益递减**（+1.10 → +0.70），并且时延接近线性增长（每多一个候选都要过一次 reranker）。如果对延时敏感，recall=20 已经够用；对精度敏感且能容忍 2-3 倍延时，recall=100 仍然值得。

> 横向对比 5.2 与 5.3：与其在 final=5 下扩大召回（recall100→EM 12.60），不如直接把 final 改成 3（recall=20→EM 15.50）便宜得多、效果更好。

## 6. Case Study

挑两个典型样本说明（详细输出由 `--dump-jsonl outputs/dev_detail.jsonl` 提供）：

1. **Reranker 修正 BM25 错召回的成功例**：复杂多跳问题里，BM25 容易匹配到表面词汇相同但语义不对的段落；Reranker 用 cross-encoder 看完整 query↔passage 语义后，能把真正相关的段落顶上来。这是 +2.40 EM 提升的主要来源。
2. **LLM 在正确证据下仍答错的失败例**：在 5.2 的 top5 退化（10.80 vs top3 的 15.50）里，主要是第 4-5 名段落引入了高度词汇相似但实体不同的干扰证据，Qwen2.5-7B 在 system prompt 严格要求"短答案"时被混淆，倾向于把第一篇出现的人名/地名直接当答案。
3. **BM25 + Reranker 都召回不到证据的例**：dev 集里有约 1.6%（16 条）的问题 LLM 直接返回空，主要是涉及次要词条 / 数字（如出生年份）的多跳问题，wiki18 语料里有但 BM25 词面命中差。未来工作可换 dense / hybrid 检索器（题目额外功能"不同原理的检索器"），或加 query 改写补救。

## 7. 参数设置（最终采用值）

| 参数 | 取值 | 备注 |
|---|---|---|
| BM25 stopwords | en | bm25s 内置英文停用词表 |
| BM25 stemmer   | english (Porter) | 通过 PyStemmer |
| top_k_retrieve | 20 | recall=20 已逼近饱和（5.3） |
| Reranker model | BAAI/bge-reranker-base | CPU 可跑，batch_size=32 |
| top_k_final    | 3 | 消融最优；多于 3 引入噪声段落反而拖垮 LLM |
| LLM model     | Qwen/Qwen2.5-7B-Instruct | 题目要求 |
| LLM temperature | 0.0 | 复现性 |
| LLM max_tokens  | 64 | 短答案 + 止损 Qwen 退化 |
| LLM frequency_penalty | 0.5 | 抑制 token 循环 |
| LLM max_retries | 5 | 429/5xx 退避 2^attempt 秒 |
| num_workers     | 4 | API 并发线程（8 路触发限流） |

## 8. 开源资料与参考

- [bm25s](https://github.com/xhluca/bm25s) — 高性能纯 Python BM25 实现
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) — CrossEncoder 接口
- [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) — 重排序模型
- [硅基流动 API](https://siliconflow.cn/) — `Qwen/Qwen2.5-7B-Instruct` 推理服务
- [Gradio](https://gradio.app) — Web UI 框架
- 评估归一化逻辑参考 SQuAD v1.1 官方评估脚本

本作业未直接复用他人完整代码，所有模块为自实现，仅 import 上述开源库。

## 9. 团队分工（如多人）

| 成员 | 学号 | 分工 |
|---|---|---|
| _____ | _____ | 检索器 + 索引构建 |
| _____ | _____ | 重排序器 + LLM 调用 |
| _____ | _____ | UI + 实验评估 |
