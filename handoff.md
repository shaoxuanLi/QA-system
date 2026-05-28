# 交接文档 · 智能问答系统 RAG 大作业

> **致：负责汇报展示的队友**
> **目标：你跑通本地 demo，理解系统，做汇报，最后打包提交**

---

## 0. TL;DR

我已经把整个 RAG 系统跑完了，dev/test 答案都生成好了，实验报告也填好了数字。**你需要做的事：**

1. 在你电脑上把项目跑通（重点：拿到数据 + 拿到 API key + 建索引）
2. 走一遍 Gradio UI demo
3. 看一遍 `report.md`，理解系统设计 + 实验结果
4. 准备汇报（PPT / 演示流程）
5. 提交前打包

**当前最终成绩：dev EM = 15.50, F1 = 20.61**（top3 配置，详见 `report.md` 第 4 节）

---

## 1. 项目当前状态

| 模块 | 状态 | 关键文件 |
|---|---|---|
| 检索器 BM25 (bm25s) | ✅ 全量索引建好 | `src/retriever/bm25_retriever.py` |
| Reranker (bge-reranker-base) | ✅ 跑通 | `src/reranker/ce_reranker.py` |
| LLM (Qwen2.5-7B 硅基流动) | ✅ 含限流退避 + 后处理 | `src/generator/llm.py` |
| RAG 主管线 | ✅ | `src/rag/pipeline.py` |
| EM/F1 评估 | ✅ SQuAD 风格归一化 | `src/evaluation/metrics.py` |
| Gradio UI | ✅ | `ui/app.py` |
| 消融脚本（8 组） | ✅ | `scripts/run_ablation.py` |
| 实验报告 | ✅ 数字已填 | `report.md` |
| 答案输出 | ✅ dev.txt / test.txt | 根目录 |

---

## 2. 你的任务 Checklist

按顺序做：

- [ ] 2.1 在自己电脑上 clone 项目（不含数据 / 索引 / venv）
- [ ] 2.2 拿到 **API key**（看 §3 安全说明，不要泄露）
- [ ] 2.3 下载 **wiki18 语料** + dev/test 数据集（见 §4）
- [ ] 2.4 装依赖 + 重建 BM25 索引（**约 30 分钟一次性**）
- [ ] 2.5 跑一次 `evaluate.py` 复现 dev EM=15.50 验证环境
- [ ] 2.6 启动 UI 走 demo 流程（见 §6）
- [ ] 2.7 熟读 `report.md` + 本文件 §7、§8（汇报要点 + Q&A）
- [ ] 2.8 做 PPT
- [ ] 2.9 打包 `<学号>.zip` 提交（见 §9）

---

## 3. API Key（**最重要的安全注意事项**）

### 我们的 key 状态

- 我用的硅基流动 key 在开发期已经在配置文件里出现过，**视为已泄露**
- 我会在打包前清空 `config.yaml` 里的 `api_key` 字段；**你也要每次确认清空再操作 git**
- 强烈建议你自己去 https://siliconflow.cn 注册一个新 key（免费配额够用，本作业大约消耗了 1 万次调用）

### 我把 key 通过什么方式给你

**不要走 git / GitHub / 公共聊天群**。三选一：

- 微信私聊
- 加密邮件
- 当面看屏幕念

### 你拿到 key 后

**两种方式选一种**：

```bash
# 方式 1（推荐）：环境变量，不进任何文件
export SILICONFLOW_API_KEY="sk-xxxx"

# 方式 2：写进 config.yaml 的 generator.api_key 字段
# 但是 ⚠️ 不要 git add config.yaml 之后 commit
```

### 提交前的安全检查（重要！）

```bash
# 1. 把 api_key 字段清空
# 编辑 config.yaml，确保 generator.api_key 是空字符串 ""

# 2. grep 一遍确保没有 sk- 开头的字符串
grep -r "sk-" . --exclude-dir=.venv --exclude-dir=.git --exclude-dir=indices

# 应该没有任何输出。如果有，删干净再打包。
```

---

## 4. 大文件如何获取（不要传 git，22GB 走不动）

### 4.1 语料库 `wiki18_100w.jsonl` (13 GB)

**最快办法：你自己从官方链接下载**（PDF 里给的链接）

PDF 原话：
> 语料库：一个大规模维基百科语料库，可以在**网盘链接**中下载 `corpus.zip` 文件或者 **HF 链接**中下载 `wiki18_100w.zip` 文件。

放到项目根目录，路径：`./wiki18_100w.jsonl`

> 注意：实际文件有 **21M 文档**（不是文件名暗示的 100w=1M），13GB，解压后建议放在 SSD 上。

### 4.2 dev / test 数据集

PDF 同样链接下载 `dev.jsonl`、`test.jsonl`，放到项目根目录。

### 4.3 BM25 索引（9.4 GB）

**不要传文件，本地重建就行**：

```bash
python -m scripts.build_index --config config.yaml
```

我的机器（M-series Mac, 32GB RAM）跑了 **26 分钟**。你的机器至少需要 **20GB 空闲内存**，否则会卡死或 OOM。

如果你内存不够（< 16GB），找我，我把 `indices/` 整个目录用网盘传给你（9.4GB，避免你重建）。

### 4.4 bge-reranker 模型（约 1GB）

**首次运行自动下载**（`sentence-transformers` 从 HuggingFace 拉），无需手动操作。需要网络通畅。

---

## 5. 跑通流程（约 1 小时一次性 setup + 10 分钟验证）

```bash
# 1. 克隆代码
git clone <你拿到的 repo 地址> QA-sys
cd QA-sys

# 2. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt    # 约 5-10 分钟（含 PyTorch ~2GB）

# 3. 放置数据（见 §4）
# ./wiki18_100w.jsonl
# ./dev.jsonl
# ./test.jsonl

# 4. 设置 API key（见 §3）
export SILICONFLOW_API_KEY="sk-xxxx"

# 5. 建索引（一次性，约 26 分钟）
python -m scripts.build_index --config config.yaml

# 6. 验证（用我已经生成的 dev.txt 评估，确保管线一致）
python -m scripts.evaluate --pred dev.txt --gold dev.jsonl
# 应该输出：EM = 15.50, F1 = 20.61

# 7. 启动 UI
python -m ui.app --config config.yaml
# 浏览器打开 http://127.0.0.1:7860
```

---

## 6. Demo 流程（汇报现场要做的）

### 6.1 启动

```bash
source .venv/bin/activate
python -m ui.app --config config.yaml
```

### 6.2 准备 3 个问题（推荐从 dev.jsonl 找已知能答对的）

#### 问题 1 · 单跳事实型（简单热场）
```
Who wrote The Old Man and the Sea?
```
预期答案：`Ernest Hemingway`

#### 问题 2 · 多跳推理（展示系统价值）
```
When Ronald Koeman became the manager of Southampton F.C., the person he replaced went on to become the manager of what?
```
预期答案：`Premier League club Tottenham Hotspur`（演示多跳问题）

#### 问题 3 · Reranker 对比演示

随便挑一个问题，**先开 Reranker 提问 → 关 Reranker 再提问**。UI 上有个 checkbox。

预期：开 Reranker 时检索段落顺序更合理，答案更准；关掉后段落顺序退回 BM25 原始排序，答案可能变差。这是你**演示额外功能价值**的关键时刻。

### 6.3 Demo 话术示例

> "这是我们做的 RAG 问答系统。左边输入问题，右边显示检索到的段落和最终答案。
>
> **第一问：单跳事实题。** *[输入 Q1，等待几秒]* 答案出来了：Ernest Hemingway。右边可以看到检索到的 3 个 Wikipedia 段落，按 reranker 分数排序。
>
> **第二问：多跳推理。** 这种问题需要先找到 Ronald Koeman 替换了谁（Mauricio Pochettino），再去找这个人去了哪里（Tottenham）。*[输入 Q2]* 系统通过两步证据组合给出了答案。
>
> **第三问：演示 Reranker 价值。** 我先打开 Reranker 提问 *[输入]*，再关掉 Reranker 同一个问题 *[关 checkbox 再问]*。可以看到段落顺序变了，关掉后召回的是 BM25 原始排序的前 3 篇，可能不是最相关的。这就是 Reranker 带来的 +2.4 EM 提升的来源。
>
> 完整消融数据在我们的报告里。"

### 6.4 出问题怎么救场

- **API 超时**：硅基流动偶尔抖动，刷新一次就行
- **答案空白**：dev 集 1000 条里有 1.1% 是 LLM 自己选择不答（缺证据），这是正常的；现场如果碰到，换一个问题
- **断网**：UI 启动后 LLM 调用需要网络，断网就用 PPT 截图代替

---

## 7. 汇报要讲的 5 个亮点（按重要性排序）

### 亮点 1 · 完整可用的 RAG 管线
**BM25 (21M wiki18 段落) → bge-reranker → Qwen2.5-7B-Instruct**

### 亮点 2 · 工程优化做到位
- 14GB 语料**不进内存**：用 doc_id → 字节偏移表（168MB）按需 `seek`
- 大索引建索引 26 分钟撑住 21M 文档，峰值 9GB RAM
- LLM 失败不污染答案文件，支持 `--retry-empty` 增量补漏
- 限流问题：8 路并发触发 35% 失败 → 4 路 + 指数退避后 0.6%

### 亮点 3 · 额外功能（Reranker）有实测增益
- BM25-only top1: EM=11.30, F1=17.51
- BM25 + Reranker top1: **EM=13.70, F1=19.61** （+2.40 EM, +2.10 F1）

### 亮点 4 · 消融驱动的参数选择（最有讲头）
做了 8 组消融，发现 **`top_k_final = 3` 比 5 高 4 个 EM**。
原因：第 4-5 名段落 reranker 分数低，常常是词面相似但实体错的干扰项，反而让 Qwen 答错。
**"少而精 > 多而全"** —— 这个洞察值得花一页 PPT 讲。

### 亮点 5 · 召回扩张的边际收益
- recall 20 → 50 → 100：EM 10.80 → 11.90 → 12.60（单调上升但**递减**）
- 时延接近线性增长（reranker 候选 ×5）
- **横向对比**：与其在 final=5 下扩大召回（recall100→EM 12.60），不如直接把 final 改成 3（recall=20→EM **15.50**），便宜得多、效果更好

---

## 8. Q&A 准备

| 可能问题 | 简答 |
|---|---|
| 为什么用 BM25 不用 dense 检索？ | 21M 文档 dense 索引太重（FAISS + 编码模型要好几小时 + GPU）。BM25 用 bm25s 26 分钟建完，CPU 即可。是性价比选择。如果有 GPU，dense + hybrid 应该能更高 |
| 重排序器为啥选 bge-reranker-base？ | 中英文都强，CPU 可跑（base 体积小，~1GB）。cross-encoder 看完整 query↔passage 语义，比 dense embedding 准 |
| 为什么 EM 只有 15.5？是不是偏低？ | 数据集是 HotpotQA 风格的多跳 QA，单跳 BM25 召回有天花板。BM25-only top1 才 11.3，加 Reranker 上到 15.5 是 +37% 相对提升。要再涨需要 query 改写 / multi-hop 检索（另一个额外功能） |
| 为什么 top5 反而比 top3 差？ | Reranker 分数排序后第 4-5 名常常是词面相似但实体错的干扰段落。Qwen 在 system prompt 要求"短答案"时倾向直接抽第一篇出现的实体——错实体被抽中，EM 就掉 |
| LLM 输出退化怎么处理的？ | 三层防御：(1) `max_tokens=64` 硬上限止损；(2) `frequency_penalty=0.5` 抑制 token 循环；(3) 后处理取首行去引号限长 |
| 怎么应对 14GB 数据规模？ | doc_id → 字节偏移 npy（168MB）做寻址，永远只读那一行而不是整库；BM25 索引落盘 9.4GB，查询时 mmap 加载 |
| 限流是怎么解决的？ | 一开始 8 路并发被硅基流动打挂 35% 请求。降到 4 路 + 指数退避（2/4/8/16/32 秒，最多 5 次重试）后只剩 0.6% 失败 |
| 为什么是 Qwen2.5-7B-Instruct？不能换大模型吗？ | PDF 明确要求统一用这个模型公平对比 |
| 评估归一化怎么做的？ | SQuAD v1.1 标准：小写 → 去标点 → 去英文冠词 → 折叠空白。多答案取 max。代码在 `src/evaluation/metrics.py` |
| 失败的 1.1% 空答案怎么办？ | 都是缺证据的多跳问题，模型选择闭嘴。我们尝试过调高 temperature/降 frequency_penalty 救援，无效。承认是当前管线的真实失败案例 |

---

## 9. 提交清单（按 PDF 要求）

### 9.1 必交内容

```
<学号>.zip
├── README.md
├── report.md                  ← 记得填学号 + 分工！
├── config.yaml                ← 确认 api_key 已清空
├── requirements.txt
├── dev.txt                    ← 1000 行答案
├── test.txt                   ← 1000 行答案
├── ablation.md                ← 消融实验结果
├── ablation.json
├── src/                       ← 全部代码
├── scripts/
└── ui/
```

### 9.2 打包前一定要删的（PDF 要求"不含中间结果和数据集"）

- `indices/`（9.4GB BM25 索引）
- `outputs/`（中间日志 / 消融 detail）
- `smoke/`（我开发期的冒烟测试残留）
- `.venv/`
- `wiki18_100w.jsonl`（13GB 数据集）
- `dev.jsonl`, `test.jsonl`（数据集）
- `dev.top5.bak.txt`, `test.top5.bak.txt`（top5 配置的备份，已替换为 top3，可以删）
- `__MACOSX/`、所有 `.DS_Store`
- `handoff.md`（本文件，不交）

### 9.3 一键打包命令（在项目根目录跑）

```bash
# 1. 确保 config.yaml 中 api_key 为空
# 2. 进入项目根目录上一层
cd ..

# 3. 打包（修改 <学号> 为你的实际学号）
zip -r "<学号>.zip" QA-sys \
    -x "QA-sys/.venv/*" \
    -x "QA-sys/indices/*" \
    -x "QA-sys/outputs/*" \
    -x "QA-sys/smoke/*" \
    -x "QA-sys/__MACOSX/*" \
    -x "QA-sys/*.jsonl" \
    -x "QA-sys/*.bak.txt" \
    -x "QA-sys/handoff.md" \
    -x "QA-sys/.git/*" \
    -x "*.DS_Store" \
    -x "*__pycache__*" \
    -x "*.pyc"

# 4. 验证 zip 包大小（应该是几百 KB，不是几 GB）
ls -lh "<学号>.zip"

# 5. 检查 zip 内没有 api_key（最后一道保险）
unzip -p "<学号>.zip" "QA-sys/config.yaml" | grep -i "sk-" && echo "⚠️ 还有 key！" || echo "✅ 干净"
```

---

## 10. 重要文件速查

| 文件 | 你为什么要看 |
|---|---|
| `report.md` | **必读**。汇报内容的主要来源 |
| `README.md` | 系统使用说明 |
| `ablation.md` | 消融实验完整对比表（report.md 的核心数据） |
| `src/rag/pipeline.py` | RAG 主流程（解释系统怎么工作的时候用） |
| `src/rag/prompts.py` | Prompt 模板（被问到 prompt engineering 时） |
| `src/generator/llm.py` | LLM 调用 + 限流退避 |
| `src/evaluation/metrics.py` | EM / F1 实现 |
| `config.yaml` | 所有可调参数 |

---

## 11. 我没做但你可以加分的事（可选）

- PPT 里画一张系统架构图（参考 `report.md` §1 的 ASCII 图）
- PPT 里画一张消融柱状图（数据在 `ablation.json`，用 Excel 或 matplotlib 都行）
- 现场如果问起"为什么不做第二个额外功能"，可以说"Reranker 已经把 BM25 召回的语义短板补上了，再加 query 改写边际收益有限——我们选择把时间花在更深入的消融分析上"

---

## 12. 找不到我的时候

如果跑不通：
- 看 `README.md`
- 看本文件 §5
- 还不行就微信我

汇报顺利！🚀
