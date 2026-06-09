# Wiki RAG QA · 3 分钟汇报大纲

> **总时长：180 秒 / 6 张幻灯片**
> 节奏：每张 ~30 秒。每张配 1-2 张图片即可。
> 配图来源：项目根目录 `presentation/` 已经生成 4 张 (`fig1`-`fig4`)；其余建议自找的图标在每张下注明。

---

## 🎬 Slide 1：开场 + 任务定义（~25 秒）

**讲什么**：1 句话 RAG 是什么 → 1 句话任务 → 1 句话我们做了什么

**讲稿**：
> "大家好。我们做的是**检索增强生成的开放领域问答系统**。给定一个英文问题，系统从 2100 万段 Wikipedia 文本里**先检索证据，再让 LLM 抽取答案**。
>
> 评估指标是 EM 和 F1。我们最终在 dev 集上拿到 **EM 15.50 / F1 20.69**。
>
> 接下来 3 分钟带大家看系统、看消融、看一个有意思的发现。"

**图片**：
- 🖼️ 左：一个简洁的"问题 → 答案"示意图（自画或图标）
- 🖼️ 右：📁 `presentation/fig1_architecture.png` 缩略图作为预告

**外部图建议**（你来找）：
- Wikipedia logo（W 字标）
- HotpotQA 数据集 logo / 截图

---

## 🎬 Slide 2：系统架构（~30 秒）

**讲什么**：5 个组件的数据流；每个组件 5-10 秒

**讲稿**：
> "系统是经典 RAG 三段式：
>
> **BM25 检索器** 用 `bm25s` 库构建，21M 文档 + 9.4GB 索引，CPU 26 分钟全量建索引。
>
> **Cross-Encoder Reranker** 用 `bge-reranker-base`，对 BM25 召回的 20 篇候选重新打分，跑在 Mac 的 MPS GPU 上，比 CPU 快 1.9 倍。
>
> **LLM 生成器** 用作业指定的 Qwen2.5-7B-Instruct，通过硅基流动 API 调用。
>
> 关键工程亮点：14GB 语料**不进内存**——我们用 doc_id 到字节偏移的索引按需读取。"

**图片**（**只用一张**全屏）：
- 🖼️ **`presentation/fig1_architecture.png`** ✅ 已生成

**外部图建议**（不强求）：
- BM25 / bge-reranker / Qwen / Gradio 4 个 logo（自找）
- Mac M-series GPU 芯片图（如果想强调 MPS 加速）

---

## 🎬 Slide 3：消融实验 — "少而精 > 多而全"（~35 秒）

**讲什么**：8 组消融的主要发现 + 反直觉洞察

**讲稿**：
> "我们跑了 8 组消融，每组在全 1000 条 dev 上评估，总耗时 3 小时。
>
> 三个核心结论：
> 1. **Reranker 全场必胜**：每个 top_k 档位下，加 Reranker 都比 BM25-only 高 2-4 个 EM
> 2. **top_k_final=3 是 sweet spot**，比 top1 高 1.8 EM、比 top5 高 4.7 EM
> 3. **召回扩张回报递减**：recall 从 20 → 100 只涨 1.8 EM 但延时 2.6 倍
>
> 最反直觉的是 top5 比 top3 差——多塞两篇低分段落，反而让 Qwen 这种 7B 小模型被噪声拖垮。"

**图片**：
- 🖼️ **`presentation/fig2_ablation.png`** ✅ 已生成（绿色高亮即最优）

---

## 🎬 Slide 4：工程优化亮点 — 限流 / 退化 / GPU（~30 秒）

**讲什么**：3 个工程难题怎么解决

**讲稿**：
> "工程上遇到过 3 个有教育意义的坑：
>
> **第一个坑：API 限流**。8 路并发 → 1000 条 dev 里 358 条 (35.8%) 因为 429 被丢。降到 4 路 + 加重退避 + 增加 `--retry-empty` 增量补救后只剩 0.6%。
>
> **第二个坑：Qwen2.5-7B 的 token 退化**。低温下偶尔会输出 `'Sym Sym Sym...'` 这种循环。我们用三层防御：max_tokens=64 硬上限 + frequency_penalty + 后处理砍循环。
>
> **第三个坑：CoT/few-shot 没用**。试了精心设计的 5-shot CoT prompt，EM 完全不动。结论是 **7B 模型跟不上复杂 prompt 模板**，这是小模型的能力上限。"

**图片**：
- 🖼️ **`presentation/fig4_journey.png`** ✅ 已生成（时间线 + 关键节点）

**外部图建议**：
- 一张 "429 Too Many Requests" HTTP 错误截图（有助于现场笑场）

---

## 🎬 Slide 5：关键发现 — 瓶颈不是检索，是 LLM（~30 秒）

**讲什么**：最有讲头的一张图 — DeepSeek 对照实验

**讲稿**：
> "我们做了一个对照实验：保持检索完全不变，只把 Qwen2.5-7B 换成 DeepSeek-V4-Pro。
>
> 在前 200 条 dev 上：F1 几乎一样 (19.49 vs 19.42)，但 **EM 多了整整 +5**（13.5 → 18.5）。
>
> 解读：F1 一样说明两个模型答的"对错覆盖范围"接近——也就是说**检索给到的证据是足够的**。但 DeepSeek 更善于精准输出短答案 span，Qwen 经常把答案包在解释里导致 EM 失败。
>
> 所以我们的结论：**系统设计正确，瓶颈在 LLM 选型**。换更现代的 LLM 立刻就能涨 5 EM。"

**图片**：
- 🖼️ **`presentation/fig3_llm_compare.png`** ✅ 已生成

---

## 🎬 Slide 6：Demo + 收尾（~30 秒）

**讲什么**：跑一次 UI demo 收尾，或者展示 UI 截图

**讲稿**：
> "最后是 demo。
>
> *[切到 UI，提问 'Who painted the Mona Lisa?']*
>
> 答案出来：Mona Lisa was painted by Leonardo da Vinci。右边是检索到的 3 个 Wikipedia 段落 + Reranker 分数。
>
> UI 支持**模型切换**：默认用作业指定的 Qwen2.5-7B；可以切到 DeepSeek-V4-Pro 看刚才说的 +5 EM 效果。
>
> 总结：BM25 + Reranker + Qwen，配合工程优化和消融驱动调参，最终 EM 15.50 / F1 20.69。谢谢大家。"

**图片**：
- 🖼️ UI 截图（直接现场跑或截图）
- 🖼️ 最终成绩表（可以从 report.md 截图）

**外部图建议**：
- Gradio logo
- "Thank you" 结尾页（可选）

---

## 📋 现场操作 checklist

**演讲前**（提前 5 分钟）：
- [ ] 启动 UI：`source .venv/bin/activate && python -m ui.app`
- [ ] 等到出现 `[warmup] ready`
- [ ] 打开 http://127.0.0.1:7860 备好提问
- [ ] 准备 2 个候补问题（万一第一个超时）：
  - `Who painted the Mona Lisa?` (主推)
  - `Who invented the telephone?` (多答案，展示 reranker)

**演讲中**：
- 不要照念讲稿，自然讲
- Slide 3、5 是核心，可以多花 5 秒强调
- 万一时间不够，Slide 4 工程坑可以一句话带过

---

## 🎨 配图制作建议

我已经生成了 4 张项目内的图（在 `presentation/` 目录），如果想再丰富：

| 想加的图 | 怎么找 | 用途 |
|---|---|---|
| Wikipedia logo | wikipedia.org 直接截 | Slide 1 任务定义 |
| BM25 / bge / Qwen / Gradio logos | 各官网截 | Slide 2 选型列表 |
| 数据集示例截图 | 从 dev.jsonl 截一个样本 | Slide 1 任务定义 |
| HTTP 429 示意 | Google 图片 "HTTP 429 too many requests" | Slide 4 坑 |
| 反应快慢/MPS vs CPU bar | 已有 `fig4` 含此信息 | Slide 2 GPU 亮点 |
| 中国地图标注高校 logo | 学校 logo | Slide 1 开头 |

如果实在没图，每张 slide 配一张 `fig1`-`fig4` + 文字也完全够用。

---

## 🔑 一句话记住每张幻灯片

| 张 | 一句话 |
|---|---|
| 1 | 我们做了 Wikipedia RAG QA，最终 EM 15.5 |
| 2 | BM25 → bge-Reranker → Qwen，21M docs 不进内存 |
| 3 | 消融发现 top3 > top5，"少而精 > 多而全" |
| 4 | 限流 / 退化 / CoT 失败 3 个工程教训 |
| 5 | 换 DeepSeek +5 EM，瓶颈是 LLM 不是检索 |
| 6 | UI demo + 谢谢 |
