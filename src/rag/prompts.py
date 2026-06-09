"""提示词模板。三套模式：
- strict (评分用 baseline)：直接要求最短答案
- cot    (评分用强化版)：few-shot + 推理 + Final Answer 提取
- friendly (UI demo)：允许完整自然语言句子
"""

# ------------------------------------------------------------------
# strict: 老版 baseline (dev EM 15.5 的那一套)
# ------------------------------------------------------------------
STRICT_SYSTEM_PROMPT = (
    "You are a precise question answering assistant. "
    "Answer the user's question using ONLY the provided passages. "
    "Output the SHORTEST possible answer span (an entity, number, or short noun phrase). "
    "Do not add explanations, do not repeat the question, do not use full sentences. "
    "If the passages do not contain the answer, output your best guess as a short span."
)


def _build_strict_user_prompt(question: str, passages: list[str]) -> str:
    ctx = "\n\n".join(f"[Passage {i+1}] {p}" for i, p in enumerate(passages))
    return (
        f"Passages:\n{ctx}\n\n"
        f"Question: {question}\n"
        f"Answer (short span only):"
    )


# ------------------------------------------------------------------
# cot: few-shot + reasoning + 最终答案抽取
# 设计要点：
# - 5 个 few-shot 示例覆盖：人名、地名、数字、年份、机构、多答案
# - 强制 "Reasoning: ... Final Answer: <span>" 格式
# - 后处理只提取 Final Answer 之后的内容
# - 让 model "想一下" 再给短答案，多跳问题会受益
# ------------------------------------------------------------------
COT_SYSTEM_PROMPT = (
    "You are a careful question answering assistant. "
    "Read the provided passages and answer the user's question. "
    "First identify which passage contains the answer, then write a single short answer span. "
    "Strictly follow this two-line format:\n"
    "Reasoning: <one short sentence about which passage and what the answer is>\n"
    "Final Answer: <the shortest possible answer span — a name, number, date, or noun phrase>\n"
    "\n"
    "Rules for Final Answer:\n"
    "- ALWAYS write full names (e.g. 'Ernest Hemingway', NEVER 'Ernest Hem' or 'Hemingway')\n"
    "- NO articles ('a'/'an'/'the') at the start\n"
    "- NO trailing punctuation\n"
    "- NO extra qualifiers like 'in 1945' for a person/place question\n"
    "- If multiple answers are valid, give the most specific one from the passages"
)


_FEW_SHOT_EXAMPLES = """
Example 1
Passages:
[Passage 1] "The Old Man and the Sea"\nThe Old Man and the Sea is a short novel written by Ernest Hemingway in Cuba in 1951 and published in 1952.
Question: Who wrote The Old Man and the Sea?
Reasoning: Passage 1 directly states the novel was written by Ernest Hemingway.
Final Answer: Ernest Hemingway

Example 2
Passages:
[Passage 1] "Canberra"\nCanberra is the capital city of Australia. With a population of 431,380, it is Australia's largest inland city.
Question: What is the capital of Australia?
Reasoning: Passage 1 states Canberra is the capital city of Australia.
Final Answer: Canberra

Example 3
Passages:
[Passage 1] "World War II"\nWorld War II ended on 2 September 1945, when Japan formally surrendered aboard the USS Missouri.
Question: When did World War II end?
Reasoning: Passage 1 says World War II ended in 1945, specifically on 2 September 1945.
Final Answer: 1945

Example 4
Passages:
[Passage 1] "Foo Fighters"\nFoo Fighters is an American rock band formed in Seattle in 1994 by Nirvana drummer Dave Grohl.
Question: Who founded the Foo Fighters?
Reasoning: Passage 1 says Foo Fighters was formed by Dave Grohl.
Final Answer: Dave Grohl

Example 5
Passages:
[Passage 1] "Letters to a Young Mathematician"\nLetters to a Young Mathematician (2006) is considered an update to G. H. Hardy's 1940 essay A Mathematician's Apology.
Question: Letters to a Young Mathematician is considered an update to an essay written in what year?
Reasoning: Passage 1 says it is an update to G. H. Hardy's 1940 essay.
Final Answer: 1940
"""


def _build_cot_user_prompt(question: str, passages: list[str]) -> str:
    ctx = "\n\n".join(f"[Passage {i+1}] {p}" for i, p in enumerate(passages))
    return (
        f"{_FEW_SHOT_EXAMPLES}\n"
        f"Now answer this question.\n\n"
        f"Passages:\n{ctx}\n\n"
        f"Question: {question}\n"
        f"Reasoning:"
    )


# ------------------------------------------------------------------
# friendly: UI demo 用，完整自然句子
# ------------------------------------------------------------------
FRIENDLY_SYSTEM_PROMPT = (
    "You are a helpful question answering assistant grounded in retrieved Wikipedia passages. "
    "Read the passages carefully and answer the user's question with a clear, complete sentence. "
    "Prefer 1-2 sentences. Always spell names in full (e.g. Ernest Hemingway, not Ernest Hem). "
    "If multiple passages disagree, cite the most specific one. "
    "If the passages truly do not contain the answer, say so briefly instead of guessing."
)


def _build_friendly_user_prompt(question: str, passages: list[str]) -> str:
    ctx = "\n\n".join(f"[Passage {i+1}]\n{p}" for i, p in enumerate(passages))
    return (
        f"Here are passages retrieved from Wikipedia:\n\n{ctx}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


# ------------------------------------------------------------------
# 统一入口
# ------------------------------------------------------------------
def build_messages(
    question: str,
    passages: list[str],
    mode: str = "strict",
) -> list[dict]:
    if mode == "cot":
        return [
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {"role": "user", "content": _build_cot_user_prompt(question, passages)},
        ]
    if mode == "friendly":
        return [
            {"role": "system", "content": FRIENDLY_SYSTEM_PROMPT},
            {"role": "user", "content": _build_friendly_user_prompt(question, passages)},
        ]
    return [
        {"role": "system", "content": STRICT_SYSTEM_PROMPT},
        {"role": "user", "content": _build_strict_user_prompt(question, passages)},
    ]


# 向后兼容
SYSTEM_PROMPT = STRICT_SYSTEM_PROMPT
build_user_prompt = _build_strict_user_prompt
