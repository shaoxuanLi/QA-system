"""提示词模板。统一放这里方便消融对比。"""

SYSTEM_PROMPT = (
    "You are a precise question answering assistant. "
    "Answer the user's question using ONLY the provided passages. "
    "Output the SHORTEST possible answer span (an entity, number, or short noun phrase). "
    "Do not add explanations, do not repeat the question, do not use full sentences. "
    "If the passages do not contain the answer, output your best guess as a short span."
)


def build_user_prompt(question: str, passages: list[str]) -> str:
    ctx = "\n\n".join(f"[Passage {i+1}] {p}" for i, p in enumerate(passages))
    return (
        f"Passages:\n{ctx}\n\n"
        f"Question: {question}\n"
        f"Answer (short span only):"
    )


def build_messages(question: str, passages: list[str]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, passages)},
    ]
