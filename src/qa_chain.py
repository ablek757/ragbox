"""RAG 问答链：检索 + LLM 生成。"""
from __future__ import annotations

from collections.abc import Generator
from functools import lru_cache

from openai import OpenAI

from . import config, retriever

SYSTEM_PROMPT = """你是一个严谨的个人知识库问答助手。请基于下方提供的「参考资料」回答用户的问题。

要求：
1. 优先使用参考资料中的内容作答，并在合适处用 [序号] 标注引用来源；
2. 如果参考资料不足以回答，请明确说明「根据已有资料无法确定」，再补充你的推测；
3. 回答使用与提问相同的语言，结构清晰、简洁。
"""


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    cfg = config.get_llm_config()
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])


def build_context(hits: list[dict]) -> str:
    if not hits:
        return "（无相关资料）"
    return "\n\n".join(
        f"[{i + 1}] 来源：{h['source']}\n{h['text']}" for i, h in enumerate(hits)
    )


def build_messages(question: str, hits: list[dict], history: list[dict] | None = None) -> list[dict]:
    history = history or []
    user_content = (
        f"参考资料：\n{build_context(hits)}\n\n"
        f"用户问题：{question}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_content},
    ]


def answer(question: str, history: list[dict] | None = None, top_k: int | None = None) -> dict:
    hits = retriever.retrieve(question, top_k=top_k)
    messages = build_messages(question, hits, history)
    client = get_client()
    cfg = config.get_llm_config()
    resp = client.chat.completions.create(
        model=cfg["model"], messages=messages, temperature=0.3
    )
    return {"answer": resp.choices[0].message.content, "sources": hits}


def stream_answer(
    question: str, history: list[dict] | None = None, top_k: int | None = None
) -> tuple[Generator[str, None, None], list[dict]]:
    hits = retriever.retrieve(question, top_k=top_k)
    messages = build_messages(question, hits, history)
    client = get_client()
    cfg = config.get_llm_config()

    def gen() -> Generator[str, None, None]:
        stream = client.chat.completions.create(
            model=cfg["model"], messages=messages, temperature=0.3, stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return gen(), hits
