"""Streamlit 入口：python -m streamlit run app.py"""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from src import config, history, qa_chain, vector_store
from src.loader import Document, chunk_documents, load_file, load_from_url

st.set_page_config(page_title="AI 个人知识库", page_icon="📚", layout="wide")
history.init_db()


def ingest_documents(docs: list[Document]) -> int:
    chunks = chunk_documents(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    return vector_store.add_chunks(chunks)


def ensure_active_conversation() -> str:
    if st.session_state.get("conv_id"):
        return st.session_state["conv_id"]
    convs = history.list_conversations()
    cid = convs[0]["id"] if convs else history.create_conversation()
    st.session_state["conv_id"] = cid
    return cid


def render_source_block(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("查看引用"):
        for i, s in enumerate(sources, 1):
            st.markdown(f"**[{i}] {s.get('source','')}** · 相似度 {s.get('score',0):.3f}")
            text = s.get("text", "")
            st.caption(text[:300] + ("..." if len(text) > 300 else ""))


def conversations_panel() -> None:
    st.subheader("💬 会话")
    convs = history.list_conversations()
    if st.button("➕ 新建会话", use_container_width=True):
        st.session_state["conv_id"] = history.create_conversation()
        st.rerun()

    active = ensure_active_conversation()
    for c in convs:
        cols = st.columns([5, 1])
        label = ("● " if c["id"] == active else "  ") + c["title"]
        if cols[0].button(label, key=f"sw-{c['id']}", use_container_width=True):
            st.session_state["conv_id"] = c["id"]
            st.rerun()
        if cols[1].button("✕", key=f"rm-{c['id']}"):
            history.delete_conversation(c["id"])
            if c["id"] == active:
                st.session_state.pop("conv_id", None)
            st.rerun()

    if convs:
        with st.expander("重命名当前会话"):
            cur = next((c for c in convs if c["id"] == active), None)
            new_title = st.text_input("标题", value=cur["title"] if cur else "")
            if st.button("保存标题"):
                history.rename_conversation(active, new_title or "未命名")
                st.rerun()


def knowledge_panel() -> None:
    st.subheader("📥 知识库")
    uploaded = st.file_uploader(
        "上传文档",
        type=["pdf", "docx", "md", "markdown", "txt"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("导入上传的文件", use_container_width=True):
        docs = []
        with st.spinner("解析文档..."):
            for f in uploaded:
                suffix = Path(f.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
                try:
                    doc = load_file(tmp_path)
                    doc.source = f.name
                    docs.append(doc)
                except Exception as e:
                    st.error(f"{f.name}: {e}")
        if docs:
            with st.spinner("生成向量并入库..."):
                n = ingest_documents(docs)
            st.success(f"已导入 {len(docs)} 个文件，共 {n} 个文本块")

    url = st.text_input("从 URL 导入", placeholder="https://...")
    if url and st.button("抓取并导入", use_container_width=True):
        try:
            with st.spinner("抓取网页..."):
                doc = load_from_url(url)
            with st.spinner("入库..."):
                n = ingest_documents([doc])
            st.success(f"导入成功，共 {n} 个文本块")
        except Exception as e:
            st.error(f"抓取失败：{e}")

    s = vector_store.stats()
    c1, c2 = st.columns(2)
    c1.metric("文档数", s["sources"])
    c2.metric("文本块", s["chunks"])

    sources = vector_store.list_sources()
    if sources:
        with st.expander("已收录文档"):
            for src in sources:
                cols = st.columns([4, 1])
                cols[0].write(src)
                if cols[1].button("删除", key=f"del-{src}"):
                    vector_store.delete_source(src)
                    st.rerun()
        if st.button("⚠️ 清空知识库", use_container_width=True):
            vector_store.reset_store()
            st.rerun()


def sidebar() -> None:
    with st.sidebar:
        conversations_panel()
        st.divider()
        knowledge_panel()
        st.divider()
        st.caption(
            f"LLM: {config.LLM_PROVIDER}  |  "
            f"Hybrid: {'on' if config.USE_HYBRID_SEARCH else 'off'}  |  "
            f"Rerank: {'on' if config.USE_RERANKER else 'off'}"
        )


def chat_panel() -> None:
    st.title("📚 AI 个人知识库问答助手")
    cid = ensure_active_conversation()
    messages = history.get_messages(cid)

    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            render_source_block(msg.get("sources") or [])

    question = st.chat_input("基于知识库提问...")
    if not question:
        return

    history.add_message(cid, "user", question)
    with st.chat_message("user"):
        st.markdown(question)

    chat_history = [{"role": m["role"], "content": m["content"]} for m in messages]

    with st.chat_message("assistant"):
        try:
            stream, hits = qa_chain.stream_answer(question, history=chat_history)
            placeholder = st.empty()
            full = ""
            for delta in stream:
                full += delta
                placeholder.markdown(full + "▌")
            placeholder.markdown(full)
            render_source_block(hits)
            history.add_message(cid, "assistant", full, sources=hits)

            convs = {c["id"]: c for c in history.list_conversations()}
            if convs.get(cid, {}).get("title") == "新对话":
                history.rename_conversation(cid, question[:20])
        except Exception as e:
            st.error(f"出错：{e}")


def main() -> None:
    sidebar()
    chat_panel()


if __name__ == "__main__":
    main()
