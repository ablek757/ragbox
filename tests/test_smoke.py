"""不依赖外部 API 的烟雾测试：导入 + 文本切分 + 历史库。"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_imports():
    from src import config, loader, qa_chain, retriever, vector_store, history, cli  # noqa: F401


def test_split_text_basic():
    from src.loader import split_text

    text = "第一句。第二句！第三句？" * 20
    chunks = split_text(text, chunk_size=50, overlap=10)
    assert chunks
    assert all(len(c) <= 60 for c in chunks)


def test_split_text_empty():
    from src.loader import split_text

    assert split_text("", 100, 10) == []
    assert split_text("   ", 100, 10) == []


def test_history_roundtrip(monkeypatch, tmp_path):
    db = tmp_path / "h.db"
    monkeypatch.setenv("HISTORY_DB_PATH", str(db))
    import importlib

    from src import config as cfg_mod

    importlib.reload(cfg_mod)
    from src import history as h_mod

    importlib.reload(h_mod)

    h_mod.init_db()
    cid = h_mod.create_conversation("t1")
    h_mod.add_message(cid, "user", "你好")
    h_mod.add_message(cid, "assistant", "你好！", sources=[{"source": "a.md", "text": "x", "score": 0.9}])

    msgs = h_mod.get_messages(cid)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["sources"][0]["source"] == "a.md"

    h_mod.rename_conversation(cid, "renamed")
    assert h_mod.list_conversations()[0]["title"] == "renamed"

    h_mod.delete_conversation(cid)
    assert h_mod.list_conversations() == []
