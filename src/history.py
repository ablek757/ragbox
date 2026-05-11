"""SQLite 对话历史持久化。支持多会话管理。"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from . import config


@contextmanager
def _conn():
    Path(config.HISTORY_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(config.HISTORY_DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db() -> None:
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT,
                created_at REAL NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id);
            """
        )


def create_conversation(title: str = "新对话") -> str:
    cid = uuid.uuid4().hex[:12]
    now = time.time()
    with _conn() as c:
        c.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?,?,?,?)",
            (cid, title, now, now),
        )
    return cid


def list_conversations() -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def rename_conversation(cid: str, title: str) -> None:
    with _conn() as c:
        c.execute("UPDATE conversations SET title=?, updated_at=? WHERE id=?", (title, time.time(), cid))


def delete_conversation(cid: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
        c.execute("DELETE FROM conversations WHERE id=?", (cid,))


def add_message(cid: str, role: str, content: str, sources: list[dict] | None = None) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO messages (conversation_id, role, content, sources, created_at) VALUES (?,?,?,?,?)",
            (cid, role, content, json.dumps(sources or [], ensure_ascii=False), time.time()),
        )
        c.execute("UPDATE conversations SET updated_at=? WHERE id=?", (time.time(), cid))


def get_messages(cid: str) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT role, content, sources FROM messages WHERE conversation_id=? ORDER BY id",
            (cid,),
        ).fetchall()
    out = []
    for r in rows:
        msg = {"role": r["role"], "content": r["content"]}
        if r["sources"]:
            try:
                msg["sources"] = json.loads(r["sources"])
            except Exception:
                msg["sources"] = []
        out.append(msg)
    return out
