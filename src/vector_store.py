"""ChromaDB 向量库与本地嵌入模型。"""
from __future__ import annotations

import hashlib
from functools import lru_cache

import chromadb
from chromadb.config import Settings

from . import config


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(
        path=config.VECTOR_STORE_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _make_id(source: str, chunk_id: int, text: str) -> str:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return f"{source}::{chunk_id}::{h}"


def add_chunks(chunks: list[dict]) -> int:
    if not chunks:
        return 0
    col = get_collection()
    texts = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "chunk_id": c["chunk_id"]} for c in chunks]
    ids = [_make_id(c["source"], c["chunk_id"], c["text"]) for c in chunks]
    embeddings = embed(texts)
    col.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(chunks)


def search(query: str, top_k: int | None = None) -> list[dict]:
    col = get_collection()
    if col.count() == 0:
        return []
    k = top_k or config.TOP_K
    q_emb = embed([query])
    res = col.query(query_embeddings=q_emb, n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return [
        {"text": d, "source": m.get("source", ""), "score": 1 - dist}
        for d, m, dist in zip(docs, metas, dists, strict=False)
    ]


def list_sources() -> list[str]:
    col = get_collection()
    if col.count() == 0:
        return []
    data = col.get()
    metas = data.get("metadatas", []) or []
    return sorted({m.get("source", "") for m in metas if m})


def delete_source(source: str) -> int:
    col = get_collection()
    data = col.get(where={"source": source})
    ids = data.get("ids", [])
    if ids:
        col.delete(ids=ids)
    return len(ids)


def reset_store() -> None:
    col = get_collection()
    data = col.get()
    ids = data.get("ids", [])
    if ids:
        col.delete(ids=ids)


def stats() -> dict:
    col = get_collection()
    return {"chunks": col.count(), "sources": len(list_sources())}
