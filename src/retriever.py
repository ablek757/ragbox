"""混合检索：BM25 关键词 + 向量召回 + RRF 融合 + 可选 reranker。"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import jieba

from . import config, vector_store


def _tokenize(text: str) -> list[str]:
    return [t for t in jieba.lcut(text) if t.strip()]


def _build_bm25(corpus_texts: list[str]):
    from rank_bm25 import BM25Okapi

    tokenized = [_tokenize(t) for t in corpus_texts]
    return BM25Okapi(tokenized)


def _vector_candidates(query: str, n: int) -> list[dict]:
    return vector_store.search(query, top_k=n)


def _bm25_candidates(query: str, n: int) -> list[dict]:
    col = vector_store.get_collection()
    if col.count() == 0:
        return []
    data = col.get()
    docs = data.get("documents", []) or []
    metas = data.get("metadatas", []) or []
    if not docs:
        return []
    bm25 = _build_bm25(docs)
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:n]
    return [
        {
            "text": docs[i],
            "source": metas[i].get("source", "") if i < len(metas) else "",
            "score": float(s),
        }
        for i, s in ranked
        if s > 0
    ]


def _rrf_fuse(rank_lists: list[list[dict]], k: int) -> list[dict]:
    fused: dict[str, dict[str, Any]] = {}
    for ranks in rank_lists:
        for rank, item in enumerate(ranks):
            key = f"{item['source']}::{item['text'][:64]}"
            entry = fused.setdefault(key, {"item": item, "score": 0.0})
            entry["score"] += 1.0 / (k + rank + 1)
    merged = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
    out = []
    for e in merged:
        item = dict(e["item"])
        item["score"] = e["score"]
        out.append(item)
    return out


@lru_cache(maxsize=1)
def _get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.RERANKER_MODEL)


def _rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    if not candidates:
        return []
    model = _get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)
    for c, s in zip(candidates, scores, strict=False):
        c["score"] = float(s)
    return sorted(candidates, key=lambda c: c["score"], reverse=True)[:top_k]


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    """混合检索入口。返回前 top_k 个最相关片段。"""
    k = top_k or config.TOP_K
    if not config.USE_HYBRID_SEARCH:
        hits = _vector_candidates(query, k)
    else:
        n = max(config.HYBRID_CANDIDATES, k * 4)
        vec = _vector_candidates(query, n)
        bm = _bm25_candidates(query, n)
        hits = _rrf_fuse([vec, bm], config.RRF_K)

    if config.USE_RERANKER and hits:
        hits = _rerank(query, hits[: max(len(hits), k * 4)], k)
    else:
        hits = hits[:k]
    return hits
