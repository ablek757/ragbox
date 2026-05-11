"""命令行工具：批量入库 / 问答 / 查看 / 清空。

用法：
    python -m src.cli ingest path/to/file_or_dir [path2 ...]
    python -m src.cli url https://example.com
    python -m src.cli ask "你的问题"
    python -m src.cli list
    python -m src.cli clear
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config, qa_chain, vector_store
from .loader import chunk_documents, load_file, load_from_url

SUPPORTED_EXT = {".pdf", ".docx", ".md", ".markdown", ".txt"}


def _collect_files(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            out.extend(f for f in path.rglob("*") if f.suffix.lower() in SUPPORTED_EXT)
        elif path.is_file():
            out.append(path)
        else:
            print(f"[skip] 路径不存在: {p}", file=sys.stderr)
    return out


def cmd_ingest(args: argparse.Namespace) -> int:
    files = _collect_files(args.paths)
    if not files:
        print("没有找到可入库的文件")
        return 1
    docs = []
    for f in files:
        try:
            docs.append(load_file(f))
            print(f"[ok] {f}")
        except Exception as e:
            print(f"[err] {f}: {e}", file=sys.stderr)
    if not docs:
        return 1
    chunks = chunk_documents(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    n = vector_store.add_chunks(chunks)
    print(f"完成：{len(docs)} 个文件，{n} 个文本块")
    return 0


def cmd_url(args: argparse.Namespace) -> int:
    doc = load_from_url(args.url)
    chunks = chunk_documents([doc], config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    n = vector_store.add_chunks(chunks)
    print(f"完成：{args.url} -> {n} 个文本块")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    stream, hits = qa_chain.stream_answer(args.question)
    for delta in stream:
        sys.stdout.write(delta)
        sys.stdout.flush()
    print()
    if hits and args.show_sources:
        print("\n--- 引用 ---")
        for i, h in enumerate(hits, 1):
            print(f"[{i}] {h['source']} (score={h['score']:.3f})")
    return 0


def cmd_list(_: argparse.Namespace) -> int:
    s = vector_store.stats()
    print(f"文档数：{s['sources']}，文本块：{s['chunks']}")
    for src in vector_store.list_sources():
        print(f"  - {src}")
    return 0


def cmd_clear(args: argparse.Namespace) -> int:
    if not args.yes:
        ans = input("确定清空整个知识库？(y/N) ").strip().lower()
        if ans != "y":
            print("已取消")
            return 1
    vector_store.reset_store()
    print("已清空")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kb", description="AI 个人知识库 CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="导入文件或目录")
    p_ing.add_argument("paths", nargs="+")
    p_ing.set_defaults(func=cmd_ingest)

    p_url = sub.add_parser("url", help="抓取 URL 并入库")
    p_url.add_argument("url")
    p_url.set_defaults(func=cmd_url)

    p_ask = sub.add_parser("ask", help="问答")
    p_ask.add_argument("question")
    p_ask.add_argument("--show-sources", action="store_true")
    p_ask.set_defaults(func=cmd_ask)

    sub.add_parser("list", help="列出已收录文档").set_defaults(func=cmd_list)

    p_clr = sub.add_parser("clear", help="清空知识库")
    p_clr.add_argument("-y", "--yes", action="store_true")
    p_clr.set_defaults(func=cmd_clear)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
