# 📚 AI Personal Knowledge Base Q&A Assistant

> [中文](./README.md) · **English**

A local knowledge-base Q&A tool based on **RAG (Retrieval-Augmented Generation)**. Upload your PDFs / Word docs / Markdown / web pages, then ask questions in natural language — answers come back with cited sources.

> Stack: Streamlit + ChromaDB + BGE Embedding + BM25 hybrid retrieval + DeepSeek / Qwen

## ✨ Features

- 📄 **Multi-format ingestion**: PDF, Word (.docx), Markdown, TXT, web URLs
- 🔍 **Hybrid retrieval**: BM25 keyword + vector recall + RRF fusion, with optional `bge-reranker` reranking
- 🔌 **Local Chinese embeddings**: `BAAI/bge-small-zh-v1.5`, no API needed
- 💬 **Multi-session management**: SQLite-persisted chat history, switch / rename / delete
- 🗂️ **Document-level management**: delete individual docs, clear the whole KB
- 🖥️ Ships with both a **Streamlit Web UI** and a **command-line CLI**
- 🐳 One-command **Docker Compose** deployment
- 🤖 OpenAI-compatible Chinese-friendly models (DeepSeek / Qwen)
- ✅ GitHub Actions CI (ruff + smoke tests)

## 🚀 Quick Start (Local)

```bash
git clone https://github.com/ablek757/ragbox.git
cd ragbox

python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS / Linux

pip install -r requirements.txt

copy .env.example .env           # then fill in DEEPSEEK_API_KEY
streamlit run app.py
```

A browser opens at `http://localhost:8501`.
On first run the BGE embedding model (~100MB) is downloaded; if you need a HuggingFace mirror: `set HF_ENDPOINT=https://hf-mirror.com`.

## 🐳 Docker One-Command Start

```bash
cp .env.example .env             # fill in API key
docker compose up -d --build
```

Visit `http://localhost:8501`. The HuggingFace cache and ChromaDB data are persisted via volumes, so restarts don't lose anything.

## 🛠️ Command-Line CLI

```bash
# Batch ingest (recursive directories supported)
python -m src.cli ingest ./docs ./notes/note1.md

# Scrape a URL
python -m src.cli url https://example.com/article

# Ask
python -m src.cli ask "What is RAG?" --show-sources

# Inspect / clear
python -m src.cli list
python -m src.cli clear -y
```

## 📁 Project Layout

```
ai-knowledge-base/
├── app.py                      # Streamlit entry point
├── Dockerfile / docker-compose.yml
├── requirements.txt / pyproject.toml
├── .env.example
├── src/
│   ├── config.py               # config loading
│   ├── loader.py               # document loading + splitting
│   ├── vector_store.py         # ChromaDB + embeddings
│   ├── retriever.py            # hybrid retrieval + rerank
│   ├── qa_chain.py             # RAG Q&A chain
│   ├── history.py              # SQLite chat history
│   └── cli.py                  # command-line entry point
├── tests/test_smoke.py
├── .github/workflows/ci.yml
└── data/
    ├── documents/              # raw documents (optional)
    ├── vector_store/           # ChromaDB persistence
    └── history.db              # chat history
```

## ⚙️ Tunables (`.env`)

| Variable | Description | Default |
|---|---|---|
| `LLM_PROVIDER` | `deepseek` or `dashscope` | `deepseek` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | chunk size / overlap | 500 / 50 |
| `TOP_K` | final fragments returned | 4 |
| `USE_HYBRID_SEARCH` | enable BM25 + vector hybrid | `true` |
| `HYBRID_CANDIDATES` | candidates per retriever | 20 |
| `RRF_K` | RRF fusion constant | 60 |
| `USE_RERANKER` | enable reranker (more accurate, slower) | `false` |
| `RERANKER_MODEL` | reranker model | `BAAI/bge-reranker-base` |
| `EMBEDDING_MODEL` | sentence-transformers model | `BAAI/bge-small-zh-v1.5` |

## 🧪 Development

```bash
pip install ruff pytest
ruff check src app.py
pytest tests -q
```

## 📜 License

MIT
