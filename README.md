# 📚 AI 个人知识库问答助手

基于 **RAG（Retrieval-Augmented Generation）** 的本地知识库问答工具。上传你的 PDF / Word / Markdown / 网页，就能用自然语言提问，回答时会标注引用来源。

> 技术栈：Streamlit + ChromaDB + BGE Embedding + BM25 混合检索 + DeepSeek / 通义千问

## ✨ 功能

- 📄 多格式导入：PDF、Word (.docx)、Markdown、TXT、网页 URL
- 🔍 **混合检索**：BM25 关键词 + 向量召回 + RRF 融合，可选 `bge-reranker` 重排序
- 🔌 **本地中文嵌入**：`BAAI/bge-small-zh-v1.5`，无需 API
- 💬 **多会话管理**：SQLite 持久化对话历史，可切换/重命名/删除
- 🗂️ 文档级管理：单独删除、清空知识库
- 🖥️ 同时提供 **Streamlit Web UI** 与 **命令行 CLI**
- 🐳 一键 **Docker Compose** 部署
- 🤖 兼容 OpenAI 接口的国产模型（DeepSeek / 通义千问）
- ✅ GitHub Actions CI（ruff + 烟雾测试）

## 🚀 快速开始（本地）

```bash
git clone https://github.com/ablek757/ragbox.git
cd ragbox

python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS / Linux

pip install -r requirements.txt

copy .env.example .env           # 然后填入 DEEPSEEK_API_KEY
streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`。
首次运行会下载 BGE 嵌入模型（~100MB），请耐心等待；如需 HF 镜像：`set HF_ENDPOINT=https://hf-mirror.com`。

## 🐳 Docker 一键启动

```bash
cp .env.example .env             # 填入 API key
docker compose up -d --build
```

访问 `http://localhost:8501`。HuggingFace 缓存与 ChromaDB 数据通过卷持久化，重启不丢。

## 🛠️ 命令行 CLI

```bash
# 批量入库（支持目录递归）
python -m src.cli ingest ./docs ./notes/note1.md

# 抓取 URL
python -m src.cli url https://example.com/article

# 问答
python -m src.cli ask "什么是 RAG？" --show-sources

# 查看 / 清空
python -m src.cli list
python -m src.cli clear -y
```

## 📁 项目结构

```
ai-knowledge-base/
├── app.py                      # Streamlit 入口
├── Dockerfile / docker-compose.yml
├── requirements.txt / pyproject.toml
├── .env.example
├── src/
│   ├── config.py               # 配置加载
│   ├── loader.py               # 文档加载与切分
│   ├── vector_store.py         # ChromaDB + 嵌入
│   ├── retriever.py            # 混合检索 + Rerank
│   ├── qa_chain.py             # RAG 问答链
│   ├── history.py              # SQLite 对话历史
│   └── cli.py                  # 命令行入口
├── tests/test_smoke.py
├── .github/workflows/ci.yml
└── data/
    ├── documents/              # 原始文档（可选）
    ├── vector_store/           # ChromaDB 持久化
    └── history.db              # 对话历史
```

## ⚙️ 可调参数（`.env`）

| 变量 | 说明 | 默认 |
|---|---|---|
| `LLM_PROVIDER` | `deepseek` 或 `dashscope` | `deepseek` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 切分大小 / 重叠 | 500 / 50 |
| `TOP_K` | 最终返回片段数 | 4 |
| `USE_HYBRID_SEARCH` | 启用 BM25 + 向量混合检索 | `true` |
| `HYBRID_CANDIDATES` | 每路召回候选数 | 20 |
| `RRF_K` | RRF 融合常数 | 60 |
| `USE_RERANKER` | 启用 reranker（更准但更慢） | `false` |
| `RERANKER_MODEL` | reranker 模型 | `BAAI/bge-reranker-base` |
| `EMBEDDING_MODEL` | sentence-transformers 模型 | `BAAI/bge-small-zh-v1.5` |

## 🧪 开发

```bash
pip install ruff pytest
ruff check src app.py
pytest tests -q
```

## 📜 License

MIT
