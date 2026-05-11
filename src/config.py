import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek").lower()

PROVIDER_CONFIG = {
    "deepseek": {
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com/v1",
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    },
    "dashscope": {
        "api_key": os.getenv("DASHSCOPE_API_KEY", ""),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": os.getenv("DASHSCOPE_MODEL", "qwen-plus"),
    },
}

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", str(ROOT_DIR / "data" / "vector_store"))
COLLECTION_NAME = "knowledge_base"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "4"))

USE_HYBRID_SEARCH = os.getenv("USE_HYBRID_SEARCH", "true").lower() == "true"
HYBRID_CANDIDATES = int(os.getenv("HYBRID_CANDIDATES", "20"))
RRF_K = int(os.getenv("RRF_K", "60"))

USE_RERANKER = os.getenv("USE_RERANKER", "false").lower() == "true"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

HISTORY_DB_PATH = os.getenv("HISTORY_DB_PATH", str(ROOT_DIR / "data" / "history.db"))


def get_llm_config() -> dict:
    cfg = PROVIDER_CONFIG.get(LLM_PROVIDER)
    if not cfg:
        raise ValueError(f"未知的 LLM_PROVIDER: {LLM_PROVIDER}")
    if not cfg["api_key"]:
        raise ValueError(f"{LLM_PROVIDER} 的 API key 未设置，请在 .env 中配置")
    return cfg
