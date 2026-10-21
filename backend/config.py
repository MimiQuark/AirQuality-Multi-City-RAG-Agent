"""Environment-backed application settings without mandatory extra dependencies."""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # .env support is optional at runtime
    pass


def _optional_float(name: str) -> float | None:
    """读取可选的数值型环境变量；未设置时返回 None（表示"用默认标定值"）。"""
    raw = os.getenv(name)
    return float(raw) if raw else None


def _csv(name: str, default: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in os.getenv(name, default).split(",") if item.strip())


# 逐条结果的相对分下限（保留 cos >= min_score × 最高 cos 的分块）。
# 默认 0 = 关闭：扫描显示该机制对语义模型无收益（Recall@5 0.955→0.955），
# 却会显著损害稀疏哈希模型（Recall@5 1.000→0.345），因此不作为默认闸门。
_DEFAULT_MIN_SCORE = os.getenv("RAG_MIN_SCORE", "0.0")


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///backend/data/air_quality.db")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    use_real_embeddings: bool = os.getenv("USE_REAL_EMBEDDINGS", "0") == "1"
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    rag_chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "400"))
    rag_chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "64"))
    # 分块策略：fixed（按字符硬切）/ structure（按章节边界）/ structure_context（+标题前缀）。
    # 默认 structure 的理由是"不切断条款"这一结构性正确性，而非评测分数提升——
    # bench_chunking 的配对 bootstrap 显示各档 chunk_size 上的提升均未达显著（见 docs/RAG_EVAL_REPORT.md）。
    rag_chunk_strategy: str = os.getenv("RAG_CHUNK_STRATEGY", "structure")
    rag_min_score: float = float(_DEFAULT_MIN_SCORE)
    # 拒答置信度阈值（原始余弦相似度）：低于该值判定为"库中无相关内容"。
    # 留空 = 使用 embedding 自带的标定值（见 backend/rag/embeddings.py），
    # 因为不同模型的相似度尺度不可比。需要覆盖时再设 RAG_REFUSAL_THRESHOLD。
    rag_refusal_threshold: float | None = _optional_float("RAG_REFUSAL_THRESHOLD")
    admin_api_token: str = os.getenv("ADMIN_API_TOKEN", "")
    cors_origins: tuple[str, ...] = _csv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")


settings = Settings()
