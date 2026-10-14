"""Environment-backed application settings without mandatory extra dependencies."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _csv(name: str, default: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in os.getenv(name, default).split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///backend/data/air_quality.db")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    use_real_embeddings: bool = os.getenv("USE_REAL_EMBEDDINGS", "0") == "1"
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    rag_chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "400"))
    rag_chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "64"))
    rag_min_score: float = float(os.getenv("RAG_MIN_SCORE", "0.15"))
    admin_api_token: str = os.getenv("ADMIN_API_TOKEN", "")
    cors_origins: tuple[str, ...] = _csv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")


settings = Settings()
