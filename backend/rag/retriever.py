"""Backward-compatible keyword-style API backed by the hybrid RAG service."""
from __future__ import annotations

from typing import Any

from backend.config import settings
from backend.rag import HybridRagService, create_embedding_provider, load_documents

MIN_SCORE = settings.rag_min_score
_service: HybridRagService | None = None


def _get_service() -> HybridRagService:
    global _service
    if _service is None:
        _service = HybridRagService(
            create_embedding_provider(settings.embedding_model, settings.use_real_embeddings),
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            min_score=MIN_SCORE,
        )
        _service.index(load_documents())
    return _service


def retrieve_detailed(query: str, top_k: int = 3) -> dict[str, Any]:
    response = _get_service().search(query, top_k=top_k)
    results = [
        {
            "id": item["source_id"],
            "chunk_id": item["chunk_id"],
            "title": item["title"],
            "content": item["text"],
            "category": item.get("category", "其他"),
            "source_url": item.get("source_url", ""),
            "score": item["score"],
        }
        for item in response["results"]
    ]
    return {
        "query": query,
        "results": results,
        "low_confidence": response["refused"],
        "best_score": results[0]["score"] if results else 0.0,
    }


def retrieve(query: str, top_k: int = 3, min_score: float = MIN_SCORE) -> list[dict[str, Any]]:
    detail = retrieve_detailed(query, top_k)
    return [item for item in detail["results"] if item["score"] >= min_score]


def query_policy(query: str) -> dict[str, Any]:
    detail = retrieve_detailed(query)
    if detail["low_confidence"]:
        return {"refused": True, "reason": "未检索到高置信政策条款", "results": []}
    return {"refused": False, "results": detail["results"]}
