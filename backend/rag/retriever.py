"""Backward-compatible keyword-style API backed by the hybrid RAG service."""
from __future__ import annotations

from typing import Any

from backend.config import settings
from backend.rag import HybridRagService, create_embedding_provider, load_documents

# 注意：这是 HybridRagService 的**相对**分下限（保留 cos >= MIN_SCORE × 最高 cos 的分块），
# 不是 0-1 的绝对分数，也不是拒答阈值。拒答阈值请用 get_refusal_threshold()。
MIN_SCORE = settings.rag_min_score
_service: HybridRagService | None = None


def _get_service() -> HybridRagService:
    global _service
    if _service is None:
        _service = HybridRagService(
            create_embedding_provider(settings.embedding_model, settings.use_real_embeddings),
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            chunk_strategy=settings.rag_chunk_strategy,
            min_score=MIN_SCORE,
            refusal_threshold=settings.rag_refusal_threshold,
        )
        _service.index(load_documents())
    return _service


def get_refusal_threshold() -> float:
    """当前服务使用的拒答阈值（原始余弦相似度），随 embedding 标定。"""
    return _get_service().refusal_threshold


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
        # best_score 是**排序用**的混合分（0-1），confidence 是拒答判据（原始余弦），
        # 两者量纲不同，展示/比较时不要混用（原实现的拒答说明就混用了这两个数）。
        "best_score": results[0]["score"] if results else 0.0,
        "confidence": response["confidence"],
        "refusal_threshold": _get_service().refusal_threshold,
    }


def retrieve(query: str, top_k: int = 3, min_score: float | None = None) -> list[dict[str, Any]]:
    """按**混合分**过滤检索结果（min_score 属 0-1 绝对分，默认 None = 不过滤）。

    与 HybridRagService 构造参数 min_score（cos 相对下限）同名不同义，勿混用。
    """
    detail = retrieve_detailed(query, top_k)
    if min_score is None:
        return detail["results"]
    return [item for item in detail["results"] if item["score"] >= min_score]


def query_policy(query: str) -> dict[str, Any]:
    detail = retrieve_detailed(query)
    if detail["low_confidence"]:
        return {"refused": True, "reason": "未检索到高置信政策条款", "results": []}
    return {"refused": False, "results": detail["results"]}
