"""Shared RAG service singleton used by the API and the agent tools."""
from __future__ import annotations

from backend.config import settings

from .embeddings import create_embedding_provider
from .service import HybridRagService, load_documents

_service: HybridRagService | None = None


def get_service() -> HybridRagService:
    global _service
    if _service is None:
        _service = HybridRagService(
            create_embedding_provider(
                settings.embedding_model, settings.use_real_embeddings
            ),
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            top_k=settings.rag_top_k,
            chunk_strategy=settings.rag_chunk_strategy,
            min_score=settings.rag_min_score,
            refusal_threshold=settings.rag_refusal_threshold,
        )
        _service.index(load_documents())
    return _service


def reindex() -> dict:
    return get_service().index(load_documents())
