"""Hybrid retrieval over the curated environmental knowledge base."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from backend.rag.embeddings import EmbeddingProvider

DEFAULT_KNOWLEDGE = Path(__file__).parent / "documents" / "knowledge.json"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_id: str
    title: str
    text: str
    source_url: str
    category: str


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    source_id: str
    title: str
    text: str
    source_url: str
    score: float
    vector_score: float
    keyword_score: float


def load_documents(path: Path = DEFAULT_KNOWLEDGE) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return list(payload.get("documents", []))


def chunk_document(document: dict[str, Any], size: int, overlap: int) -> list[Chunk]:
    text = re.sub(r"\s+", " ", str(document.get("content", ""))).strip()
    if not text:
        return []
    step = max(1, size - overlap)
    chunks: list[Chunk] = []
    for offset in range(0, len(text), step):
        piece = text[offset : offset + size].strip()
        if not piece:
            continue
        digest = hashlib.sha1(f"{document.get('id')}:{offset}:{piece}".encode()).hexdigest()[:12]
        chunks.append(
            Chunk(
                chunk_id=digest,
                source_id=str(document.get("id", digest)),
                title=str(document.get("title", "未命名条款")),
                text=piece,
                source_url=str(document.get("source_url", "")),
                category=str(document.get("category", "其他")),
            )
        )
        if offset + size >= len(text):
            break
    return chunks


def _tokens(text: str) -> list[str]:
    normalized = re.sub(r"\s+", "", text.lower())
    return list(normalized) + [normalized[i : i + 2] for i in range(max(0, len(normalized) - 1))]


class HybridRagService:
    def __init__(
        self,
        embedding: EmbeddingProvider,
        *,
        chunk_size: int = 400,
        chunk_overlap: int = 64,
        top_k: int = 5,
        vector_weight: float = 0.4,
        min_score: float = 0.15,
    ) -> None:
        self.embedding = embedding
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.vector_weight = vector_weight
        self.min_score = min_score
        self.chunks: list[Chunk] = []
        self.vectors = np.empty((0, embedding.dimension), dtype=np.float32)

    def index(self, documents: list[dict[str, Any]]) -> dict[str, int | str]:
        self.chunks = [
            chunk
            for document in documents
            for chunk in chunk_document(document, self.chunk_size, self.chunk_overlap)
        ]
        embeddings = self.embedding.embed_documents([chunk.text for chunk in self.chunks])
        self.vectors = np.asarray(embeddings, dtype=np.float32)
        return {"documents": len(documents), "chunks": len(self.chunks), "embedding": self.embedding.name}

    def search(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        if not self.chunks:
            self.index(load_documents())
        limit = top_k or self.top_k
        query_vector = np.asarray(self.embedding.embed_query(query), dtype=np.float32)
        vector_scores = self.vectors @ query_vector
        query_tokens = set(_tokens(query))
        keyword_scores = np.asarray(
            [
                len(query_tokens & set(_tokens(chunk.text))) / math.sqrt(max(1, len(query_tokens)))
                for chunk in self.chunks
            ],
            dtype=np.float32,
        )

        def ranks(scores: np.ndarray) -> dict[int, int]:
            order = np.argsort(-scores)
            return {int(index): rank + 1 for rank, index in enumerate(order)}

        vector_ranks = ranks(vector_scores)
        keyword_ranks = ranks(keyword_scores)
        results: list[SearchResult] = []
        normalized_query = re.sub(r"\s+", "", query.lower())
        for index, chunk in enumerate(self.chunks):
            fused = (
                self.vector_weight / (60 + vector_ranks[index])
                + (1 - self.vector_weight) / (60 + keyword_ranks[index])
            )
            normalized_title = re.sub(r"\s+", "", chunk.title.lower())
            if normalized_title in normalized_query or normalized_query in normalized_title:
                title_score = 1.0
            else:
                title_tokens = set(_tokens(chunk.title))
                title_score = len(query_tokens & title_tokens) / max(1, len(title_tokens))
            rerank_score = (
                0.60 * title_score
                + 0.25 * min(1.0, float(keyword_scores[index]) * 3)
                + 0.15 * max(0.0, (float(vector_scores[index]) + 1.0) / 2.0)
                + fused
            )
            if fused < self.min_score / 100:
                continue
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    source_id=chunk.source_id,
                    title=chunk.title,
                    text=chunk.text,
                    source_url=chunk.source_url,
                    score=round(float(rerank_score), 6),
                    vector_score=round(float(vector_scores[index]), 6),
                    keyword_score=round(float(keyword_scores[index]), 6),
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        selected = results[:limit]
        strongest = selected[0] if selected else None
        domain_terms = (
            "空气质量", "AQI", "IAQI", "PM2.5", "PM10", "臭氧", "O3",
            "NO2", "SO2", "CO", "污染", "雾霾", "健康", "预警", "标准", "限值",
        )
        has_domain_term = any(term.lower() in query.lower() for term in domain_terms)
        has_evidence = bool(strongest and has_domain_term)
        return {
            "query": query,
            "refused": not has_evidence,
            "results": [result.__dict__ for result in selected],
        }




