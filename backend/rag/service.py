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


# 小节边界：中文序号标题（一、二、）、【】标注、第 X 条/章、markdown 标题
_SECTION_RE = re.compile(
    r"(?m)^(?=[一二三四五六七八九十]+、|【[^】]{1,20}】|第[一二三四五六七八九十百]+[条章节]|#{1,6}\s)"
)
_SENTENCE_RE = re.compile(r"(?<=[。；！？])")


def split_sections(text: str) -> list[str]:
    """按文档自身的章节标题切分；没有标题时退化为整篇。"""
    sections = [part.strip() for part in _SECTION_RE.split(text) if part.strip()]
    return sections or ([text.strip()] if text.strip() else [])


def _split_long_section(section: str, size: int, overlap: int) -> list[str]:
    """单个小节超过 size 时，按句子边界二次切分（保持语义完整）。"""
    if len(section) <= size:
        return [section]
    sentences = [piece for piece in _SENTENCE_RE.split(section) if piece.strip()]
    pieces: list[str] = []
    buffer = ""
    for sentence in sentences:
        if buffer and len(buffer) + len(sentence) > size:
            pieces.append(buffer)
            buffer = buffer[-overlap:] if overlap else ""
        buffer += sentence
    if buffer.strip():
        pieces.append(buffer)
    # 兜底：极端情况下（单句超长）仍按长度硬切
    final: list[str] = []
    for piece in pieces:
        if len(piece) <= size:
            final.append(piece)
        else:
            step = max(1, size - overlap)
            final.extend(piece[i:i + size] for i in range(0, len(piece), step) if piece[i:i + size].strip())
    return final


def chunk_document_structured(
    document: dict[str, Any], size: int, overlap: int, *, with_title_context: bool = False
) -> list[Chunk]:
    """按文档结构分块。

    与固定长度分块的区别：
    1. 切分点落在章节标题处，不会把一个条款从中间切断；
    2. 相邻小节会被合并到不超过 size，减少碎片；
    3. 小节仍然超长时，退化到句子边界切分，而不是字符硬切。

    with_title_context=True 时在每个块前加上文档标题，作为检索时的上下文提示
    （生产 RAG 常见做法；单独作为一组对照，因为它同时改变了块的文本构成）。
    """
    raw = str(document.get("content", ""))
    title = str(document.get("title", "未命名条款"))
    source_id = str(document.get("id", ""))
    source_url = str(document.get("source_url", ""))
    category = str(document.get("category", "其他"))

    units: list[str] = []
    for section in split_sections(raw):
        units.extend(_split_long_section(section, size, overlap))

    merged: list[str] = []
    buffer = ""
    for unit in units:
        if buffer and len(buffer) + len(unit) > size:
            merged.append(buffer)
            buffer = unit
        else:
            buffer = f"{buffer}\n{unit}" if buffer else unit
    if buffer.strip():
        merged.append(buffer)

    chunks: list[Chunk] = []
    for index, piece in enumerate(merged):
        body = piece.strip()
        if not body:
            continue
        digest = hashlib.sha1(f"{source_id}:sec:{index}:{body}".encode()).hexdigest()[:12]
        chunks.append(
            Chunk(
                chunk_id=digest,
                source_id=source_id or digest,
                title=title,
                text=f"{title}\n{body}" if with_title_context else body,
                source_url=source_url,
                category=category,
            )
        )
    return chunks


CHUNK_STRATEGIES = {
    "fixed": lambda doc, size, overlap: chunk_document(doc, size, overlap),
    "structure": lambda doc, size, overlap: chunk_document_structured(doc, size, overlap),
    "structure_context": lambda doc, size, overlap: chunk_document_structured(
        doc, size, overlap, with_title_context=True
    ),
}


def _tokens(text: str) -> list[str]:
    normalized = re.sub(r"\s+", "", text.lower())
    return list(normalized) + [normalized[i : i + 2] for i in range(max(0, len(normalized) - 1))]


@dataclass(frozen=True)
class HybridWeights:
    """最终打分权重。

    在 v1（题面照抄标题的简单集）与 v2（口语化困难集）两套评测集上做网格搜索标定。
    原实现为 标题0.60 / 关键词0.25 / 向量0.15，该组权重是在 v1 上调出来的——
    v1 中 100% 的题面原样包含文档标题，title_score 恒为 1.0，导致 0.60 的标题权重
    直接锁定答案，评测必然满分，但把语义信号压到 0.15，在真实口语化提问上
    Recall@1 反而损失约 30 个百分点。
    """
    vector: float = 0.70
    keyword: float = 0.10
    title: float = 0.10
    rrf: float = 0.02


class HybridRagService:
    def __init__(
        self,
        embedding: EmbeddingProvider,
        *,
        chunk_size: int = 400,
        chunk_overlap: int = 64,
        top_k: int = 5,
        chunk_strategy: str = "fixed",
        vector_weight: float = 0.4,
        min_score: float = 0.0,
        refusal_threshold: float | None = None,
        weights: HybridWeights | None = None,
    ) -> None:
        self.embedding = embedding
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        if chunk_strategy not in CHUNK_STRATEGIES:
            raise ValueError(f"未知分块策略: {chunk_strategy}，可选 {sorted(CHUNK_STRATEGIES)}")
        self.chunk_strategy = chunk_strategy
        self.vector_weight = vector_weight    # 仅用于 RRF 两路加权
        self.min_score = min_score            # 相对下限：保留 cos >= min_score * 最高分 的分块
        # 留空则采用 embedding 自带标定值（不同模型的相似度尺度不可比）
        self.refusal_threshold = (refusal_threshold if refusal_threshold is not None
                                  else getattr(embedding, "default_refusal_threshold", 0.47))
        self.weights = weights or HybridWeights()
        self.chunks: list[Chunk] = []
        self.vectors = np.empty((0, embedding.dimension), dtype=np.float32)

    def index(self, documents: list[dict[str, Any]]) -> dict[str, int | str]:
        chunker = CHUNK_STRATEGIES[self.chunk_strategy]
        self.chunks = [
            chunk
            for document in documents
            for chunk in chunker(document, self.chunk_size, self.chunk_overlap)
        ]
        embeddings = self.embedding.embed_documents([chunk.text for chunk in self.chunks])
        self.vectors = np.asarray(embeddings, dtype=np.float32)
        return {"documents": len(documents), "chunks": len(self.chunks),
                "embedding": self.embedding.name, "chunk_strategy": self.chunk_strategy}

    def search(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        """混合检索 + 基于置信度的拒答。

        打分：score = w_v·(cos+1)/2 + w_k·min(1,kw·3) + w_t·title + w_r·rrf_norm
        score 只用于**排序**；RRF 仅把两路排名融合到同一尺度（并归一化到 0-1）。
        拒答用的 confidence 是**原始余弦**的全库最大值（见下方说明），
        不再依赖硬编码领域词表，判据为 confidence < refusal_threshold。
        """
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
        fused = np.asarray(
            [
                self.vector_weight / (60 + vector_ranks[index])
                + (1.0 - self.vector_weight) / (60 + keyword_ranks[index])
                for index in range(len(self.chunks))
            ],
            dtype=np.float32,
        )
        # 归一化后才与其它分项可比；原实现直接把 RRF 值（量级 ~1/60）与 0-1 分项相加，
        # 又用 min_score/100 做过滤，导致该过滤条件永远不成立（死代码）。
        if fused.size and float(fused.max()) > 0:
            fused = fused / float(fused.max())

        normalized_query = re.sub(r"\s+", "", query.lower())
        w = self.weights
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        for index, chunk in enumerate(self.chunks):
            normalized_title = re.sub(r"\s+", "", chunk.title.lower())
            if normalized_title in normalized_query or normalized_query in normalized_title:
                title_score = 1.0
            else:
                title_tokens = set(_tokens(chunk.title))
                title_score = len(query_tokens & title_tokens) / max(1, len(title_tokens))
            scores[index] = (
                w.vector * (float(vector_scores[index]) + 1.0) / 2.0
                + w.keyword * min(1.0, float(keyword_scores[index]) * 3.0)
                + w.title * title_score
                + w.rrf * float(fused[index])
            )

        # 置信度用"原始余弦相似度"而不是最终混合分：
        # 混合分里 (cos+1)/2 把 cos=0 映射到 0.5，无关文本的底噪被抬到 0.5-0.6，
        # 阈值失去可解释性（也难标定）。原始余弦的语义明确：越低越不相关。
        confidence = float(vector_scores.max()) if vector_scores.size else 0.0

        # 相对下限（默认 0 = 关闭）：保留 cos >= min_score × 最高 cos 的分块，
        # 削掉长尾弱相关项。用比值而非绝对值，换 embedding 不必重标定。
        keep_floor = confidence * self.min_score if self.min_score > 0 else -1.0

        results: list[SearchResult] = []
        for index, chunk in enumerate(self.chunks):
            if float(vector_scores[index]) < keep_floor:
                continue
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    source_id=chunk.source_id,
                    title=chunk.title,
                    text=chunk.text,
                    source_url=chunk.source_url,
                    score=round(float(scores[index]), 6),
                    vector_score=round(float(vector_scores[index]), 6),
                    keyword_score=round(float(keyword_scores[index]), 6),
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return {
            "query": query,
            "refused": confidence < self.refusal_threshold,
            "confidence": round(confidence, 6),
            "results": [result.__dict__ for result in results[:limit]],
        }
