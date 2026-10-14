"""Embedding providers used by the RAG pipeline.

The deterministic provider keeps unit and CI tests offline. The sentence-transformers
provider is loaded lazily only when real embeddings are explicitly enabled.
"""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    dimension: int
    name: str
    # 该 provider 下"拒答置信度阈值"的标定值（原始余弦相似度）。
    # 不同 embedding 的相似度尺度不可比，必须随模型一起标定；
    # 标定脚本：python -m backend.evaluation.eval_rag --sweep
    default_refusal_threshold: float

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HashEmbeddingProvider:
    """Deterministic feature hashing for Chinese text and tests."""

    name = "hash-v1"

    # 纯词面哈希的语义区分度不足：平衡拒答集上应拒样本均值 0.191、应答样本均值 0.227，
    # 且"必须拒答"的锚点得分（0.337）高于"必须应答"的锚点（0.135）——不存在能同时
    # 满足两者的阈值。这里取一个保证领域内问题（如"PM2.5 标准"=0.446）不被误拒、
    # 同时能拦住明显超纲问题（如"个税专项扣除"=0.159）的保守值；其拒答指标仅供参考，
    # 不能作为线上依据。离线 provider 只用于 CI 回归。
    default_refusal_threshold = 0.18

    def __init__(self, dimension: int = 128) -> None:
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dimension, dtype=np.float32)
        normalized = re.sub(r"\s+", "", text.lower())
        tokens = list(normalized)
        tokens.extend(normalized[i : i + 2] for i in range(max(0, len(normalized) - 1)))
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class SentenceTransformerEmbedding:
    name = "sentence-transformers"

    # BGE 系列默认值：知识库扩容（11 篇/4199 字）后重新标定。
    # 平衡拒答集上的可行区间为 (0.489, 0.567)，取区间中点 0.53：
    # 应拒命中 23/24、误拒 4/24、总体准确率 0.896。
    # 取中点而非样本内最优，是为了对语料/分块变化更稳健。
    # 本项目"宁可少答也不编造条款"，因此偏向提高应拒命中。
    # 该值随语料与分块策略漂移，改动知识库后必须重新标定：
    #   python -m backend.evaluation.eval_rag --sweep
    default_refusal_threshold = 0.53

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        # sentence-transformers 5.x 起该方法改名为 get_embedding_dimension
        getter = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dimension = int(getter())
        self.name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()


def create_embedding_provider(model_name: str, use_real: bool) -> EmbeddingProvider:
    if use_real:
        return SentenceTransformerEmbedding(model_name)
    return HashEmbeddingProvider()
