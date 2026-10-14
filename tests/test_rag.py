import json

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.rag import HybridRagService, create_embedding_provider
from backend.rag.service import (
    CHUNK_STRATEGIES,
    HybridRagService,
    chunk_document,
    chunk_document_structured,
    load_documents,
    split_sections,
)


def test_hybrid_search_returns_citations_and_refuses_irrelevant() -> None:
    service = HybridRagService(create_embedding_provider("", False))
    service.index(load_documents())
    result = service.search("HJ 633-2012 环境空气质量指数(AQI)技术规定")
    assert result["refused"] is False
    assert result["results"][0]["source_id"] == "hj633-1"
    assert result["results"][0]["source_url"]
    assert service.search("个人所得税专项扣除")["refused"] is True


def test_rag_search_api() -> None:
    client = TestClient(app)
    response = client.post("/api/rag/search", json={"query": "PM2.5 标准", "top_k": 3})
    assert response.status_code == 200
    payload = response.json()
    assert payload["refused"] is False
    assert len(payload["results"]) == 3


def test_reindex_disabled_without_admin_token() -> None:
    client = TestClient(app)
    assert client.post("/api/rag/reindex").status_code == 404


def test_knowledge_sources_have_metadata() -> None:
    documents = load_documents()
    assert len(documents) == 11
    assert all(item.get("source_url") for item in documents)
    assert all(item.get("version") == "2026-curated-2" for item in documents)


# --------------------------------------------------------------------------
# 分块策略
# --------------------------------------------------------------------------

def test_split_sections_follows_heading_boundaries() -> None:
    document = next(d for d in load_documents() if d["id"] == "gb3095-1")
    sections = split_sections(document["content"])
    assert len(sections) >= 4, "结构化文档应能按章节标题切开"
    assert sections[0].startswith("一、")
    # 无标题文本退化为整篇，不应丢内容
    assert split_sections("一段没有任何标题的文字。") == ["一段没有任何标题的文字。"]


def test_structured_chunking_starts_chunks_at_section_boundaries() -> None:
    document = next(d for d in load_documents() if d["id"] == "gb3095-1")
    structured = chunk_document_structured(document, 400, 64)
    fixed = chunk_document(document, 400, 64)
    assert len(structured) >= 2
    # 结构分块：第二个块从某个小节标题开始，不会从条款中间切断
    assert structured[1].text.startswith(("二、", "三、", "四、", "五、", "六、"))


def test_structured_chunking_long_section_falls_back_to_sentences() -> None:
    """单个小节超过 size 时应按句子边界二次切分，而不是字符硬切。"""
    long_section = "一、测试小节\n" + "这是一个用于测试的完整句子。" * 40
    document = {"id": "t1", "title": "测试文档", "content": long_section,
                "source_url": "https://example.com", "category": "测试"}
    chunks = chunk_document_structured(document, 200, 32)
    assert len(chunks) >= 2
    assert all(chunk.text.strip() for chunk in chunks)
    assert all(chunk.source_id == "t1" for chunk in chunks)


def test_all_chunk_strategies_index_without_error() -> None:
    documents = load_documents()
    for name in CHUNK_STRATEGIES:
        service = HybridRagService(create_embedding_provider("", False), chunk_strategy=name)
        info = service.index(documents)
        assert info["chunks"] > 0, name
        assert info["chunk_strategy"] == name
        response = service.search("PM2.5 的国家标准限值", top_k=3)
        assert response["results"], name
        assert 0.0 <= response["confidence"] <= 1.0

def test_unknown_chunk_strategy_rejected() -> None:
    with pytest.raises(ValueError):
        HybridRagService(create_embedding_provider("", False), chunk_strategy="not-a-strategy")


# --------------------------------------------------------------------------
# retriever 兼容层
# --------------------------------------------------------------------------

def test_retriever_helpers_expose_scores_and_refusal() -> None:
    from backend.rag import retriever

    detail = retriever.retrieve_detailed("PM2.5 的国家标准限值是多少")
    assert detail["results"], "领域内问题应能召回"
    assert detail["best_score"] > 0
    assert detail["results"][0]["source_url"]
    assert isinstance(retriever.retrieve("空气污染应对措施", min_score=0.0), list)
    assert retriever.query_policy("PM2.5 的国家标准限值是多少")["refused"] is False
    assert retriever.query_policy("个人所得税专项扣除")["refused"] is True
