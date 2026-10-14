import json
from fastapi.testclient import TestClient
from backend.main import app
from backend.rag import HybridRagService, create_embedding_provider
from backend.rag.service import load_documents


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
    assert all(item.get("version") == "2026-curated-1" for item in documents)


