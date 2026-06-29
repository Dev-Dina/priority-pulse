"""Smoke test for the /api/v1/query pipeline with all providers mocked.

The LLM (Gemini/Groq), vector retrieval, and ML service are patched so the test
never makes a network call or loads a model — it verifies the route wiring,
response shape, and guard behaviour only.
"""

from fastapi.testclient import TestClient

from app.core.logger import app_logger
from app.main import app
from app.services import llm_service as llm_mod
from app.services import ml_service as ml_mod
from app.services import rag_service as rag_mod


def _patch_services(monkeypatch):
    monkeypatch.setattr(llm_mod.llm_service, "check_topic", lambda q: (True, "on_topic"))
    monkeypatch.setattr(
        llm_mod.llm_service, "answer_with_context", lambda q, s: ("RAG answer", 12.3, 0.0001)
    )
    monkeypatch.setattr(llm_mod.llm_service, "answer_plain", lambda q: ("Plain answer", 20.0, 0.0002))
    monkeypatch.setattr(llm_mod.llm_service, "classify_priority", lambda t: ("URGENT", 8.0, 0.00001))
    monkeypatch.setattr(
        rag_mod.rag_service,
        "retrieve",
        lambda q: (None, [{"airline": "Delta", "text": "bag lost", "priority": "URGENT", "similarity": 0.71}]),
    )
    monkeypatch.setattr(rag_mod.rag_service, "is_low_similarity", lambda s: False)
    monkeypatch.setattr(
        ml_mod.ml_service,
        "predict",
        lambda t: {"label": "URGENT", "confidence": 0.91, "latency_ms": 1.0, "cost_usd": 0.0},
    )
    monkeypatch.setattr(app_logger, "log_query", lambda record: None)


def test_query_happy_path(monkeypatch):
    _patch_services(monkeypatch)
    client = TestClient(app)
    resp = client.post("/api/v1/query", json={"query": "my bag is lost"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["rag_answer"] == "RAG answer"
    assert body["non_rag_answer"] == "Plain answer"
    assert body["ml_prediction"]["label"] == "URGENT"
    assert body["llm_prediction"]["label"] == "URGENT"
    assert body["retrieved_sources"][0]["airline"] == "Delta"
    assert body["low_similarity_warning"] is False


def test_query_rejects_empty(monkeypatch):
    _patch_services(monkeypatch)
    client = TestClient(app)
    resp = client.post("/api/v1/query", json={"query": "   "})
    assert resp.status_code == 422


def test_query_blocks_off_topic(monkeypatch):
    _patch_services(monkeypatch)
    monkeypatch.setattr(llm_mod.llm_service, "check_topic", lambda q: (False, "off_topic"))
    client = TestClient(app)
    resp = client.post("/api/v1/query", json={"query": "write me some python code"})
    assert resp.status_code == 422
