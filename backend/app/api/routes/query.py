"""
Query endpoint — the core of the four-way comparison.

This file is intentionally thin. All logic lives in the service layer:
  EmbeddingService / VectorStore  →  vector_service.py
  RAGService                      →  rag_service.py
  LLMService                      →  llm_service.py  (also owns the topic guard)
  MLService                       →  ml_service.py

Pipeline per request:
  0. Guard     — topic check: reject queries outside airline/travel support scope
  1. Retrieve  — embed query, fetch top-k from Qdrant
  2. RAG        — LLM answer with retrieved context
  3. Plain      — LLM answer without context (baseline)
  4. ML         — trained binary classifier (URGENT / NORMAL)
  5. LLM zero-shot — same question to Claude, no training
  6. Log        — append audit record to queries.jsonl
"""

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.logger import app_logger
from app.services.llm_service import llm_service
from app.services.ml_service import ml_service
from app.services.rag_service import rag_service

router = APIRouter()
#Separate schemas services - apis TODO

# ── Schemas ────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    rag_answer: str
    non_rag_answer: str
    retrieved_sources: List[Dict[str, Any]]
    low_similarity_warning: bool
    ml_prediction: Dict[str, Any]
    llm_prediction: Dict[str, Any]
    metrics: Dict[str, Any]


# ── Friendly messages for each block reason ───────────────────────────────

_BLOCK_MESSAGES = {
    "code_request":          "This assistant doesn't write or debug code.",
    "arithmetic_expression": "This assistant doesn't solve maths problems.",
    "creative_writing":      "This assistant doesn't write creative content.",
    "general_trivia":        "This assistant doesn't answer general knowledge questions.",
    "jailbreak_attempt":     "That type of instruction is not permitted here.",
    "off_topic":             "Your question doesn't appear to be about airline customer support.",
}

_SCOPE_HINT = (
    "Priority Pulse only handles airline customer support questions — "
    "flights, delays, cancellations, baggage, refunds, check-in, boarding, "
    "loyalty programmes, and related travel issues."
)
# separate this block

# ── Endpoint ───────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def process_query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="Query cannot be empty.")

    # ── 0. topic guard ─────────────────────────────────────────────────────
    is_allowed, reason = llm_service.check_topic(req.query)
    if not is_allowed:
        app_logger.warning(f"Blocked query [{reason}]: {req.query[:120]}")
        raise HTTPException(
            status_code=422,
            detail={
                "code":    "off_topic",
                "reason":  reason,
                "message": _BLOCK_MESSAGES.get(reason, "Query is out of scope."),
                "hint":    _SCOPE_HINT,
            },
        )

    app_logger.info(f"Query accepted [{reason}]: {req.query[:80]}")

    # ── 1. retrieve ────────────────────────────────────────────────────────
    _embedding, sources = rag_service.retrieve(req.query)
    low_sim = rag_service.is_low_similarity(sources)

    # ── 2. RAG answer ──────────────────────────────────────────────────────
    rag_text, rag_ms, rag_cost = llm_service.answer_with_context(req.query, sources)

    # ── 3. non-RAG answer ──────────────────────────────────────────────────
    plain_text, plain_ms, plain_cost = llm_service.answer_plain(req.query)

    # ── 4. ML prediction ───────────────────────────────────────────────────
    ml_result = ml_service.predict(req.query)

    # ── 5. LLM zero-shot priority ──────────────────────────────────────────
    llm_label, llm_ms, llm_cost = llm_service.classify_priority(req.query)

    # ── 6. log ─────────────────────────────────────────────────────────────
    app_logger.log_query({
        "query": req.query,
        "retrieved_count": len(sources),
        "low_similarity": low_sim,
        "rag_latency_ms": rag_ms,
        "rag_cost_usd": rag_cost,
        "non_rag_latency_ms": plain_ms,
        "non_rag_cost_usd": plain_cost,
        "ml_label": ml_result.get("label"),
        "ml_confidence": ml_result.get("confidence"),
        "llm_label": llm_label,
        "llm_latency_ms": llm_ms,
        "llm_cost_usd": llm_cost,
    })

    return QueryResponse(
        rag_answer=rag_text,
        non_rag_answer=plain_text,
        retrieved_sources=sources,
        low_similarity_warning=low_sim,
        ml_prediction=ml_result,
        llm_prediction={
            "label": llm_label,
            "latency_ms": llm_ms,
            "cost_usd": llm_cost,
        },
        metrics={
            "rag_ms": rag_ms,
            "rag_cost_usd": rag_cost,
            "non_rag_ms": plain_ms,
            "non_rag_cost_usd": plain_cost,
        },
    )
