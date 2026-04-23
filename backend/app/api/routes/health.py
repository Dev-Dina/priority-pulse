"""
Health and metrics endpoints.

/health  — liveness + readiness check (used by Docker Compose health checks)
/metrics — pre-computed ML test set metrics (read by frontend comparison panel)
"""

import json

from fastapi import APIRouter

from app.config import settings
from app.services.vector_service import vector_store
from app.services.ml_service import ml_service

router = APIRouter()


@router.get("/health")
def health():
    """
    Liveness + readiness check.

    Checks:
      qdrant_status  — "connected" | "disconnected"
      vector_count   — number of ticket embeddings stored
      ml_model       — "ready" | "not_trained"
    """
    try:
        count = vector_store.count()
        qdrant_status = "connected"
    except Exception:
        count = 0
        qdrant_status = "disconnected"

    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "qdrant": qdrant_status,
        "vector_count": count,
        "ml_model": "ready" if ml_service.is_ready else "not_trained",
    }


@router.get("/metrics")
def get_metrics():
    """
    Pre-computed val/test-set metrics from the last training run.

    The frontend comparison panel reads this on mount so it can display
    ML accuracy, F1, and ROC-AUC alongside per-call latency and cost.
    Returns {"status": "not_trained"} if train_models.py hasn't been run.
    """
    metrics_path = settings.MODEL_DIR / "metrics.json"
    if not metrics_path.exists():
        return {
            "status": "not_trained",
            "message": "Run: python scripts/train_models.py",
        }
    with open(metrics_path) as f:
        return json.load(f)


@router.get("/data-stats")
def get_data_stats():
    """
    Pipeline cleaning stats saved by run_data_pipeline.py.

    The frontend Data & Labels page reads this to display the cleaning funnel,
    agent-response coverage, labeling distribution, and non-selected TWCS brands.
    Returns {"status": "not_run"} if the pipeline hasn't been run yet.
    """
    stats_path = settings.DATA_PROCESSED_DIR / "pipeline_stats.json"
    if not stats_path.exists():
        return {
            "status": "not_run",
            "message": "Run: python scripts/run_data_pipeline.py",
        }
    with open(stats_path) as f:
        return json.load(f)
