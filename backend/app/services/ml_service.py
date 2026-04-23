"""
ML Service — trained binary classifier for priority prediction.

Lazy-loads the model on first call so FastAPI starts even if training
hasn't been run yet. Returns a structured dict matching the API response shape.

Binary prediction: URGENT (1) or NORMAL (0).
Cost is always $0.00 — in-process inference has zero marginal cost.
This is the whole point of the ML vs LLM comparison.
"""

import time
from typing import List, Optional

import joblib
import pandas as pd

from app.config import settings
from app.core.logger import app_logger
from app.ml.feature_engineering import engineer_features


class MLService:

    def __init__(self):
        self._model = None
        self._feature_names: Optional[List[str]] = None
        self._classes: Optional[List[str]] = None

    def _load(self) -> bool:
        """Attempt to load the saved model. Returns True on success."""
        if self._model is not None:
            return True

        model_path = settings.MODEL_DIR / "priority_classifier.pkl"
        features_path = settings.MODEL_DIR / "feature_names.pkl"

        if not model_path.exists():
            app_logger.warning(
                f"ML model not found at {model_path}. "
                "Run: python scripts/train_models.py"
            )
            return False

        self._model = joblib.load(model_path)
        self._feature_names = joblib.load(features_path)
        self._classes = list(self._model.classes_)
        app_logger.info(f"ML model loaded. Classes: {self._classes}")
        return True

    @property
    def is_ready(self) -> bool:
        return self._load()

    def predict(self, text: str) -> dict:
        """
        Predict priority for a single text string.

        Returns label, confidence, latency_ms, cost_usd.
        """
        if not self._load():
            return {
                "label": "UNAVAILABLE",
                "confidence": None,
                "latency_ms": None,
                "cost_usd": 0.0,
                "note": "Model not trained. Run: python scripts/train_models.py",
            }

        features = engineer_features(text)
        X = pd.DataFrame([features])[self._feature_names]

        t0 = time.perf_counter()
        label = self._model.predict(X)[0]
        proba = self._model.predict_proba(X)[0]
        latency_ms = (time.perf_counter() - t0) * 1000

        label_str = "URGENT" if str(label) == "1" else "NORMAL"
        return {
            "label": label_str,
            "confidence": round(float(max(proba)), 4),
            "latency_ms": round(latency_ms, 4),
            "cost_usd": 0.0,
        }


ml_service = MLService()
