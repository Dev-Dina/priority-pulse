import json
import joblib
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import settings
from app.core.logger import ml_logger


FEATURE_COLS = [
    "text_length",
    "word_count",
    "exclamation_count",
    "question_count",
    "caps_ratio",
    "has_refund",
    "has_cancel",
    "has_delay",
    "has_help",
    "has_broken",
    "has_stranded",
    "has_medical",
    "profanity_count",
    "has_time_mention",
]

TARGET_COL = "is_urgent"   # binary: 1 = URGENT, 0 = NORMAL


def _metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred, pos_label=1)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
    }


def run_training() -> dict:
    """
    Full training run. Returns per-model results so callers can inspect metrics.
    """
    ml_logger.info("=" * 60)
    ml_logger.info("ML TRAINING PIPELINE  (60 / 20 / 20 split)")
    ml_logger.info("=" * 60)

    # ── load data ──────────────────────────────────────────────────────────
    path = settings.ML_FEATURES_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"ml_features.csv not found at {path}. "
            "Run: python scripts/run_data_pipeline.py first."
        )

    df = pd.read_csv(path)
    ml_logger.info(f"Loaded {len(df):,} rows from {path.name}")

    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in ml_features.csv: {missing}")

    X = df[FEATURE_COLS].fillna(0)
    y = df[TARGET_COL]

    class_counts = y.value_counts().to_dict()
    ml_logger.info(
        f"Class distribution — URGENT (1): {class_counts.get(1, 0):,} "
        f"({100 * class_counts.get(1, 0) / len(y):.1f}%)  "
        f"NORMAL (0): {class_counts.get(0, 0):,}"
    )

    # ── 60 / 20 / 20 split ────────────────────────────────────────────────
    # Step 1: carve out 60% train, leave 40% for val+test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )
    # Step 2: split the remaining 40% equally → 20% val, 20% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    ml_logger.info(
        f"Split — Train: {len(X_train):,}  "
        f"Val: {len(X_val):,}  "
        f"Test: {len(X_test):,}"
    )

    # ── models to compare ──────────────────────────────────────────────────
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}

    for name, model in candidates.items():
        ml_logger.info(f"\nTraining {name}…")
        model.fit(X_train, y_train)

        # ── validation (used for model selection) ─────────────────────────
        val_pred  = model.predict(X_val)
        val_proba = model.predict_proba(X_val)[:, 1]
        val_m = _metrics(y_val, val_pred, val_proba)

        ml_logger.info(
            f"  [val]  accuracy={val_m['accuracy']:.4f}  "
            f"f1={val_m['f1']:.4f}  roc_auc={val_m['roc_auc']:.4f}"
        )
        ml_logger.info(
            "\n[val] " + classification_report(y_val, val_pred, target_names=["NORMAL", "URGENT"])
        )

        # ── test (held-out; evaluated once after selection) ───────────────
        test_pred  = model.predict(X_test)
        test_proba = model.predict_proba(X_test)[:, 1]
        test_m = _metrics(y_test, test_pred, test_proba)

        ml_logger.info(
            f"  [test] accuracy={test_m['accuracy']:.4f}  "
            f"f1={test_m['f1']:.4f}  roc_auc={test_m['roc_auc']:.4f}"
        )
        ml_logger.info(
            "\n[test] " + classification_report(y_test, test_pred, target_names=["NORMAL", "URGENT"])
        )

        results[name] = {
            "model": model,
            "val_accuracy":  val_m["accuracy"],
            "val_f1":        val_m["f1"],
            "val_roc_auc":   val_m["roc_auc"],
            "test_accuracy": test_m["accuracy"],
            "test_f1":       test_m["f1"],
            "test_roc_auc":  test_m["roc_auc"],
        }

    # ── select best model by validation F1 ────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["val_f1"])
    best = results[best_name]
    ml_logger.info(
        f"\nBest model (by val F1): {best_name}  "
        f"(val_F1={best['val_f1']:.4f}, test_F1={best['test_f1']:.4f}, "
        f"test_AUC={best['test_roc_auc']:.4f})"
    )

    # ── save model artifacts ───────────────────────────────────────────────
    settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path   = settings.MODEL_DIR / "priority_classifier.pkl"
    features_path = settings.MODEL_DIR / "feature_names.pkl"

    joblib.dump(best["model"], model_path)
    joblib.dump(FEATURE_COLS, features_path)

    # ── save metrics.json (read by /api/v1/metrics and frontend) ──────────
    metrics_data = {
        "trained_at":       datetime.now(timezone.utc).isoformat(),
        "best_model":       best_name,
        "dataset_size":     len(df),
        "train_size":       len(X_train),
        "val_size":         len(X_val),
        "test_size":        len(X_test),
        "class_distribution": {str(k): int(v) for k, v in class_counts.items()},
        "features":         FEATURE_COLS,
        "models": {
            name: {
                "val_accuracy":  r["val_accuracy"],
                "val_f1":        r["val_f1"],
                "val_roc_auc":   r["val_roc_auc"],
                "test_accuracy": r["test_accuracy"],
                "test_f1":       r["test_f1"],
                "test_roc_auc":  r["test_roc_auc"],
            }
            for name, r in results.items()
        },
    }
    metrics_path = settings.MODEL_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)

    ml_logger.info(f"Saved model    → {model_path}")
    ml_logger.info(f"Saved features → {features_path}")
    ml_logger.info(f"Saved metrics  → {metrics_path}")
    ml_logger.info("Training complete.")

    return {k: {m: v for m, v in r.items() if m != "model"} for k, r in results.items()}


if __name__ == "__main__":
    run_training()
