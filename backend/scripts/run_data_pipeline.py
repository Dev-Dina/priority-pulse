"""
Data Processing Pipeline Runner

Steps:
  1. clean      — filter & normalise raw TWCS tweets, join agent responses
  2. label      — apply weak-supervision SLA triage (URGENT / NORMAL)
  3. ml_export  — engineer ML features and save a tabular CSV for model training
  4. stats      — save pipeline_stats.json for the frontend Data & Labels page

Run from the backend/ directory:
    python scripts/run_data_pipeline.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

import pandas as pd

from app.config import settings
from app.core.logger import data_logger
from app.data_processing.cleaners import DataCleaner
from app.data_processing.labelers import PriorityLabeler
from app.ml.feature_engineering import engineer_features


# ── step 1: clean ──────────────────────────────────────────────────────────

def run_cleaning() -> tuple[pd.DataFrame, dict]:
    data_logger.info("=" * 50)
    data_logger.info("STARTING CLEANING PIPELINE")
    data_logger.info("=" * 50)

    cleaner = DataCleaner()
    df_raw = cleaner.load_raw_data()
    data_logger.info(f"Raw shape: {df_raw.shape}")

    df_cleaned = cleaner.clean_pipeline(df_raw)
    cleaner.save_cleaned_data(df_cleaned)

    data_logger.info("CLEANING COMPLETE")
    return df_cleaned, cleaner.stats


# ── step 2: label ──────────────────────────────────────────────────────────

def run_labeling(df_cleaned: pd.DataFrame) -> pd.DataFrame:
    data_logger.info("=" * 50)
    data_logger.info("STARTING SLA LABELING PIPELINE")
    data_logger.info("=" * 50)

    labeler = PriorityLabeler()
    df_labeled = labeler.label_dataset(df_cleaned, text_column="text")

    output_path = settings.LABELED_DATASET_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_labeled.to_csv(output_path, index=False)

    data_logger.info(f"Saved labeled data to {output_path}")
    data_logger.info(f"Shape: {df_labeled.shape}")

    debug_cols = ["text", "priority", "score", "has_critical_signal"]
    available = [c for c in debug_cols if c in df_labeled.columns]
    if len(available) >= 2:
        for _, row in df_labeled[available].head(3).iterrows():
            data_logger.info(
                f"  [{row.get('priority')} | score={row.get('score')}] "
                f"{str(row.get('text', ''))[:80]}…"
            )

    data_logger.info("LABELING COMPLETE")
    return df_labeled


# ── step 3: ML feature export ──────────────────────────────────────────────

def run_ml_export(df_labeled: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to every labeled ticket and save ml_features.csv.

    Columns:
      Identity   — tweet_id, airline, created_at
      Target     — is_urgent (1/0), priority (label), score (0–10)
      ML features — 14 engineered features (independent of labeling rule)
      Label features — 4 weak-supervision rule flags (stored for transparency)

    Honesty note: is_urgent was derived from the labeling rule, so the model
    will partly learn to reproduce that rule. That is weak supervision —
    expected and documented.
    """
    data_logger.info("=" * 50)
    data_logger.info("STARTING ML FEATURE EXPORT")
    data_logger.info("=" * 50)

    df = df_labeled.copy()
    df["is_urgent"] = (df["priority"] == "URGENT").astype(int)

    data_logger.info(f"Engineering features for {len(df):,} tickets…")
    feature_rows = df["text"].apply(engineer_features).apply(pd.Series)
    df = df.drop(columns=[c for c in feature_rows.columns if c in df.columns], errors="ignore")
    df = pd.concat([df, feature_rows], axis=1)

    id_cols = ["tweet_id", "airline", "created_at"]
    target_cols = ["is_urgent", "priority", "score"]
    ml_features = [
        "text_length", "word_count", "exclamation_count", "question_count", "caps_ratio",
        "has_refund", "has_cancel", "has_delay", "has_help", "has_broken",
        "has_stranded", "has_medical", "profanity_count", "has_time_mention",
    ]
    label_features = [
        "has_urgency_keywords", "has_critical_signal", "has_profanity", "has_sarcasm",
    ]

    keep = id_cols + target_cols + label_features + ml_features
    keep = [c for c in keep if c in df.columns]
    df_ml = df[keep]

    output_path = settings.DATA_PROCESSED_DIR / "ml_features.csv"
    df_ml.to_csv(output_path, index=False)

    data_logger.info(f"Saved ML feature table to {output_path}")
    data_logger.info(f"Shape: {df_ml.shape}  |  columns: {df_ml.columns.tolist()}")
    data_logger.info(
        f"Class balance — is_urgent=1: {df_ml['is_urgent'].sum():,} "
        f"({100*df_ml['is_urgent'].mean():.1f}%)  "
        f"is_urgent=0: {(df_ml['is_urgent']==0).sum():,}"
    )
    data_logger.info("ML FEATURE EXPORT COMPLETE")
    return df_ml


# ── step 4: save pipeline stats ────────────────────────────────────────────

def save_pipeline_stats(cleaning_stats: dict, df_labeled: pd.DataFrame) -> None:
    """
    Write pipeline_stats.json to data/processed/ for the frontend to display.
    Includes cleaning funnel, agent-response coverage, labeling distribution,
    and the non-selected brand handles that appear in TWCS.
    """
    label_counts = df_labeled["priority"].value_counts()
    total = len(df_labeled)

    urgent = int(label_counts.get("URGENT", 0))
    normal = int(label_counts.get("NORMAL", 0))

    stats = {
        **cleaning_stats,
        "labeling": {
            "urgent": urgent,
            "normal": normal,
            "urgent_pct": round(100 * urgent / total, 1) if total else 0,
        },
        "pipeline_run_at": datetime.now(timezone.utc).isoformat(),
    }

    stats_path = settings.DATA_PROCESSED_DIR / "pipeline_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    data_logger.info(f"Pipeline stats saved → {stats_path}")


# ── orchestrator ───────────────────────────────────────────────────────────

def main():
    data_logger.info("Starting full data pipeline…")

    df_cleaned, cleaning_stats = run_cleaning()
    df_labeled = run_labeling(df_cleaned)
    run_ml_export(df_labeled)
    save_pipeline_stats(cleaning_stats, df_labeled)

    data_logger.info("=" * 50)
    data_logger.info("PIPELINE COMPLETE")
    data_logger.info("=" * 50)
    data_logger.info(f"Rows processed: {len(df_labeled):,}")
    data_logger.info("Outputs:")
    data_logger.info(f"  cleaned  → {settings.CLEANED_DATASET_PATH}")
    data_logger.info(f"  labeled  → {settings.LABELED_DATASET_PATH}")
    data_logger.info(f"  ml feats → {settings.DATA_PROCESSED_DIR / 'ml_features.csv'}")
    data_logger.info(f"  stats    → {settings.DATA_PROCESSED_DIR / 'pipeline_stats.json'}")
    data_logger.info("Next: python scripts/init_db.py && python scripts/populate_vectors.py")


if __name__ == "__main__":
    main()
