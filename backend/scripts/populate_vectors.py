"""
Chunking & Embedding Pipeline → Qdrant

Reads labeled_tickets.csv → one chunk per tweet → embeds with
sentence-transformers → upserts into Qdrant.

─── Chunking strategy ────────────────────────────────────────────────────────
Tweets are already short (≤ 280 chars after cleaning), so splitting further
would destroy meaning. Each tweet = one chunk.

The chunk_text embeds the customer complaint only, prefixed with the airline
name for domain context:
    "[Delta Air Lines] my bag never arrived and nobody answers"
Agent responses are stored in the Qdrant payload and injected into the LLM
prompt at query time — not embedded, so the vector stays in complaint space
and aligns with the user's query.
The priority label is excluded so retrieval is content-driven, not label-driven.

─── Resumable ────────────────────────────────────────────────────────────────
Re-running is safe — Qdrant upsert overwrites the same point ID, so no
duplicates. Use --recreate to wipe and start fresh.

Usage:
    cd backend
    python scripts/populate_vectors.py              # full 90k run
    python scripts/populate_vectors.py --limit 500  # smoke-test
    python scripts/populate_vectors.py --recreate   # wipe + re-insert
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from qdrant_client.models import PointStruct

from app.config import settings
from app.core.logger import data_logger
from app.services.vector_service import EmbeddingService, VectorStore

EMBED_BATCH = 256    # rows per sentence-transformers call
INSERT_BATCH = 256   # points per Qdrant upsert call


# ── helpers ────────────────────────────────────────────────────────────────

def create_chunk_text(row: pd.Series) -> str:
    """
    Build the text that gets embedded — the customer complaint only.

    Format: "[Airline] <customer tweet>"

    The airline prefix gives domain context.
    Agent responses are stored in the payload and injected into the LLM
    prompt at query time. They must not be part of the embedded text because
    blending complaint + resolution pulls the vector away from complaint space,
    misaligning it with the user's query (which is always a complaint).
    The priority label is excluded so retrieval stays content-driven.
    """
    airline = str(row.get("airline", "Unknown")).strip()
    text = str(row.get("text", "")).strip()
    return f"[{airline}] {text}"


def load_labeled_data(limit: int | None = None) -> pd.DataFrame:
    path = settings.LABELED_DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Labeled dataset not found at {path}.\n"
            "Run:  python scripts/run_data_pipeline.py  first."
        )
    df = pd.read_csv(path)
    data_logger.info(f"Loaded {len(df):,} tickets from {path.name}")
    if limit:
        df = df.head(limit)
        data_logger.info(f"Limited to first {limit:,} rows (--limit flag).")
    return df


def safe_val(v, cast=None):
    """Return None for NaN/NaT, otherwise optionally cast."""
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return cast(v) if cast else v


def build_payload(row: pd.Series) -> dict:
    """
    All ticket metadata stored alongside the vector in Qdrant.
    This becomes the 'source panel' data returned to the frontend.
    agent_response is the cleaned outbound reply from the support agent —
    present when the agent responded to this ticket in the TWCS dataset.
    """
    return {
        "tweet_id": safe_val(row.get("tweet_id"), int),
        "customer_id": safe_val(row.get("customer_id"), int),
        "airline": str(row.get("airline", "Unknown")),
        "created_at": str(row.get("created_at", "")),
        "original_text": str(row.get("original_text", "")),
        "agent_response": str(row.get("agent_response", "")),
        "chunk_text": str(row.get("chunk_text", "")),
        "priority": str(row.get("priority", "NORMAL")),
        "priority_score": safe_val(row.get("score", 0.0), float) or 0.0,
        # labeling features — stored for transparency
        "has_urgency_keywords": safe_val(row.get("has_urgency_keywords"), bool),
        "has_critical_signal": safe_val(row.get("has_critical_signal"), bool),
        "has_delay": safe_val(row.get("has_delay"), bool),
        "has_profanity": safe_val(row.get("has_profanity"), bool),
        "caps_ratio": safe_val(row.get("caps_ratio"), float),
        "exclamation_count": safe_val(row.get("exclamation_count"), int),
    }


# ── main pipeline ──────────────────────────────────────────────────────────

def populate(limit: int | None = None, recreate: bool = False) -> None:
    # ── 1. load data ───────────────────────────────────────────────────────
    df = load_labeled_data(limit)

    # ── 2. chunking (1 tweet = 1 chunk, enriched with airline prefix) ──────
    data_logger.info("Creating chunk texts…")
    df["chunk_text"] = df.apply(create_chunk_text, axis=1)
    df = df.reset_index(drop=True)   # index becomes Qdrant point ID

    # ── 3. init Qdrant collection ──────────────────────────────────────────
    store = VectorStore()
    store.create_collection(recreate=recreate)

    existing = store.count()
    if existing > 0 and not recreate:
        data_logger.info(
            f"Collection already has {existing:,} points. "
            "Pass --recreate to wipe and re-insert."
        )
        return

    # ── 4. generate embeddings ─────────────────────────────────────────────
    total = len(df)
    data_logger.info(
        f"Generating embeddings for {total:,} chunks "
        f"(model={settings.EMBEDDING_MODEL}, batch={EMBED_BATCH})…"
    )
    texts = df["chunk_text"].tolist()
    embeddings = EmbeddingService.embed(texts, batch_size=EMBED_BATCH, show_progress=True)
    data_logger.info(f"Embeddings ready: shape={embeddings.shape}")

    # ── 5. upsert into Qdrant in batches ──────────────────────────────────
    data_logger.info(f"Upserting into Qdrant (batch={INSERT_BATCH})…")
    num_batches = (total + INSERT_BATCH - 1) // INSERT_BATCH
    total_inserted = 0

    for i in range(num_batches):
        start = i * INSERT_BATCH
        end = min(start + INSERT_BATCH, total)

        points = [
            PointStruct(
                id=int(idx),                        # row index as stable point ID
                vector=embeddings[idx].tolist(),
                payload=build_payload(df.iloc[idx]),
            )
            for idx in range(start, end)
        ]

        store.upsert_batch(points)
        total_inserted += len(points)

        pct = 100 * total_inserted / total
        data_logger.info(
            f"  batch {i + 1}/{num_batches} — {total_inserted:,}/{total:,} ({pct:.1f}%)"
        )

    data_logger.info(f"Done. Total points in Qdrant: {store.count():,}")


# ── entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chunk, embed, and upsert tickets into Qdrant"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N rows (smoke-test)",
    )
    parser.add_argument(
        "--recreate", action="store_true",
        help="Delete the collection and re-insert everything",
    )
    args = parser.parse_args()

    data_logger.info("=" * 60)
    data_logger.info("CHUNKING & EMBEDDING PIPELINE (Qdrant)")
    data_logger.info("=" * 60)
    populate(limit=args.limit, recreate=args.recreate)
    data_logger.info("=" * 60)
    data_logger.info("PIPELINE COMPLETE")
    data_logger.info("=" * 60)
    data_logger.info("Next: start the FastAPI backend and test /api/routes/query")
