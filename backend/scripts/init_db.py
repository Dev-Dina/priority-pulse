"""
Qdrant Collection Initialiser

Creates the 'tickets' collection in Qdrant if it doesn't already exist.
Run this before populate_vectors.py.

Usage:
    cd backend
    python scripts/init_db.py             # create if not exists
    python scripts/init_db.py --recreate  # drop and recreate (clean slate)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.logger import data_logger
from app.services.vector_service import VectorStore


def main(recreate: bool = False) -> None:
    data_logger.info("=" * 55)
    data_logger.info("QDRANT COLLECTION INIT")
    data_logger.info("=" * 55)

    store = VectorStore()

    try:
        store.create_collection(recreate=recreate)
        count = store.count()
        data_logger.info(f"Collection ready. Current point count: {count:,}")
    except Exception as e:
        data_logger.error(f"Failed to connect to Qdrant: {e}")
        data_logger.error(
            "Make sure Qdrant is running:\n"
            "  docker run -p 6333:6333 qdrant/qdrant\n"
            "or start the full stack:\n"
            "  docker compose up qdrant"
        )
        sys.exit(1)

    data_logger.info("Done. Run populate_vectors.py next.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialise Qdrant collection")
    parser.add_argument(
        "--recreate", action="store_true",
        help="Drop and recreate the collection (deletes all existing vectors)",
    )
    args = parser.parse_args()
    main(recreate=args.recreate)
