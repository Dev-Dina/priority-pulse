"""
Vector store operations using Qdrant.

Why Qdrant:
  - Runs as a dedicated Docker service.
  - Zero extension or driver setup — just pull the image.
  - Native cosine similarity, payload filtering, and scroll API.
  - Python client (qdrant-client) is clean and well-maintained.

Two classes:
  EmbeddingService  — loads sentence-transformers model, generates embeddings
  VectorStore       — thin wrapper around QdrantClient for insert + search
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.core.logger import app_logger

# ── Embedding ──────────────────────────────────────────────────────────────

class EmbeddingService:
    """
    Lazy singleton wrapper around SentenceTransformer.

    normalize_embeddings=True → dot product == cosine similarity,
    which is what Qdrant's Distance.COSINE uses.
    """

    _model: Optional[SentenceTransformer] = None

    @classmethod
    def get_model(cls) -> SentenceTransformer:
        if cls._model is None:
            app_logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            cls._model = SentenceTransformer(settings.EMBEDDING_MODEL)
            app_logger.info("Embedding model ready.")
        return cls._model

    @classmethod
    def embed(
        cls,
        texts: List[str],
        batch_size: int = 256,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts → float32 ndarray of shape (N, 384)."""
        return cls.get_model().encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    @classmethod
    @lru_cache(maxsize=512)
    def embed_single(cls, text: str) -> np.ndarray:
        return cls.embed([text])[0]


# ── Vector store ───────────────────────────────────────────────────────────

class VectorStore:
    """
    Qdrant operations: create collection, upsert points, similarity search.

    Each Qdrant point holds:
      vector  — 384-dim float32 embedding
      payload — all ticket metadata (text, airline, priority, score, features)

    The point ID is the row index (0-based integer) from the labeled CSV.
    tweet_id is stored in the payload for lookups.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection: str | None = None,
    ):
        self._host = host or settings.QDRANT_HOST
        self._port = port or settings.QDRANT_PORT
        self.collection = collection or settings.QDRANT_COLLECTION
        self._client: Optional[QdrantClient] = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(host=self._host, port=self._port)
        return self._client

    # ── setup ──────────────────────────────────────────────────────────────

    def create_collection(self, recreate: bool = False) -> None:
        """
        Create the Qdrant collection if it does not already exist.

        recreate=True drops and rebuilds (use for a clean re-run).
        Distance.COSINE works with normalized vectors from sentence-transformers.
        """
        existing = [c.name for c in self.client.get_collections().collections]

        if self.collection in existing:
            if recreate:
                app_logger.info(f"Dropping existing collection '{self.collection}'…")
                self.client.delete_collection(self.collection)
            else:
                app_logger.info(
                    f"Collection '{self.collection}' already exists — skipping creation."
                )
                return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=settings.EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
        app_logger.info(
            f"Created collection '{self.collection}' "
            f"(dim={settings.EMBEDDING_DIM}, distance=COSINE)."
        )

    def count(self) -> int:
        return self.client.count(collection_name=self.collection).count

    # ── write ──────────────────────────────────────────────────────────────

    def upsert_batch(self, points: List[PointStruct]) -> None:
        self.client.upsert(collection_name=self.collection, points=points)

    # ── read / similarity search ───────────────────────────────────────────

    def retrieve_similar(
        self,
        query_embedding: np.ndarray,
        k: int | None = None,
        threshold: float | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Return top-k tickets by cosine similarity.

        Qdrant's score for Distance.COSINE is the cosine similarity directly
        (1 = identical, 0 = orthogonal).  score_threshold filters low-quality
        results before they reach the LLM context.
        """
        k = k or settings.RAG_TOP_K
        threshold = threshold if threshold is not None else settings.RAG_SIMILARITY_THRESHOLD

        result = self.client.query_points(
            collection_name=self.collection,
            query=query_embedding.tolist(),
            limit=k,
            with_payload=True,
            score_threshold=threshold,
        )

        return [
            {
                "tweet_id":      h.payload.get("tweet_id"),
                "airline":       h.payload.get("airline"),
                "text":          h.payload.get("original_text"),
                "agent_response": h.payload.get("agent_response", ""),
                "priority":      h.payload.get("priority"),
                "priority_score": h.payload.get("priority_score"),
                "similarity":    round(h.score, 4),
            }
            for h in result.points
        ]

    def retrieve_similar_by_text(
        self,
        query: str,
        k: int | None = None,
        threshold: float | None = None,
    ) -> List[Dict[str, Any]]:
        """Convenience: embed the query then search."""
        return self.retrieve_similar(
            EmbeddingService.embed_single(query),
            k=k,
            threshold=threshold,
        )


# ── module-level singleton (used by FastAPI routes) ────────────────────────
vector_store = VectorStore()
