"""
RAG Service — retrieval side of the pipeline.

Owns the embed-then-search step so the query endpoint stays thin.
Also computes whether similarity is "low" so the frontend can warn the user
when retrieved context might be irrelevant.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from app.config import settings
from app.core.logger import app_logger
from app.services.vector_service import EmbeddingService, vector_store


class RAGService:

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        threshold: float | None = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Embed the query and fetch the top-k most similar tickets from Qdrant.

        Returns:
            embedding — the query vector (passed to query.py for optional logging)
            sources   — list of dicts: text, airline, priority, similarity
        """
        embedding = EmbeddingService.embed_single(query)
        sources = vector_store.retrieve_similar(
            embedding,
            k=k or settings.RAG_TOP_K,
            threshold=threshold if threshold is not None else settings.RAG_SIMILARITY_THRESHOLD,
        )
        app_logger.debug(
            f"Retrieved {len(sources)} sources "
            f"(top similarity={sources[0]['similarity'] if sources else 'n/a'})"
        )
        return embedding, sources

    def is_low_similarity(self, sources: List[Dict[str, Any]]) -> bool:
        """
        True when Qdrant returned nothing or the best match is below the threshold.

        The frontend shows a warning when True so the user knows the RAG answer
        may not be grounded in relevant past cases.
        """
        if not sources:
            return True
        return sources[0]["similarity"] < settings.RAG_SIMILARITY_THRESHOLD


rag_service = RAGService()
