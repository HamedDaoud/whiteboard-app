# src/whiteboard/retrieval.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import time

# Source adapters / processing
from .ingestion.wikipedia import fetch as wiki_fetch
from .ingestion.clean import clean_sections
from .ingestion.chunk import chunk_sections

# Embeddings and index
from .embeddings.model import Embedder
from .index.milvus_store import MilvusStore


# ---------- Public schema ----------

@dataclass(frozen=True)
class RetrievedChunk:
    topic: str
    chunk_id: str
    text: str
    score: float
    tokens: int
    embedding_model: str
    source: Dict[str, Optional[str]]  # {kind,url,title,section}


# ---------- Exceptions ----------

class RetrievalError(Exception):
    """Raised for any failure in the retrieval pipeline."""
    pass


# ---------- Service ----------

class RetrievalService:
    """
    Orchestrates:
      - On-demand ingestion: fetch → clean → chunk → embed → upsert
      - Retrieval: query embedding → vector search (topic-scoped)
    """

    def __init__(
        self,
        store: Optional[MilvusStore] = None,
        embedder: Optional[Embedder] = None,
        *,
        chunk_max_tokens: int = 800,
        chunk_overlap: int = 100,
    ) -> None:
        self.store = store or MilvusStore()
        self.embedder = embedder or Embedder()
        self.chunk_max_tokens = chunk_max_tokens
        self.chunk_overlap = chunk_overlap

    # ----- Public entry points -----

    def get_chunks(
        self, topic: str, query: Optional[str] = None, k: int = 6
    ) -> List[RetrievedChunk]:
        """
        Returns top-k relevant chunks for `topic` (and optional `query`).
        Ingests the topic on first request; subsequent calls reuse the index.

        Output list is sorted by descending score.
        """
        topic = topic.strip()
        if not topic:
            raise ValueError("topic must be a non-empty string")
        if k <= 0:
            raise ValueError("k must be positive")

        try:
            self._ensure_indexed(topic)
            # Build query vector
            if query and query.strip():
                qvec = self.embedder.encode_one(query.strip())
            else:
                # Default query = topic string (simple, stable, fast)
                qvec = self.embedder.encode_one(topic)

            # Topic-scoped search
            hits = self.store.search(topic=topic, query_vector=qvec, k=k)

            # Expected hit item schema from store:
            # {
            #   "chunk_id": str, "score": float, "text": str,
            #   "tokens": int, "url": str, "title": str, "section": Optional[str],
            #   "embedding_model": str
            # }
            results: List[RetrievedChunk] = []
            for h in hits:
                results.append(
                    RetrievedChunk(
                        topic=topic,
                        chunk_id=h["chunk_id"],
                        text=h["text"],
                        score=float(h["score"]),
                        tokens=int(h.get("tokens", 0)),
                        embedding_model=h.get("embedding_model", self.embedder.model_name),
                        source={
                            "kind": "wikipedia",
                            "url": h.get("url"),
                            "title": h.get("title"),
                            "section": h.get("section"),
                        },
                    )
                )
            return results
        except Exception as e:  # noqa: BLE001
            raise RetrievalError(f"get_chunks failed: {e}") from e

    def is_indexed(self, topic: str) -> bool:
        return self.store.is_indexed(topic.strip())

    def reingest(self, topic: str) -> None:
        """Force reingestion (useful if sources changed)."""
        self._ingest(topic.strip(), force=True)

    def purge(self, topic: str) -> None:
        """Optional maintenance: remove a topic from the store."""
        self.store.purge(topic.strip())

    # ----- Internal helpers -----

    def _ensure_indexed(self, topic: str) -> None:
        if self.store.is_indexed(topic):
            return
        self._ingest(topic, force=False)

    def _ingest(self, topic: str, *, force: bool) -> None:
        # 1) Fetch
        article = wiki_fetch(topic)
        # Expected shape:
        # {
        #   "title": str, "url": str,
        #   "sections": [{"title": Optional[str], "text": str, "url": str}]
        # }
        sections = article.get("sections", [])
        if not sections:
            raise RetrievalError("No content sections fetched for topic")

        # 2) Clean
        clean_secs = clean_sections(sections)  # same structure, text cleaned

        # 3) Chunk
        chunks = chunk_sections(
            clean_secs,
            max_tokens=self.chunk_max_tokens,
            overlap=self.chunk_overlap,
            model_name=self.embedder.model_name,
        )
        # Each chunk:
        # {"chunk_id": str, "text": str, "tokens": int,
        #  "section": Optional[str], "url": str, "title": str}
        if not chunks:
            raise RetrievalError("No chunks produced after cleaning")

        # 4) Embed (batched)
        vectors = self.embedder.encode([c["text"] for c in chunks])

        # 5) Upsert
        self.store.upsert(
            topic=topic,
            items=[
                dict(
                    chunk_id=c["chunk_id"],
                    text=c["text"],
                    tokens=int(c.get("tokens", 0)),
                    url=c.get("url"),
                    title=c.get("title", article.get("title")),
                    section=c.get("section"),
                    embedding_model=self.embedder.model_name,
                    ingested_at=int(time.time()),
                )
                for c in chunks
            ],
            vectors=vectors,
            force=force,
        )


# ---------- Module-level convenience ----------

# A lightweight singleton-style service for callers that don't want to manage instances.
_service: Optional[RetrievalService] = None


def _get_service() -> RetrievalService:
    global _service
    if _service is None:
        _service = RetrievalService()
    return _service


def get_chunks(topic: str, query: Optional[str] = None, k: int = 6) -> List[Dict[str, Any]]:
    """
    Functional API wrapper. Returns a list[dict] using the public schema.
    """
    svc = _get_service()
    results = svc.get_chunks(topic=topic, query=query, k=k)
    # Convert dataclass to plain dicts for easy JSON serialization.
    return [vars(r) for r in results]


def is_indexed(topic: str) -> bool:
    return _get_service().is_indexed(topic)


def reingest(topic: str) -> None:
    _get_service().reingest(topic)


def purge(topic: str) -> None:
    _get_service().purge(topic)