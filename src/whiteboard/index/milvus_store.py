# src/whiteboard/index/milvus_store.py
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType, Collection,
    utility
)

# -----------------------------
# Config (env-driven)
# -----------------------------
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

_DEFAULT_COLLECTION = "whiteboard_chunks"
# sentence-transformers/all-MiniLM-L6-v2 -> 384 dims
_DEFAULT_DIM = 384

# Vector index defaults (balanced for MVP; switch if corpus grows)
# HNSW (COSINE) is a solid default for ST embeddings.
_VEC_INDEX_TYPE = "HNSW"           # alternatives: "IVF_FLAT", "FLAT"
_VEC_METRIC = "COSINE"             # aligns with sentence-transformers
_HNSW_PARAMS = {"M": 16, "efConstruction": 200}
_SEARCH_PARAMS = {"metric_type": _VEC_METRIC, "params": {"ef": 64}}

# Scalar filtering uses boolean expressions like: topic == "Linear algebra"
# (Milvus supports scalar filters during search.)  :contentReference[oaicite:3]{index=3}

class MilvusStore:
    """
    One-collection design:
      - Vector field: embedding (FLOAT_VECTOR)
      - Primary key: chunk_id (VarChar)
      - Filterable scalar fields: topic, tokens, embedding_model
      - Payload: text, url, title, section, ingested_at

    Rationale:
      - Use scalar filter `topic == ...` instead of many partitions for MVP. :contentReference[oaicite:4]{index=4}
      - HNSW index with COSINE for st-embeddings; IVF_FLAT/FLAT are alternatives. :contentReference[oaicite:5]{index=5}
    """

    def __init__(self, collection: str = _DEFAULT_COLLECTION, dim: int = _DEFAULT_DIM):
        self.collection_name = collection
        self.dim = dim
        # 1) connect once
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        # 2) ensure collection + index
        self.col = self._get_or_create_collection()
        self._ensure_indexes()
        self.col.load()

    # ---------- public API ----------

    def is_indexed(self, topic: str) -> bool:
        """
        Returns True iff there are any rows for this topic.
        Uses a lightweight `query` with a row limit.
        """
        topic = topic.strip()
        expr = f'topic == "{topic}"'
        try:
            res = self.col.query(expr=expr, output_fields=["chunk_id"], limit=1, consistency_level="Bounded")
            return len(res) > 0
        except Exception:
            return False

    def upsert(self, *, topic: str, items: List[Dict], vectors, force: bool = False) -> None:
        """
        Idempotent upsert: delete existing chunk_ids, then insert.
        - items[i] corresponds to vectors[i]
        Required item keys:
          chunk_id, text, tokens, url, title, section, embedding_model, ingested_at
        """
        if not items:
            return

        chunk_ids = [it["chunk_id"] for it in items]
        # Delete existing rows for these primary keys (if any)
        # (Milvus doesn't have true upsert; delete+insert is standard.)
        self._delete_by_ids(chunk_ids)

        # Prepare column-wise data (Milvus ORM expects columnar order)
        data = [
            chunk_ids,
            [topic] * len(items),
            [it["text"] for it in items],
            [int(it.get("tokens", 0)) for it in items],
            [it.get("embedding_model", "") for it in items],
            [it.get("url", "") for it in items],
            [it.get("title", "") for it in items],
            [it.get("section") if it.get("section") is not None else "" for it in items],
            [int(it.get("ingested_at", int(time.time()))) for it in items],
            vectors,  # FLOAT_VECTOR
        ]

        self.col.insert(data)
        # No need to recreate index; just ensure collection stays loaded
        self.col.flush()
        self.col.load()

    def search(self, *, topic: str, query_vector, k: int) -> List[Dict]:
        """
        Topic-scoped similarity search with scalar filter.
        Returns list of dicts with payload + score.
        """
        topic = topic.strip()
        expr = f'topic == "{topic}"'
        results = self.col.search(
            data=[query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)],
            anns_field="embedding",
            param=_SEARCH_PARAMS,
            limit=k,
            expr=expr,
            output_fields=["chunk_id", "text", "tokens", "embedding_model", "url", "title", "section"],
            consistency_level="Bounded",
        )

        hits = []
        if results and len(results) > 0:
            for hit in results[0]:
                fields = hit.entity.get("fields", None) or hit  # compatibility
                hits.append({
                    "chunk_id": fields["chunk_id"] if isinstance(fields, dict) else hit.entity.get("chunk_id"),
                    "text": fields["text"] if isinstance(fields, dict) else hit.entity.get("text"),
                    "tokens": int(fields["tokens"] if isinstance(fields, dict) else hit.entity.get("tokens")),
                    "embedding_model": fields["embedding_model"] if isinstance(fields, dict) else hit.entity.get("embedding_model"),
                    "url": fields["url"] if isinstance(fields, dict) else hit.entity.get("url"),
                    "title": fields["title"] if isinstance(fields, dict) else hit.entity.get("title"),
                    "section": fields["section"] if isinstance(fields, dict) else hit.entity.get("section"),
                    "score": float(hit.distance),
                })
        return hits

    def purge(self, topic: str) -> None:
        """Remove all rows belonging to a topic."""
        topic = topic.strip()
        expr = f'topic == "{topic}"'
        self.col.delete(expr)

    # ---------- internals ----------

    def _get_or_create_collection(self) -> Collection:
        # OLD:
        # if self.collection_name in [c.name for c in Collection.list_collections()]:
        #     return Collection(self.collection_name)

        # NEW:
        if utility.has_collection(self.collection_name):
            return Collection(self.collection_name)

        # Define schema (unchanged) ...
        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=128),
            FieldSchema(name="topic", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="tokens", dtype=DataType.INT64),
            FieldSchema(name="embedding_model", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="ingested_at", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
        ]
        schema = CollectionSchema(fields=fields, description="Whiteboard chunks (Wikipedia, cleaned & chunked)")
        col = Collection(name=self.collection_name, schema=schema)
        return col


    def _ensure_indexes(self) -> None:
        # Vector index
        existing = self.col.indexes
        if not any(idx.field_name == "embedding" for idx in existing):
            self.col.create_index(
                field_name="embedding",
                index_params={"index_type": _VEC_INDEX_TYPE, "metric_type": _VEC_METRIC, "params": _HNSW_PARAMS},
            )
        # (Optional) scalar indexes can be added later if needed for frequent filters.
        # Milvus supports scalar filtering without explicit scalar index; keep MVP simple. :contentReference[oaicite:6]{index=6}

    def _delete_by_ids(self, chunk_ids: List[str]) -> None:
        if not chunk_ids:
            return
        # Milvus delete supports boolean expressions; use IN list batching
        # chunk_id is a string primary key; wrap values in quotes.
        # Split into manageable batches to avoid overly long exprs.
        B = 500
        for i in range(0, len(chunk_ids), B):
            batch = chunk_ids[i : i + B]
            quoted = ",".join([f'"{cid}"' for cid in batch])
            expr = f"chunk_id in [{quoted}]"
            self.col.delete(expr)