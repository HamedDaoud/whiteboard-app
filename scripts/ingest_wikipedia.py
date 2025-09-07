# scripts/ingest_wikipedia.py
from __future__ import annotations

import argparse
import sys
from textwrap import shorten
from pathlib import Path

# --- make 'src' importable ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYS_SRC = PROJECT_ROOT / "src"
if str(SYS_SRC) not in sys.path:
    sys.path.insert(0, str(SYS_SRC))

# Orchestrator API
from whiteboard.retrieval import is_indexed, reingest, get_chunks
# For counting/diagnostics
from whiteboard.index.milvus_store import MilvusStore
from whiteboard.embeddings.model import Embedder



def _count_chunks(store: MilvusStore, topic: str) -> int:
    """
    Count stored chunks for a topic (best-effort within Milvus query window).
    Milvus enforces (offset + limit) <= 16384, so we cap the query.
    Returns the count up to 16384; if you need exact counts beyond that,
    prefer an aggregation or stats API.
    """
    expr = f'topic == "{topic.strip()}"'
    try:
        rows = store.col.query(expr=expr, output_fields=["chunk_id"], limit=16384)
        return len(rows)
    except Exception as e:
        print(f"[warn] count failed: {e}")
        return -1



def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest a Wikipedia topic into Milvus.")
    parser.add_argument("topic", type=str, help='Topic title, e.g. "Linear algebra"')
    parser.add_argument("--force", action="store_true", help="Force re-ingestion (delete/replace existing chunks).")
    parser.add_argument("--k", type=int, default=3, help="How many preview chunks to show after ingestion.")
    args = parser.parse_args()

    topic = args.topic.strip()
    if not topic:
        print("Topic must be a non-empty string.")
        return 2

    store = MilvusStore()  # uses MILVUS_HOST/PORT from env
    already = is_indexed(topic)

    if args.force or not already:
        action = "Re-ingesting" if args.force and already else "Ingesting"
        print(f"{action} topic: {topic!r} ...")
        # Force reingest ensures a fresh pull & upsert
        reingest(topic)
    else:
        print(f"Topic {topic!r} already indexed. Skipping ingestion.")

    # Count how many chunks are present
    n = _count_chunks(store, topic)
    if n >= 0:
        print(f"Ingested chunks for {topic!r}: {n}")

    # Show a few preview hits (topic-as-query)
    print("\nPreview (top-k):")
    try:
        # Using topic as the query gives a representative sample
        previews = get_chunks(topic, query=topic, k=args.k)
        for i, ch in enumerate(previews, 1):
            text_preview = shorten(ch["text"].replace("\n", " "), width=160, placeholder="…")
            print(f"{i:>2}. score={ch['score']:.4f}  tokens={ch['tokens']:>3}  url={ch['source']['url']}")
            print(f"    {text_preview}")
    except Exception as e:
        print(f"[warn] preview search failed: {e}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())