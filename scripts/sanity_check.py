# scripts/sanity_check.py
from __future__ import annotations

import sys
from textwrap import shorten
from pathlib import Path

# --- make 'src' importable ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYS_SRC = PROJECT_ROOT / "src"
if str(SYS_SRC) not in sys.path:
    sys.path.insert(0, str(SYS_SRC))

from whiteboard.retrieval import get_chunks



def main() -> int:
    topic = "Linear algebra"
    query = "eigenvalues"
    k = 4

    print(f"[sanity] Testing retrieval pipeline with topic={topic!r}, query={query!r} …")

    try:
        chunks = get_chunks(topic, query=query, k=k)
    except Exception as e:
        print(f"[fail] retrieval error: {e}")
        return 1

    if not chunks or len(chunks) < k:
        print(f"[fail] expected at least {k} chunks, got {len(chunks) if chunks else 0}")
        return 2

    # Check fields
    required = {"topic", "chunk_id", "text", "score", "tokens", "embedding_model", "source"}
    for i, ch in enumerate(chunks, 1):
        missing = required - set(ch.keys())
        if missing:
            print(f"[fail] chunk {i} missing fields: {missing}")
            return 3
        if not ch["text"] or not ch["source"].get("url"):
            print(f"[fail] chunk {i} has empty text or url")
            return 4

    # Print preview
    for i, ch in enumerate(chunks, 1):
        text_preview = shorten(ch["text"].replace("\n", " "), width=100, placeholder="…")
        print(f"  {i:>2}. score={ch['score']:.4f} tokens={ch['tokens']} url={ch['source']['url']}")
        print(f"      {text_preview}")

    print("\nALL GOOD ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())