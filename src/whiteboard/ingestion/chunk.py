# src/whiteboard/ingestion/chunk.py
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional

from transformers import AutoTokenizer

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _get_tokenizer(model_name: str) -> AutoTokenizer:
    # Use the same family as the embedder to avoid silent truncation
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # Some ST models expose max_seq_length=256; use tokenizer.model_max_length as a hint
    return tok

def _window_token_ids(
    token_ids: List[int],
    max_tokens_wo_special: int,
    overlap: int,
) -> List[tuple[int, int]]:
    """
    Produce (start, end) windows over `token_ids` with given size/overlap.
    end is exclusive.
    """
    n = len(token_ids)
    if n == 0:
        return []
    if max_tokens_wo_special <= 0:
        return [(0, n)]
    step = max(max_tokens_wo_special - overlap, 1)
    spans = []
    start = 0
    while start < n:
        end = min(start + max_tokens_wo_special, n)
        spans.append((start, end))
        if end == n:
            break
        start += step
    return spans

def _decode_slice(tok: AutoTokenizer, ids: List[int]) -> str:
    # Decode a slice *without* adding special tokens to keep plain text
    return tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()

def chunk_sections(
    sections: List[Dict],
    *,
    max_tokens: int = 256,
    overlap: int = 32,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    min_chars: int = 80,   # drop tiny fragments
) -> List[Dict]:
    """
    Input sections: [{ "title": str|None, "text": str, "url": str }]
    Output chunks:  [{
        "chunk_id": str,          # stable hash
        "text": str,
        "tokens": int,
        "section": str|None,
        "url": str
    }, ...]
    Notes:
      - Chunks sized by the *embedder's* tokenizer to avoid truncation.
      - `max_tokens` counts WordPiece tokens *including* special tokens in the embedder.
        We reserve 2 tokens ([CLS], [SEP]) for BERT-family models → we window size = max_tokens - 2.
    """
    assert max_tokens > 8, "max_tokens should be > 8"
    assert overlap >= 0, "overlap must be non-negative"

    tok = _get_tokenizer(model_name)

    # Reserve for special tokens commonly added by BERT encoders
    special_reserve = 2
    size_wo_special = max(max_tokens - special_reserve, 1)

    chunks: List[Dict] = []
    for sec in sections:
        raw = (sec.get("text") or "").strip()
        if not raw:
            continue
        if len(raw) < min_chars:
            continue

        # Tokenize WITHOUT adding special tokens; we manage windows ourselves
        enc = tok(
            raw,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        ids: List[int] = enc["input_ids"]

        # Make windows over token ids
        spans = _window_token_ids(ids, size_wo_special, overlap)

        # Use section URL (includes page URL or page#anchor from wikipedia.py)
        sec_title: Optional[str] = sec.get("title")
        sec_url: str = sec.get("url") or ""

        for i, (a, b) in enumerate(spans):
            piece_ids = ids[a:b]
            text = _decode_slice(tok, piece_ids)
            if len(text) < min_chars:
                continue

            # Stable chunk id: hash of (url | section title | token-start-index | token-end-index)
            base = f"{sec_url}|{sec_title or ''}|{a}|{b}"
            cid = _sha1(base)

            chunks.append({
                "chunk_id": cid,
                "text": text,
                "tokens": len(piece_ids) + special_reserve,  # approximate embed-time length
                "section": sec_title,
                "url": sec_url,
                # NOTE: no "title" (page title) here; retrieval.py will fill from article.title
            })

    return chunks