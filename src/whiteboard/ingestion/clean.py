from __future__ import annotations

import re
from typing import Dict, List

# Precompiled regexes for speed & clarity
_BRACKETED_CITATION = re.compile(r"\[(?:\d+|citation needed)\]", flags=re.IGNORECASE)
_EXTRA_SPACE = re.compile(r"[ \t\u00A0]+")
_MULTI_NL = re.compile(r"\n{3,}")

def _clean_text(t: str) -> str:
    if not t:
        return ""
    # Remove inline citation markers like [12] and [citation needed]
    t = _BRACKETED_CITATION.sub("", t)
    # Normalize spaces
    t = _EXTRA_SPACE.sub(" ", t)
    # Collapse excessive newlines to at most two
    t = _MULTI_NL.sub("\n\n", t)
    # Strip outer whitespace
    return t.strip()

def clean_sections(sections: List[Dict]) -> List[Dict]:
    """
    Input/Output shape (unchanged):
      { "title": str|None, "text": str, "url": str }

    - Drops empty/near-empty sections after cleaning
    - Leaves structure intact for downstream chunker
    """
    cleaned: List[Dict] = []
    for s in sections:
        text = _clean_text(s.get("text", ""))
        # Drop very short fragments (e.g., stray captions)
        if len(text) < 80:  # keep this conservative; chunker will do final sizing
            continue
        cleaned.append({
            "title": s.get("title"),
            "text": text,
            "url": s.get("url"),
        })
    return cleaned