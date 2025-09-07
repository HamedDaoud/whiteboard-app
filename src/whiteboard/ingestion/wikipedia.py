from __future__ import annotations

from typing import Dict, List, Optional
from urllib.parse import quote

import wikipediaapi

_BLACKLIST_SECTIONS = {
    "references",
    "external links",
    "see also",
    "further reading",
    "notes",
    "sources",
    "bibliography",
}

def _wiki_client(lang: str = "en") -> wikipediaapi.Wikipedia:
    return wikipediaapi.Wikipedia(
        user_agent="whiteboard-app/0.1 (https://github.com/HamedDaoud/whiteboard-app)",
        language=lang,
        extract_format=wikipediaapi.ExtractFormat.WIKI,
    )


def _page_url(title: str, lang: str = "en") -> str:
    base = f"https://{lang}.wikipedia.org/wiki/"
    return base + quote(title.replace(" ", "_"))

def _section_anchor(title: str) -> str:
    # Wikipedia anchors use underscores; quoting handles punctuation safely.
    return "#" + quote(title.replace(" ", "_"))

def _collect_sections(node, page_title: str, page_url: str) -> List[Dict]:
    out: List[Dict] = []
    for s in node.sections:
        if s.title and s.title.strip().lower() in _BLACKLIST_SECTIONS:
            # skip low-signal sections
            continue
        # Section text is already plaintext with wikipediaapi
        out.append({
            "title": s.title or None,
            "text": s.text or "",
            "url": page_url + (_section_anchor(s.title) if s.title else ""),
        })
        # Recurse into subsections
        out.extend(_collect_sections(s, page_title, page_url))
    return out

def fetch(topic: str, lang: str = "en") -> Dict:
    """
    Returns:
    {
      "title": str,
      "url": str,
      "sections": [ { "title": str|None, "text": str, "url": str } ]
    }
    """
    topic = topic.strip()
    if not topic:
        raise ValueError("topic must be a non-empty string")

    wiki = _wiki_client(lang)
    page = wiki.page(topic)

    if not page.exists():
        # Try capitalized (common case), else raise
        alt = topic[:1].upper() + topic[1:]
        page = wiki.page(alt)
        if not page.exists():
            raise ValueError(f"Wikipedia page not found for topic: {topic!r}")

    title = page.title
    url = _page_url(title, lang)

    sections: List[Dict] = []

    # Add a lead section first (high-signal for LLM prompts).
    lead_text = (page.summary or "").strip()
    if lead_text:
        sections.append({"title": None, "text": lead_text, "url": url})

    # Add all remaining sections (recursively)
    sections.extend(_collect_sections(page, title, url))

    # Keep deterministic order (lead first, then natural traversal order)
    return {"title": title, "url": url, "sections": sections}