# Whiteboard App

Minimal RAG bootstrap with **Python 3.11** and **Milvus Standalone**.  
Implements ingestion, cleaning, chunking, embedding, indexing, and retrieval of Wikipedia text.  
All other teams interact through **`get_chunks(topic, query, k)`**.

---

## Prerequisites
- **Python 3.11**
  - Windows: install from python.org and verify with `py -3.11 -V`
  - macOS: `brew install python@3.11` (or python.org) and verify with `python3.11 -V`
- **Docker Desktop** (WSL2 backend on Windows) — running
- **Git**

---

## Quick Start

### Windows (PowerShell)
```powershell
git clone https://github.com/HamedDaoud/whiteboard-app.git
cd whiteboard-app

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

cp .env.example .env

docker compose up -d

python scripts\smoke.py   # full pipeline check, expect: ALL GOOD ✅
```

### macOS / Linux
```bash
git clone https://github.com/HamedDaoud/whiteboard-app.git
cd whiteboard-app

python3.11 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env

docker compose up -d

python scripts/smoke.py   # full pipeline check, expect: ALL GOOD ✅
```

---

## Project Layout
```
whiteboard-app/
  docker-compose.yml      # Milvus (standalone) + deps
  requirements.txt
  .env.example
  scripts/
    ingest_wikipedia.py   # manually ingest a topic
    sanity_check.py       # check retrieval on an already indexed topic
    smoke.py              # end-to-end pipeline check
  src/whiteboard/
    retrieval.py          # public API
    ingestion/            # fetch, clean, chunk
    embeddings/           # embedding wrapper
    index/                # Milvus integration
```

---

## Retrieval Flow

- **Topic (required)**  
  - Defines the scope of retrieval (which Wikipedia page).  
  - If not indexed → fetch, clean, chunk, embed, upsert.  
  - Always required for `get_chunks`.  

- **Query (optional)**  
  - Defines the focus of retrieval within the topic.  
  - If provided → embed the query and search inside the topic.  
  - If omitted → the topic string itself is used as the query.  

- **Return** → list of dicts:
```json
{
  "topic": "Linear algebra",
  "chunk_id": "...",
  "text": "chunk text...",
  "score": 0.61,
  "tokens": 256,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "source": {
    "kind": "wikipedia",
    "url": "https://en.wikipedia.org/wiki/Linear_algebra#Vector_spaces",
    "title": "Linear algebra",
    "section": "Vector spaces"
  }
}
```

---

## Scripts
- **`smoke.py`** — verifies end-to-end pipeline: embeddings → Wikipedia fetch → Milvus search. Run after setup to make sure everything works.  
- **`ingest_wikipedia.py`** — manually index a topic. Useful before running `sanity_check.py`.  
- **`sanity_check.py`** — checks retrieval on a topic that is already indexed (e.g., `Linear algebra` with query `eigenvalues`).  

---

## Source Choice for MVP
Retrieval uses **Wikipedia only**.  
This is the best decision for an MVP: online fetching, cleaning, and chunking from large OERs or PDFs is heavy and error-prone. Wikipedia provides broad coverage and consistent structure, making it a reliable source for demonstrating the pipeline. Other sources can be added later if needed, but are deliberately out of scope for the MVP.

---

## Usage Notes
- Retrieval is on-demand: the first query ingests, subsequent queries reuse the index.  
- Chunks are sized to match the embedding model’s limit.
- All chunks carry citation metadata (`url`, `title`, `section`) and can be surfaced directly in downstream outputs.  