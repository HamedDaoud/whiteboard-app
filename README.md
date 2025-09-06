# Whiteboard App

Minimal RAG bootstrap with **Python 3.11** and **Milvus Standalone**. Includes a smoke test to verify embeddings → Wikipedia fetch → Milvus search.

## Prerequisites
- **Python 3.11**
  - Windows: install from python.org and verify with `py -3.11 -V`
  - macOS: `brew install python@3.11` (or python.org) and verify with `python3.11 -V`
- **Docker Desktop** (WSL2 backend on Windows) — running
- **Git**

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

python scripts\smoke.py   # expect: ALL GOOD ✅
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

python scripts/smoke.py   # expect: ALL GOOD ✅
```

## Project Layout
```
whiteboard-app/
  docker-compose.yml    # Milvus (standalone) + deps
  requirements.txt
  .env.example
  scripts/
    smoke.py            # end-to-end sanity test
  src/whiteboard/       # future retrieval modules
```
## Notes
- Always activate the venv before running commands.
- If `requirements.txt` changes, run `pip install -r requirements.txt` again.
- Milvus status: `docker compose ps` (look for **healthy**, port **19530** exposed).