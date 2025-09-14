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

### Script Usage Examples

> Run these from the repo root with your venv activated and Milvus running.

**1) End-to-End Smoke Test**

```bash
python scripts/smoke.py
# expect: ALL GOOD ✅
```

**2) Ingest a Topic (Wikipedia → clean → chunk → embed → index)**

```bash
python scripts/ingest_wikipedia.py "Linear algebra"
```

Force re-ingestion (refresh the stored chunks):

```bash
python scripts/ingest_wikipedia.py "Linear algebra" --force
```

**3) Sanity Check (topic must already be indexed)**

```bash
python scripts/sanity_check.py
# checks retrieval on "Linear algebra" with query "eigenvalues"
```

---

## Source Choice for MVP

Retrieval uses **Wikipedia only**.  
This is the best decision for an MVP: online fetching, cleaning, and chunking from large OERs or PDFs is heavy and error-prone. Wikipedia provides broad coverage and consistent structure, making it a reliable source for demonstrating the pipeline. Other sources can be added later if needed, but are deliberately out of scope for the MVP.

---

## Usage Notes

- Retrieval is on-demand: the first query ingests, subsequent queries reuse the index.
- Chunks are sized to match the embedding model’s limit.
- All chunks carry citation metadata (`url`, `title`, `section`) and can be surfaced directly in downstream outputs.

## Content Generation

The `whiteboard-app` includes a content generation module that creates educational lessons and quizzes based on retrieved Wikipedia chunks. This module, located in `src/whiteboard/content_generator.py`, uses the Hugging Face Inference API to generate content and stores results in a Supabase database for persistence.

### Prerequisites for Content Generation

- **Hugging Face API Token**: Obtain a token from [Hugging Face](https://huggingface.co/docs/inference-api) and add it to your `.env` file as `HF_API_TOKEN`.
- **Supabase Configuration**: Set up a Supabase project and add `SUPABASE_URL` (https://zxwhnkbwgcloqlkpjzok.supabase.co) and `SUPABASE_KEY` to your `.env` file.
- **Dependencies**: Ensure all requirements are installed via:
  ```bash
  pip install -r requirements.txt
  ```

### Using the Content Generator

The content generator creates a lesson (300-500 words) and a quiz (3 multiple-choice questions and 1 open-ended question) for a given topic. It uses the `get_chunks` function from `retrieval.py` to fetch relevant Wikipedia chunks, processes them with the Hugging Face model (`mistralai/Mixtral-8x7B-Instruct-v0.1`), and saves the output to Supabase.

#### Running the Content Generator

Use the `test_content_generator.py` script to generate and test content for a topic. Example:

```bash
python scripts/test_content_generator.py "Artificial Intelligence"
```

This will:

1. Fetch Wikipedia chunks for the topic.
2. Generate a lesson with an introduction, key points, an example, and sources.
3. Create a quiz based on the lesson.
4. Save the results to Supabase.
5. Print the generated lesson, quiz, and verify storage in Supabase.

**Expected Output**: A formatted lesson, quiz questions with answers, and confirmation of Supabase storage.

#### Key Functions

- **`generate_content_for_topic(topic, query=None, k=6)`**: Main function to generate content. Takes a topic (required), an optional query to focus retrieval, and `k` (number of chunks to retrieve, default 6). Returns a `GeneratedContent` object with topic, lesson, quiz, and retrieved chunks.
- **`fetch_lessons()`**: Retrieves all saved lessons from Supabase, ordered by creation date.
- **`save_content(content)`**: Saves the generated content to the Supabase `lessons` table.

#### Example Generated Content

For topic "Artificial Intelligence":

- **Lesson**: Includes an introduction, 3-5 key points, a practical example (e.g., self-driving cars for narrow AI), and cited sources.
- **Quiz**: Three multiple-choice questions and one open-ended question, e.g., "Explain the difference between general and narrow AI."

### Troubleshooting

- **Hugging Face API Issues**:
  - **Rate Limit (503)**: Wait and retry, or check your API token's limits.
  - **Authentication (401)**: Verify `HF_API_TOKEN` in `.env`.
  - **Model Errors**: Ensure the model (`mistralai/Mixtral-8x7B-Instruct-v0.1`) is supported. See [Hugging Face Docs](https://huggingface.co/docs/inference-api).
- **Supabase Issues**:
  - Verify `SUPABASE_URL` and `SUPABASE_KEY` in `.env`.
  - Check table permissions and schema in your Supabase project.
- **Logs**: Check logs for detailed errors:
  ```bash
  python scripts/test_content_generator.py "Your Topic" | tee log.txt
  ```

### Notes

- The content generator relies on the retrieval pipeline (`get_chunks`). Ensure Milvus is running (`docker compose up -d`) and the topic is indexed or will be fetched on-demand.
- Generated content is stored in the Supabase `lessons` table with fields: `topic`, `lesson`, `quiz`, `retrieved_chunks`, and `created_at`.
- For testing, run `scripts/smoke.py` first to verify the retrieval pipeline, then use `test_content_generator.py` for content generation.
