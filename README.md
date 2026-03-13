# RAG Query Chunks

Hybrid retrieval: dense (vector) + BM25 + RRF fusion. HTTP POST API.

## Configuration

Create `.env`:

| Variable         | Description                    |
|------------------|--------------------------------|
| `QDRANT_URL`     | Qdrant URL (default: local)    |
| `QDRANT_API_KEY` | API key for Qdrant Cloud       |
| `EMBEDDING_URL`  | Local embedding API (default: :8001) |
| `EMBEDDING_MODEL`| Model name (default: BAAI/bge-m3)   |
| `VECTOR_SIZE`    | Embedding dim (default: 1024)  |
| `COLLECTION_NAME`| Qdrant collection (default: rag_dev) |
| `TOP_K_DENSE`    | Dense recall size before RRF (default: 20) |
| `RRF_K`          | RRF constant (default: 60)    |
| `PORT`           | Server port (default: 8000)    |

## How to start

```bash
# 1. Create venv and install deps
python3 -m venv venv
source venv/bin/activate 
pip install -r requirements.txt

# 2. Create .env (see Configuration above)

# 3. Start the server
python main.py
# or: uvicorn main:app --host 0.0.0.0 --port 8000
```

Server runs at http://localhost:8000. API docs: http://localhost:8000/docs. Health: `GET /health`.

## Usage

**POST** `/query` with JSON body:

| Field       | Type | Required | Default     |
|-------------|------|----------|-------------|
| `query`    | str  | yes      | —           |
| `k`        | int  | no       | 10          |
| `collection` | str | no       | `COLLECTION_NAME` |

Example:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "who is taixing'\''s visa status", "k": 5}'
```

Response: array of chunks with `rank`, `chunk_id`, `score`, `text`, `source`, `metadata`, `scores` (rrf_score, dense_score, bm25_score).
