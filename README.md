# retrieve-chunks

RAG hybrid retrieval: dense (vector) + BM25 + RRF fusion. Library package.

## Installation

```bash
# From GitHub
pip install git+https://github.com/YOUR_ORG/layer-retrieve-chunks-v1-logging.git@main

# From source (clone first)
pip install .

# Editable (development)
pip install -e .
```

**From GitHub Actions artifacts:** After pushing to `main`, the build workflow produces `dist/` artifacts. Download from the Actions tab → latest run → Artifacts. Then `pip install dist/retrieve_chunks-1.0.0.tar.gz`.

## Configuration

Create `.env` or set environment variables:

| Variable         | Description                    |
|------------------|--------------------------------|
| `QDRANT_URL`     | Qdrant URL (default: local)    |
| `QDRANT_API_KEY` | API key for Qdrant Cloud       |
| `EMBEDDING_URL`  | Local embedding API (default: :8001) |
| `EMBEDDING_MODEL`| Model name (default: BAAI/bge-m3)   |
| `VECTOR_SIZE`    | Embedding dim (default: 1024)  |
| `COLLECTION_NAME`| Qdrant collection base (default: rag_dev) |
| `ENV`           | Suffix: actual collection = COLLECTION_NAME + "_" + ENV |
| `TOP_K_DENSE`    | Dense recall size before RRF (default: 20) |
| `RRF_K`          | RRF constant (default: 60)    |

## Usage

```python
from retrieve_chunks import query_chunks, embed_text, configure

# Hybrid retrieval (uses env / .env)
chunks = query_chunks("who is taixing's visa status", k=5)

# Override Qdrant per call
chunks = query_chunks("query", k=5, qdrant_url="http://localhost:6333")

# Configure once before any calls. Collection accessed is collection_name + "_" + env.
configure(
    qdrant_url="http://localhost:6333",
    embedding_url="http://localhost:8001",
    embedding_model="BAAI/bge-m3",
    collection_name="taixing_knowledge",
    env="dev"
)
# Uses collection "taixing_knowledge_dev"
chunks = query_chunks("query", k=5)

# Pass your own QdrantClient
from qdrant_client import QdrantClient
client = QdrantClient(url="...", api_key="...")
chunks = query_chunks("query", k=5, client=client)

# Embedding only
vector = embed_text("some text")
```

## Build

**Local:**
```bash
pip install build
python -m build
```
Output: `dist/retrieve_chunks-1.0.0.tar.gz` and `dist/retrieve_chunks-1.0.0-*.whl`

**GitHub:** Pushes to `main` trigger the build workflow (`.github/workflows/build.yml`). Artifacts are uploaded to each run.
