# layer-rag-query

RAG hybrid retrieval: dense (vector) + BM25 + RRF fusion. Library package.

## Installation

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

**From GitHub Actions artifacts:** After pushing to `main`, the build workflow produces `dist/` artifacts. Download from the Actions tab → latest run → Artifacts. Then `pip install dist/layer_rag_query-1.0.0.tar.gz` (exact filename may vary; use the artifact from your run).

## Configuration

Create `.env` in the project root (loaded automatically on `import app`) or export the same variables in your shell. Copy the template and edit:

```bash
cp .env.example .env
```

Importing `app` raises `ValueError` if any required variable below is missing from the environment. Empty values are allowed for `QDRANT_API_KEY` and `EMBEDDING_INTERNAL_KEY`; gated embedding APIs need a real `EMBEDDING_INTERNAL_KEY` (sent as `X-Internal-Key`, same as curl). You must pass non-empty `request_id` and `session_id` to `embed_text` / `query_chunks` for `X-Request-Id` and `X-Session-Id` (no auto-UUIDs).

| Variable | Description |
|----------|-------------|
| `QDRANT_URL` | Qdrant HTTP URL |
| `QDRANT_API_KEY` | Qdrant API key (empty for local / no auth) |
| `EMBEDDING_URL` | Embedding API base URL (`/v1/embeddings` is appended) |
| `EMBEDDING_INTERNAL_KEY` | Sent as `X-Internal-Key` when non-empty |
| `EMBEDDING_MODEL` | Model id in the request body |
| `VECTOR_SIZE` | Expected embedding length (must match the model) |
| `ENV` | Deploy/stage suffix for Qdrant: ``dev`` / ``qa`` / ``prod`` — ``query_chunks`` uses ``{collection_name}_{ENV}``. Use empty value for no suffix |
| `TOP_K_DENSE` | Dense recall count before RRF |
| `RRF_K` | RRF constant `k` |

Required keys match `REQUIRED_ENV_VARS` in `app/config.py` and `.env.example`.

Set `ENV` in `.env` to `dev`, `qa`, or `prod` (or export it). The second argument to `query_chunks` is the collection **base** name; Qdrant resolves `taixing_knowledge_dev` when base is `taixing_knowledge` and `ENV=dev`. There is no runtime `configure()` — change `.env` or shell exports and restart the process.

## Logging

On `import app`, logs go to **stderr** as **JSON lines** (`ts`, `request_id`, `session_id`, `method`, `path`, `status`, `message`, …) with [America/New_York](https://docs.python.org/3/library/zoneinfo.html) timestamps; request context is set during `embed_text` / `query_chunks` / RAG / MCP tool calls (`method`/`path`/`status` stay `-` unless you use `bind_http_context` or `extra=` on the log call). Optional **Grafana Loki**: install the project (`tb-loki-central-logger` is a dependency) and set `GRAFANA_CLOUD_*` as in [.env.example](.env.example) (same pattern as [layer-gateway-embed logging](https://github.com/taixingbi/layer-gateway-embed-v1/blob/main/app/logging_config.py), using [tb-loki-central-logger](https://github.com/taixingbi/layer-loki-central-logger)).

## curl smoke tests

Load variables from `.env` (or substitute literals). Use these to verify each dependency before running Python.

**Embedding API** — same paths and headers as `app/http/embed.py` (`/v1/embeddings`):

```bash
set -a && source .env && set +a   # or: export EMBEDDING_URL=... etc.

curl -sS -X POST "${EMBEDDING_URL}/v1/embeddings" \
  -H "X-Internal-Key: ${EMBEDDING_INTERNAL_KEY}" \
  -H "X-Request-Id: request_id_1" \
  -H "X-Session-Id: session_id_1" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"${EMBEDDING_MODEL}\", \"input\": \"hello world\"}"
```

**Inference / chat** — OpenAI-compatible `POST /v1/chat/completions` (see `INFERENCE_URL` / `INFERENCE_MODEL` in `.env`):

```bash
set -a && source .env && set +a

curl -sS -X POST "${INFERENCE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"${INFERENCE_MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"where is jersey city\"}], \"max_tokens\": 50}"
```

**OpenAPI docs** (if the inference server exposes them):

```bash
curl -sS -o /dev/null -w "%{http_code}\n" "${INFERENCE_URL}/docs"
```

**Qdrant** — list collections (optional `-H "api-key: ..."` if `QDRANT_API_KEY` is set):

```bash
set -a && source .env && set +a

curl -sS "${QDRANT_URL}/collections"
```

## Usage

```python
import asyncio
from app import embed_text, query_chunks
from qdrant_client import AsyncQdrantClient

async def main():
    # request_id / session_id are required (X-Request-Id / X-Session-Id on the embedding API)
    chunks = await query_chunks(
        "who is taixing's visa status",
        "taixing_knowledge",
        k=5,
        request_id="request_id_1",
        session_id="session_id_1",
    )

    # Pass your own AsyncQdrantClient (or omit ``client=`` to open one per call)
    async with AsyncQdrantClient(url="...", api_key="...") as client:
        chunks = await query_chunks(
            "query",
            "my_collection",
            k=5,
            request_id="request_id_1",
            session_id="session_id_1",
            client=client,
        )

    vector = await embed_text("some text", request_id="r1", session_id="s1")

asyncio.run(main())
```

## MCP (FastMCP)

Optional [FastMCP](https://gofastmcp.com) server over **stdio** (e.g. Cursor): tools `retrieve_chunks`, `embed_text`, and `answer_from_inference` (RAG + `INFERENCE_URL` chat completion).

```bash
uv pip install -e ".[mcp]"
source .venv/bin/activate
fastmcp run main.py:mcp --transport http --port 8000
```

On **HTTP** transport, **MCP** clients use `http://127.0.0.1:8000/mcp` . The same process also serves **`POST http://127.0.0.1:8000/v1/rag/query`** (JSON body; response `{"answer":"...","citations":[{"cite_id":1,"chunk_id":"...","source":"...","text":"..."},...]}` ) for plain `curl` scripts.

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "what is taixing visa",
    "collection_base": "taixing_knowledge",
    "request_id": "req-abc123",
    "session_id": "ses-xyz789",
    "k": 5,
    "k_max": 40
  }'
```

Response: `answer` is model text (with inline `[n]` citations); `citations` lists only passages actually cited in `answer` (`cite_id`, `chunk_id`, `source`, `text`). Optional body field: `"max_tokens": 512` .

**Cursor** (`.cursor/mcp.json` or global MCP settings): point the server at the repo root so `.env` resolves; use your venv’s `python` if needed:

```json
{
  "mcpServers": {
    "layer-rag-query": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["/absolute/path/to/layer-rag-query-v1/main.py"]
    }
  }
}
```

CLI: `fastmcp run main.py:mcp`

## RAG + inference (chat API)

End-to-end: **hybrid retrieval** (`query_chunks` in `app/retrieval.py`; Qdrant client setup in `app/qdrant/client.py`) → **prompt** → OpenAI-compatible **`POST /v1/chat/completions`** (e.g. Qwen on port 30080).

```bash
# Or set INFERENCE_URL / INFERENCE_MODEL in `.env` (see `.env.example`)
python -m app.rag_answer "where is jersey city" -c taixing_knowledge -k 5
```

Same flow as: retrieve grounded passages, join them as context, then call your stack’s chat endpoint with `messages` (see `app/rag_answer.py`). Inspect the OpenAPI UI at `http://<host>:30080/docs` for extra fields (temperature, etc.) if you extend the script.

## test
curl -X POST http://localhost:8000/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "what is taixing visa status",
    "collection_base": "taixing_knowledge",
    "request_id": "request_id_1",
    "session_id": "session_id_1",
    "k": 5
  }'

## eva
python eva/test.py -i eva/dataset/dataset-gold-test-1.0.0.json -o eva/result/dataset-gold-test-1.0.0.json

python eva/metric.py -i eva/result/dataset-gold-test-1.0.0.json -o eva/result/dataset-gold-test-eva-1.0.0.json