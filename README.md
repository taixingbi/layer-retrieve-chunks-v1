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

Importing `app` raises `ValueError` if any required variable below is missing from the environment. Empty values are allowed for `QDRANT_API_KEY`. You must pass non-empty `request_id` and `session_id` to `embed_text` / `query_chunks` for `X-Request-Id` and `X-Session-Id` (no auto-UUIDs); `trace_id` is optional and forwarded as `X-Trace-Id` when set. On `POST /v1/rag/query`, correlation is read **only from headers** (`X-Request-Id`, `X-Session-Id`, `X-Trace-Id`); missing request or session headers get fresh UUIDs for that call. Putting `request_id`, `session_id`, or `trace_id` in the JSON body returns **400**.

| Variable | Description |
|----------|-------------|
| `QDRANT_URL` | Qdrant HTTP URL |
| `QDRANT_API_KEY` | Qdrant API key (empty for local / no auth) |
| `EMBEDDING_URL` | Embedding API base URL (`/v1/embeddings` is appended) |
| `EMBEDDING_MODEL` | Model id in the request body |
| `VECTOR_SIZE` | Expected embedding length (must match the model) |
| `ENV` | Deploy/stage suffix for Qdrant: ``dev`` / ``qa`` / ``prod`` — ``query_chunks`` uses ``{collection_name}_{ENV}``. Use empty value for no suffix |
| `TOP_K_DENSE` | Dense recall count before RRF |
| `RRF_K` | RRF constant `k` |

Required keys match `REQUIRED_ENV_VARS` in `app/config.py` and `.env.example`.

Set `ENV` in `.env` to `dev`, `qa`, or `prod` (or export it). The second argument to `query_chunks` is the collection **base** name; Qdrant resolves `taixing_knowledge_dev` when base is `taixing_knowledge` and `ENV=dev`. There is no runtime `configure()` — change `.env` or shell exports and restart the process.

## Logging

On `import app`, logs go to **stderr** as **JSON lines** (`ts`, `level`, `request_id`, `session_id`, `trace_id`, `method`, `path`, `status`, `message`, …) with [America/New_York](https://docs.python.org/3/library/zoneinfo.html) timestamps; request context is set during `embed_text` / `query_chunks` / RAG / MCP tool calls (`method`/`path`/`status` stay `-` unless you use `bind_http_context` or `extra=` on the log call). `trace_id` is `"-"` unless set via `bind_request_context(..., trace_id=...)` (e.g. `X-Trace-Id` on `/v1/rag/query`).

Correlation IDs are forwarded as HTTP headers on every upstream call — embedding (`/v1/embeddings`), rerank (`/v1/rerank`), and chat (`/v1/chat/completions`) — as `X-Request-Id`, `X-Session-Id`, and (when set) `X-Trace-Id`, so logs from all four services can be stitched together by `request_id` / `session_id` / `trace_id`.

## curl smoke tests

Load variables from `.env` (or substitute literals). Use these to verify each dependency before running Python.

**Embedding API** — same paths and headers as `app/http/embed.py` (`/v1/embeddings`). `X-Trace-Id` is forwarded only when set:

```bash
set -a && source .env && set +a   # or: export EMBEDDING_URL=... etc.

curl -sS -X POST "${EMBEDDING_URL}/v1/embeddings" \
  -H "X-Request-Id: request_id_1" \
  -H "X-Session-Id: session_id_1" \
  -H "X-Trace-Id: trace-001" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"${EMBEDDING_MODEL}\", \"input\": \"hello world\"}"
```

**Inference / chat** — OpenAI-compatible `POST /v1/chat/completions` (see `INFERENCE_URL` / `INFERENCE_MODEL` in `.env`). Same correlation headers as embedding (`X-Trace-Id` forwarded only when set):

```bash
set -a && source .env && set +a

curl -sS -X POST "${INFERENCE_URL}/v1/chat/completions" \
  -H "X-Request-Id: request_id_1" \
  -H "X-Session-Id: session_id_1" \
  -H "X-Trace-Id: trace-001" \
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
    # request_id / session_id are required (X-Request-Id / X-Session-Id on the embedding API);
    # trace_id is optional (X-Trace-Id when set).
    chunks = await query_chunks(
        "who is taixing's visa status",
        "taixing_knowledge",
        k=5,
        request_id="request_id_1",
        session_id="session_id_1",
        trace_id="trace-001",
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

    vector = await embed_text(
        "some text", request_id="r1", session_id="s1", trace_id="trace-001"
    )

asyncio.run(main())
```

## MCP (FastMCP)

Optional [FastMCP](https://gofastmcp.com) server over **stdio** (e.g. Cursor): tools `retrieve_chunks`, `embed_text`, and `answer_from_inference` (RAG + `INFERENCE_URL` chat completion).

```bash
uv pip install -e ".[mcp]"
source .venv/bin/activate
fastmcp run app/main.py:mcp --transport http --host 0.0.0.0 --port 8000
```

`-m app.main` is the module-style entrypoint. In this mode, FastMCP uses the module's own startup (`mcp.run()`), so CLI transport/host/port flags are ignored.

On **HTTP** transport, **MCP** clients use `http://127.0.0.1:8000/mcp` . The same process also serves **`POST http://127.0.0.1:8000/v1/rag/query`** (JSON body; default response includes `answer`, `citations`, `follow_up_questions`, and `latency_ms` — per-phase millisecond timings) for plain `curl` scripts, plus liveness/readiness probes:

- `GET /health` — always `200 {"status":"ok"}` while the process is up (no I/O, no headers required).
- `GET /ready` — `200 {"status":"ready"}` when Qdrant responds to `get_collections`; `503 {"status":"not_ready","detail":"<error type>"}` otherwise.

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/ready
```

Correlation on `/v1/rag/query` is **header-only** (never put `request_id`, `session_id`, or `trace_id` in the JSON body — **400**). `X-Request-Id` and `X-Session-Id` are optional: if either header is missing or blank, the server generates a UUID for that call. `X-Trace-Id` is optional and is not auto-generated. On **200** responses, `X-Request-Id`, `X-Session-Id`, and `X-Trace-Id` (when sent) are echoed in response headers so clients can confirm or read server-generated IDs (`curl -D -`).

**Access control.** `/v1/rag/query` also reads `X-User-Id` / `X-User-Roles` / `X-User-Groups` / `X-User-Teams` (header-only; sending these in the body returns **400**). Roles default to `["anyuser"]` when absent so chunks tagged `access.roles ∋ "anyuser"` form the public set; `admin` bypasses filtering; chunks without `payload.access` are deny-by-default for non-admins. The match rule is **ANY-OVERLAP across dimensions** (`should` in Qdrant). `X-User-Id` is echoed on **200** alongside the correlation headers. Full semantics, payload shape, and `curl` examples: [`docs/access-control.md`](docs/access-control.md).

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "what is taixing visa",
    "collection_base": "taixing_knowledge",
    "k": 5,
    "k_max": 50
  }'
```

With explicit correlation (recommended for gateways and log stitching):

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/rag/query \
  -H "Content-Type: application/json" \
  -H "X-Request-Id: req-abc123" \
  -H "X-Session-Id: ses-xyz789" \
  -H "X-Trace-Id: trace-001" \
  -d '{
    "question": "what is taixing visa",
    "collection_base": "taixing_knowledge",
    "k": 5,
    "k_max": 50
  }'
```

**Response (default):** `answer` is model text (with inline `[n]` citations); `citations` lists only passages actually cited in `answer` (`cite_id`, `chunk_id`, `source`, `text`). `follow_up_questions` is always present (possibly `[]`): strings from a second chat call, trimmed by the reranker. `latency_ms` is always present (`total`, `embed`, `retrieve`, `chunk_rerank`, `chat`, `follow_up_chat`, `follow_up_rerank`).

**Optional body fields:** `"max_tokens": 512`, rerank controls `"rerank_top_n": 50`, `"rerank_return_top_k": 25` (must be `>= final_context_top_k` when reranking), `"retrieve_fallback_n": 8`, `"final_context_top_k": 12`, `"use_reranker": true`, `"expand_on_not_found": true`, and follow-up controls `"include_follow_up_questions": true` (default), `"follow_up_candidates": 8` (3–12), `"follow_up_final": 3` (must be `<= follow_up_candidates`). Defaults also come from `.env` (`RERANK_TOP_N`, `RERANK_RETURN_TOP_K`, `RETRIEVE_FALLBACK_N`, `FINAL_CONTEXT_TOP_K`).

**Optional `retrieval_hits` (eval / debug):** If any of these booleans is true, the response also includes `retrieval_hits`: `include_retrieval_hits`, `debug`, `trace_retrieval`, `return_retrieval_hits`. Each hit is a small object (no passage text): `stage` (`retrieve` = RRF order after hybrid fusion, `rerank` = cross-encoder order when reranking ran), `rank` (1-based within that stage), `chunk_id`, `source`, `score`. Scores are not comparable across stages (retrieve uses RRF; rerank uses the rerank API).

**Streaming (SSE).** For chat-style UIs, opt in via `Accept: text/event-stream` or `"stream": true` in the JSON body (query-param triggers like `?stream=1` are not supported). The stream is `text/event-stream`: `meta` and early `latency` phases, then optional `retrieval_widen` events (context slice retry after `NOT_FOUND` / empty — **no** streamed `NOT_FOUND` text), then `answer_start` and `answer_delta` chunks for the **final** answer only, `answer_end`, `citations`, `follow_up_questions`, remaining `latency` lines (including `total` wall time), and `done`. Errors after the first frame use `event: error` then `event: done`. Response headers match JSON mode plus `Cache-Control: no-cache` and `X-Accel-Buffering: no`. **Pause / Stop from the UI** = abort the `fetch` (use `AbortController`); the server cancels the upstream chat completion automatically and frees the GPU slot. Details and a JS / `httpx` cancel example: [`docs/streaming.md`](docs/streaming.md).

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/rag/query \
  -H "Content-Type: application/json" \
  -H "X-Request-Id: req-abc123" \
  -H "X-Session-Id: ses-xyz789" \
  -d '{
    "question": "what is taixing visa",
    "collection_base": "taixing_knowledge",
    "k": 5,
    "k_max": 50,
    "include_retrieval_hits": true
  }'
```

See also [`docs/streaming.md`](docs/streaming.md), [`docs/access-control.md`](docs/access-control.md), [`docs/smoke-tests.md`](docs/smoke-tests.md), [`docs/follow-up-questions.md`](docs/follow-up-questions.md), and [`docs/log-json-schema.md`](docs/log-json-schema.md).

**Cursor** (`.cursor/mcp.json` or global MCP settings): point the server at the repo root so `.env` resolves; use your venv’s `python` if needed:

```json
{
  "mcpServers": {
    "layer-rag-query": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["/absolute/path/to/layer-rag-query-v1/app/main.py"]
    }
  }
}
```

CLI: `fastmcp run app/main.py:mcp --transport http --host 0.0.0.0 --port 8000`

## RAG + inference (chat API)

End-to-end: **hybrid retrieval** (`query_chunks` in `app/retrieval.py`; Qdrant client setup in `app/qdrant/client.py`) → optional **rerank** (`POST /v1/rerank`) → **prompt** → OpenAI-compatible **`POST /v1/chat/completions`** (e.g. Qwen on port 30180).

```bash
# Or set INFERENCE_URL / INFERENCE_MODEL in `.env` (see `.env.example`)
python -m app.rag_answer "where is jersey city" -c taixing_knowledge -k 5
```

Useful flags: `--single-pass` (one chat, no context widen on `NOT_FOUND`), `--no-reranker`, `--no-follow-ups`, `--retrieval-hits` (print `retrieval_hits` in the JSON, same shape as HTTP when the debug flags are on), `--follow-up-candidates` / `--follow-up-final`.

Same flow as: retrieve grounded passages, join them as context, then call your stack’s chat endpoint with `messages` (see `app/rag_answer.py`). Inspect the OpenAPI UI at `http://<host>:30180/docs` for extra fields (temperature, etc.) if you extend the script.

The MCP tool `answer_from_inference` accepts the same optional booleans as the HTTP body for retrieval hits (`include_retrieval_hits`, `debug`, `trace_retrieval`, `return_retrieval_hits`).

## Evaluation

```bash
python eva/test.py -i eva/dataset/dataset-gold-test-1.0.0.json -o eva/result/dataset-gold-test-1.0.0.json

python eva/metric.py -i eva/result/dataset-gold-test-1.0.0.json -o eva/result/dataset-gold-test-eva-1.0.0.json
```