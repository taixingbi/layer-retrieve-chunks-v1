# Smoke tests (curl)

Reference list of every HTTP endpoint exposed by **layer-rag-query** plus the upstream services it calls. Use these to verify each dependency end-to-end without writing Python.

All snippets assume a `.env` at the repo root (see [README.md](../README.md#configuration)). Load it once per shell:

```bash
set -a && source .env && set +a
```

When the FastMCP HTTP server is running locally, the base URL is `http://127.0.0.1:8000` (start it with `fastmcp run app/main.py:mcp --transport http --host 0.0.0.0 --port 8000`).

## Liveness / readiness

No `request_id` / `session_id` required for either probe.

```bash
curl -sS http://127.0.0.1:8000/health
```

Expected: `200 {"status":"ok"}`.

```bash
curl -sS -o /dev/stdout -w "\nHTTP %{http_code}\n" http://127.0.0.1:8000/ready
```

Expected when Qdrant is reachable: `200 {"status":"ready"}`. When Qdrant is unreachable / mis-configured: `503 {"status":"not_ready","detail":"<ExceptionType>"}`.

## RAG query — default response

`answer`, `citations`, `follow_up_questions`, `latency_ms`.

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "what is taixing visa",
    "collection_base": "taixing_knowledge",
    "request_id": "req-abc123",
    "session_id": "ses-xyz789",
    "k": 5,
    "k_max": 50
  }'
```

## RAG query — include retrieval_hits (debug)

Adds `retrieval_hits` to the response. Equivalent flags: `include_retrieval_hits`, `debug`, `trace_retrieval`, `return_retrieval_hits`.

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "what is taixing visa",
    "collection_base": "taixing_knowledge",
    "request_id": "req-abc123",
    "session_id": "ses-xyz789",
    "k": 5,
    "k_max": 50,
    "include_retrieval_hits": true
  }'
```

## RAG query — disable follow-ups / reranker

Single-pass evaluation; no chat for follow-ups, no cross-encoder rerank.

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "what is taixing visa",
    "collection_base": "taixing_knowledge",
    "request_id": "req-abc123",
    "session_id": "ses-xyz789",
    "k": 5,
    "k_max": 50,
    "use_reranker": false,
    "include_follow_up_questions": false,
    "expand_on_not_found": false
  }'
```

## RAG query — tune follow-ups

`follow_up_candidates` ∈ `[3, 12]`; `follow_up_final` ∈ `[1, 8]` and **must be ≤ `follow_up_candidates`** (422 otherwise).

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "what is taixing visa",
    "collection_base": "taixing_knowledge",
    "request_id": "req-abc123",
    "session_id": "ses-xyz789",
    "follow_up_candidates": 10,
    "follow_up_final": 5
  }'
```

## Embedding API (upstream)

Same path/headers as [`app/http/embed.py`](../app/http/embed.py) (`POST /v1/embeddings`).

```bash
curl -sS -X POST "${EMBEDDING_URL}/v1/embeddings" \
  -H "X-Request-Id: request_id_1" \
  -H "X-Session-Id: session_id_1" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"${EMBEDDING_MODEL}\", \"input\": \"hello world\"}"
```

## Inference / chat (upstream)

OpenAI-compatible `POST /v1/chat/completions` (used by [`app/rag_answer.py`](../app/rag_answer.py)).

```bash
curl -sS -X POST "${INFERENCE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"${INFERENCE_MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"where is jersey city\"}], \"max_tokens\": 50}"
```

OpenAPI / Swagger docs (HTTP status only; some servers do not expose `/docs`):

```bash
curl -sS -o /dev/null -w "%{http_code}\n" "${INFERENCE_URL}/docs"
```

## Rerank API (upstream)

Used by [`app/http/rerank.py`](../app/http/rerank.py) when `use_reranker=true`. Path is `POST /v1/rerank` on `RERANK_URL`.

```bash
curl -sS -X POST "${RERANK_URL}/v1/rerank" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${RERANK_MODEL}\",
    \"query\": \"what is taixing visa\",
    \"documents\": [\"Taixing has an H1B visa.\", \"Jersey City is in NJ.\", \"The capital of France is Paris.\"],
    \"top_n\": 3
  }"
```

## Qdrant (upstream)

List collections (skip `api-key` header for local / no-auth):

```bash
curl -sS "${QDRANT_URL}/collections" \
  -H "api-key: ${QDRANT_API_KEY}"
```

Inspect one collection (replace base + `ENV` to match `.env`, e.g. `taixing_knowledge_dev`):

```bash
curl -sS "${QDRANT_URL}/collections/taixing_knowledge_${ENV}" \
  -H "api-key: ${QDRANT_API_KEY}"
```

## Quick cheat sheet

| Purpose | Method | URL |
|---------|--------|-----|
| Liveness | `GET` | `http://127.0.0.1:8000/health` |
| Readiness (probes Qdrant) | `GET` | `http://127.0.0.1:8000/ready` |
| RAG answer | `POST` | `http://127.0.0.1:8000/v1/rag/query` |
| MCP transport | (MCP) | `http://127.0.0.1:8000/mcp` |
| Embedding (upstream) | `POST` | `${EMBEDDING_URL}/v1/embeddings` |
| Chat (upstream) | `POST` | `${INFERENCE_URL}/v1/chat/completions` |
| Rerank (upstream) | `POST` | `${RERANK_URL}/v1/rerank` |
| Qdrant collections | `GET` | `${QDRANT_URL}/collections` |
