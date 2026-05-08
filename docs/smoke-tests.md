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

## Correlation headers

`POST /v1/rag/query` reads correlation IDs **only from headers**. Putting `request_id`, `session_id`, or `trace_id` in the JSON body returns **400**.

| Header | Required | Notes |
|--------|----------|-------|
| `X-Request-Id` | no | If missing or blank, the server generates a UUID for this call. Forwarded to the embedding API; appears as `request_id` in stderr JSON logs. |
| `X-Session-Id` | no | If missing or blank, the server generates a UUID for this call. Forwarded to the embedding API; appears as `session_id` in stderr JSON logs. |
| `X-Trace-Id` | no | When set, forwarded to embedding API as `X-Trace-Id` and appears as `trace_id` in stderr JSON logs (else `"-"`). Not auto-generated. |

## RAG query — minimal (no correlation headers)

Same as default; logs and embedding calls use the server-generated `request_id` / `session_id` for this request. On **200** responses, the same values are echoed in `X-Request-Id` and `X-Session-Id` response headers (use `curl -D -` to capture them).

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

## RAG query — default response

`answer`, `citations`, `follow_up_questions`, `latency_ms`. With explicit correlation (recommended behind gateways). On **200** responses, `X-Request-Id`, `X-Session-Id`, and `X-Trace-Id` (when sent) are echoed on the response.

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

## RAG query — include retrieval_hits (debug)

Adds `retrieval_hits` to the response. Equivalent flags: `include_retrieval_hits`, `debug`, `trace_retrieval`, `return_retrieval_hits`.

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

## RAG query — disable follow-ups / reranker

Single-pass evaluation; no chat for follow-ups, no cross-encoder rerank.

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
  -H "X-Request-Id: req-abc123" \
  -H "X-Session-Id: ses-xyz789" \
  -d '{
    "question": "what is taixing visa",
    "collection_base": "taixing_knowledge",
    "follow_up_candidates": 10,
    "follow_up_final": 5
  }'
```

## RAG query — streaming (SSE)

Opt in with **any** of: `?stream=1` query param, `Accept: text/event-stream` header, or `"stream": true` in the JSON body. Response is `text/event-stream` with `Cache-Control: no-cache` and `X-Accel-Buffering: no`. See [streaming.md](streaming.md) for the full event sequence and error semantics; behind nginx-ingress, also set `nginx.ingress.kubernetes.io/proxy-buffering: "off"` on the route.

Query-param style (most explicit on the wire):

```bash
curl -N -sS -X POST 'http://127.0.0.1:8000/v1/rag/query?stream=1' \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -H "X-Request-Id: req-stream-1" \
  -H "X-Session-Id: ses-stream-1" \
  -H "X-Trace-Id: trace-stream-1" \
  -d '{
    "question": "what is taixing visa",
    "collection_base": "taixing_knowledge",
    "k": 5,
    "k_max": 50
  }'
```

Body-flag style (handy when callers can't change the URL or `Accept` header):

```bash
curl -N -sS -X POST http://127.0.0.1:8000/v1/rag/query \
  -H "Content-Type: application/json" \
  -H "X-Request-Id: req-stream-2" \
  -H "X-Session-Id: ses-stream-2" \
  -d '{
    "question": "what is taixing visa",
    "collection_base": "taixing_knowledge",
    "k": 5,
    "k_max": 50,
    "stream": true
  }'
```

Either request expects `meta`, retrieval `latency` phases, `answer_start`, several `answer_delta` frames (final answer only; widen retries do not stream `NOT_FOUND`), `answer_end`, `citations`, `follow_up_questions`, remaining `latency` events including `total`, and `done`. If the model triggers a context widen, you will also see one or more `retrieval_widen` events before `answer_start`. `-N` disables curl's own output buffering — without it the deltas batch on stdout.

## RAG query — error cases

```bash
curl -sS -o /dev/stdout -w "\nHTTP %{http_code}\n" \
  -X POST http://127.0.0.1:8000/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question":"q","collection_base":"taixing_knowledge","request_id":"x","session_id":"y"}'
# -> HTTP 400 (request_id/session_id/trace_id must not appear in body)
```

## Embedding API (upstream)

Same path/headers as [`app/http/embed.py`](../app/http/embed.py) (`POST /v1/embeddings`). `X-Trace-Id` is forwarded only when present.

```bash
curl -sS -X POST "${EMBEDDING_URL}/v1/embeddings" \
  -H "X-Request-Id: request_id_1" \
  -H "X-Session-Id: session_id_1" \
  -H "X-Trace-Id: trace-001" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"${EMBEDDING_MODEL}\", \"input\": \"hello world\"}"
```

## Inference / chat (upstream)

OpenAI-compatible `POST /v1/chat/completions` (used by [`app/rag_answer.py`](../app/rag_answer.py)). Same correlation headers as embedding (`X-Trace-Id` forwarded only when set).

```bash
curl -sS -X POST "${INFERENCE_URL}/v1/chat/completions" \
  -H "X-Request-Id: request_id_1" \
  -H "X-Session-Id: session_id_1" \
  -H "X-Trace-Id: trace-001" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"${INFERENCE_MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"where is jersey city\"}], \"max_tokens\": 50}"
```

OpenAPI / Swagger docs (HTTP status only; some servers do not expose `/docs`):

```bash
curl -sS -o /dev/null -w "%{http_code}\n" "${INFERENCE_URL}/docs"
```

## Rerank API (upstream)

Used by [`app/http/rerank.py`](../app/http/rerank.py) when `use_reranker=true`. Path is `POST /v1/rerank` on `RERANK_URL`. Same correlation headers as embedding (`X-Trace-Id` forwarded only when set).

```bash
curl -sS -X POST "${RERANK_URL}/v1/rerank" \
  -H "X-Request-Id: request_id_1" \
  -H "X-Session-Id: session_id_1" \
  -H "X-Trace-Id: trace-001" \
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
| RAG answer (JSON) | `POST` | `http://127.0.0.1:8000/v1/rag/query` |
| RAG answer (SSE) | `POST` | `http://127.0.0.1:8000/v1/rag/query?stream=1` |
| MCP transport | (MCP) | `http://127.0.0.1:8000/mcp` |
| Embedding (upstream) | `POST` | `${EMBEDDING_URL}/v1/embeddings` |
| Chat (upstream) | `POST` | `${INFERENCE_URL}/v1/chat/completions` |
| Rerank (upstream) | `POST` | `${RERANK_URL}/v1/rerank` |
| Qdrant collections | `GET` | `${QDRANT_URL}/collections` |
