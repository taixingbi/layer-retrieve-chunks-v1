# HTTP request and response schema

Contract reference for **layer-rag-query** HTTP routes. Implementation: [`app/main.py`](../app/main.py) (`AnswerFromInferenceBody`, `_answer_payload`). For runnable `curl` examples see [smoke-tests.md](smoke-tests.md).

---

## Endpoints

| Method | Path | Response | Purpose |
|--------|------|----------|---------|
| `POST` | `/v1/rag/query` | `application/json` (default) or `text/event-stream` | Full RAG answer |
| `GET` | `/health` | JSON | Liveness |
| `GET` | `/ready` | JSON | Readiness (Qdrant reachable) |

MCP tools (`retrieve_chunks`, `embed_text`, `answer_from_inference`) use the same RAG logic but different wire shapes; this document covers **HTTP only**.

---

## `POST /v1/rag/query`

### Request headers

#### Correlation (header-only)

| Header | Required | Type | Notes |
|--------|----------|------|-------|
| `Content-Type` | yes | string | `application/json` |
| `X-Request-Id` | no | string | If missing or blank, server generates a UUID for this call. |
| `X-Session-Id` | no | string | If missing or blank, server generates a UUID for this call. |
| `X-Trace-Id` | no | string | Not auto-generated. Omitted → `trace_id` is JSON `null` in the body / SSE `meta`. |

**Forbidden in JSON body:** `request_id`, `session_id`, `trace_id` → **400**.

#### Access control (header-only)

| Header | Required | Type | Default | Notes |
|--------|----------|------|---------|-------|
| `X-User-Id` | no | string | `"-"` | Echoed on **200**; in SSE `meta.user_id`. |
| `X-User-Roles` | no | comma-separated string | `["anyuser"]` | `admin` bypasses Qdrant filter. |
| `X-User-Groups` | no | comma-separated string | `[]` | |
| `X-User-Teams` | no | comma-separated string | `[]` | |

**Forbidden in JSON body:** `user_id`, `user_roles`, `user_groups`, `user_teams` → **400**.

Semantics: [access-control.md](access-control.md).

#### Streaming opt-in

| Header | Required | Notes |
|--------|----------|-------|
| `Accept` | no | Include `text/event-stream` to request SSE (OR with body `"stream": true`). |

Query-param streaming (`?stream=1`) is **not** supported.

### Request body (JSON object)

| Field | Required | Type | Default | Constraints |
|-------|----------|------|---------|-------------|
| `question` | **yes** | string | — | User question. |
| `collection_base` | **yes** | string | — | Logical collection name; resolved to `{collection_base}_{ENV}` using `ENV` from `.env` (empty `ENV` → no suffix). |
| `k` | no | integer | `5` | `≥ 1`. Initial context slice size. |
| `k_max` | no | integer | `50` | `≥ 1`, must be `≥ k`. Max chunks retrieved before rerank / widen. |
| `max_tokens` | no | integer \| null | env / server default | Main chat completion budget. |
| `expand_on_not_found` | no | boolean | `true` | When true, widen context slice on empty answer or exact `NOT_FOUND`. |
| `rerank_top_n` | no | integer \| null | env | `≥ 1` when set. Cross-encoder pool size. |
| `rerank_return_top_k` | no | integer \| null | env | `≥ 1` when set. |
| `retrieve_fallback_n` | no | integer \| null | env | `≥ 0` when set. |
| `final_context_top_k` | no | integer \| null | env | `≥ 1` when set. Cap on chunks passed to the model. |
| `use_reranker` | no | boolean | `true` | |
| `include_follow_up_questions` | no | boolean | `true` | |
| `follow_up_candidates` | no | integer | `8` | `3`–`12`. LLM candidate count before rerank. |
| `follow_up_final` | no | integer | `3` | `1`–`8`, must be `≤ follow_up_candidates`. |
| `include_retrieval_hits` | no | boolean | `false` | Include `retrieval_hits` in JSON / SSE. |
| `debug` | no | boolean | `false` | Alias for `include_retrieval_hits`. |
| `trace_retrieval` | no | boolean | `false` | Alias for `include_retrieval_hits`. |
| `return_retrieval_hits` | no | boolean | `false` | Alias for `include_retrieval_hits`. |
| `stream` | no | boolean | `false` | `true` → SSE (OR with `Accept: text/event-stream`). |
| `conversation_id` | no | string \| null | server-generated | Thread id for upstream chat/embed/rerank. Blank / omitted → `conv_<hex>`. |

#### Example request

```json
{
  "question": "What is the current US visa status of Taixing?",
  "conversation_id": "conv_rag_1",
  "collection_base": "taixing_knowledge",
  "k": 5,
  "k_max": 40
}
```

---

## JSON response (`200 application/json`)

Default mode. Correlation ids appear in the **body** and are echoed as response headers.

### Response headers (`200`)

| Header | When |
|--------|------|
| `X-Request-Id` | always |
| `X-Session-Id` | always |
| `X-Conversation-Id` | always (resolved thread id) |
| `X-User-Id` | always |
| `X-Trace-Id` | only when the request sent a non-empty `X-Trace-Id` |

### Response body

| Field | Always | Type | Description |
|-------|--------|------|-------------|
| `answer` | yes | string | Model text with inline `[n]` citation markers. |
| `citations` | yes | array | Passages **referenced** in `answer` via `[n]` (see [Citation](#citation)). |
| `follow_up_questions` | yes | array of string | Suggested next questions; `[]` when disabled or on failure. |
| `latency_ms` | yes | object | Per-phase timings in milliseconds (see [Latency](#latency_ms)). |
| `request_id` | yes | string | Same as `X-Request-Id`. |
| `session_id` | yes | string | Same as `X-Session-Id`. |
| `trace_id` | yes | string \| null | Same as `X-Trace-Id` when sent; else `null`. |
| `conversation_id` | yes | string | Resolved thread id (same as `X-Conversation-Id`). |
| `retrieval_hits` | no | array | Present when any retrieval-hits flag is true (see [Retrieval hit](#retrieval_hit)). |

#### Example response

```json
{
  "answer": "Taixing Bi's current US visa status is H4 EAD, and there is no visa sponsorship required [1].",
  "citations": [
    {
      "cite_id": 1,
      "chunk_id": "1607b45e-1c07-5c29-975d-bbf47ef3129c",
      "source": "personal_profile",
      "text": "Q: What is Taixing Bi's visa status / work authorization?\nA: H4 EAD. No visa sponsorship required."
    }
  ],
  "follow_up_questions": [
    "Can Taixing apply for a different type of visa?",
    "Does Taixing need to renew the H4 EAD?",
    "What are the requirements for maintaining H4 EAD status?"
  ],
  "latency_ms": {
    "total": 2794,
    "embed": 178,
    "retrieve": 55,
    "chunk_rerank": 228,
    "chat": 606,
    "follow_up_chat": 1684,
    "follow_up_rerank": 42
  },
  "request_id": "req-abc123",
  "session_id": "ses-xyz789",
  "trace_id": "trc-001",
  "conversation_id": "conv_rag_1"
}
```

### Nested types

#### Citation

| Field | Type | Description |
|-------|------|-------------|
| `cite_id` | integer | `1`-based index matching `[n]` in `answer`. |
| `chunk_id` | string | Qdrant point id. |
| `source` | string | Document / passage label. |
| `text` | string | Full cited passage. |

#### `latency_ms`

| Field | Type | Description |
|-------|------|-------------|
| `total` | integer | Wall time for the full handler. |
| `embed` | integer | Query embedding. |
| `retrieve` | integer | Hybrid Qdrant search. |
| `chunk_rerank` | integer | Cross-encoder rerank over retrieved pool. |
| `chat` | integer | Main answer generation (cumulative across widen retries). |
| `follow_up_chat` | integer | Follow-up LLM call. |
| `follow_up_rerank` | integer | Follow-up candidate rerank. |

#### Retrieval hit

Only when `include_retrieval_hits`, `debug`, `trace_retrieval`, or `return_retrieval_hits` is true. Two stages may appear: `retrieve` (RRF order) then `rerank` (cross-encoder order). No passage text.

| Field | Type | Description |
|-------|------|-------------|
| `stage` | string | `"retrieve"` or `"rerank"`. |
| `rank` | integer | 1-based rank within that stage. |
| `chunk_id` | string | |
| `source` | string | |
| `score` | number | RRF or rerank score. |

---

## Error responses (JSON mode)

Errors **before** the SSE stream opens return normal JSON with an HTTP status. The stream is never started.

| Status | Body | When |
|--------|------|------|
| `400` | `{"detail": "<string>"}` | Invalid JSON, body not an object, forbidden body keys, `k_max < k`, empty retrieval, etc. |
| `422` | `{"detail": [<pydantic errors>]}` | Body validation failure (types, ranges, `follow_up_final > follow_up_candidates`). |
| `502` | `{"detail": "<upstream text>"}` | Upstream embedding / inference / rerank HTTP error. |

`detail` for forbidden correlation keys:

```json
{
  "detail": "request_id, session_id, and trace_id must not appear in the JSON body; use X-Request-Id, X-Session-Id, and X-Trace-Id headers instead."
}
```

---

## SSE response (`200 text/event-stream`)

Opt-in via `Accept: text/event-stream` **or** `"stream": true`. Wire format and event order: [streaming.md](streaming.md).

### Response headers (`200`)

| Header | Value |
|--------|-------|
| `Content-Type` | `text/event-stream` |
| `X-Request-Id` | echoed or generated |
| `X-Session-Id` | echoed or generated |
| `X-Conversation-Id` | resolved `conversation_id` |
| `X-User-Id` | echoed |
| `X-Trace-Id` | when request sent it |
| `Cache-Control` | `no-cache` |
| `X-Accel-Buffering` | `no` |

### Frame format

```
event: <name>
data: <one-line JSON>

```

(blank line after each event)

### Events (summary)

| Event | `data` shape | Notes |
|-------|--------------|-------|
| `meta` | `request_id`, `session_id`, `trace_id`, `user_id`, `conversation_id`, `collection`, `k`, `k_max` | First frame. `trace_id` may be `null`. |
| `latency` | `phase`, `ms` | `phase` ∈ `embed`, `retrieve`, `chunk_rerank`, `chat`, `follow_up_chat`, `follow_up_rerank`, `total`. |
| `retrieval_widen` | `reason`, `prev_k`, `next_k` | Before a context widen retry (`reason` is `"not_found"`). |
| `answer_start` | `{}` | |
| `answer_delta` | `text` | Final answer only, ~48 UTF-8 chars per frame. |
| `answer_end` | `{}` | |
| `citations` | `items` | Array of [Citation](#citation). |
| `follow_up_questions` | `items` | Array of strings. |
| `retrieval_hits` | `items` | Optional; same rows as JSON `retrieval_hits`. |
| `error` | `detail` | Always followed by `done`. HTTP status stays **200**. |
| `done` | `{}` | Terminal sentinel. |

#### Example `meta` event

```
event: meta
data: {"request_id": "req-abc123", "session_id": "ses-xyz789", "trace_id": "trc-001", "user_id": "-", "conversation_id": "conv_rag_1", "collection": "taixing_knowledge", "k": 5, "k_max": 40}
```

Pre-stream validation errors (`400` / `422`) return JSON, not SSE.

---

## `GET /health`

**200**

```json
{ "status": "ok" }
```

---

## `GET /ready`

| Status | Body |
|--------|------|
| `200` | `{"status": "ready"}` |
| `503` | `{"status": "not_ready", "detail": "<ExceptionType>"}` |

---

## Upstream forwarding

On `/v1/rag/query`, the resolved `conversation_id` is forwarded in JSON bodies to downstream services when non-empty:

| Service | Path | Field |
|---------|------|-------|
| Embedding | `POST /v1/embeddings` | `conversation_id` |
| Rerank | `POST /v1/rerank` | `conversation_id` |
| Inference | `POST /v1/chat/completions` | `conversation_id` |

Correlation headers (`X-Request-Id`, `X-Session-Id`, `X-Trace-Id`) are forwarded on those calls per [`app/http/_correlation.py`](../app/http/_correlation.py).

---

## Related docs

- [smoke-tests.md](smoke-tests.md) — `curl` recipes
- [streaming.md](streaming.md) — SSE ordering, cancel, proxies
- [access-control.md](access-control.md) — `X-User-*` semantics
- [follow-up-questions.md](follow-up-questions.md) — follow-up pipeline
- [log-json-schema.md](log-json-schema.md) — stderr JSON log fields
