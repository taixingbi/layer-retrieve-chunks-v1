# Log JSON schema (design)

Structured logs for **layer-rag-query** are emitted by the stdlib logger `layer_rag.query`, configured in [`app/logging_config.py`](../app/logging_config.py). This document describes the **on-the-wire JSON** shape (one UTF-8 JSON object per line).

## Goals

- **Machine-parseable**: single-line JSON suitable for `jq`, log agents, or downstream indexing.
- **Correlation**: tie log lines to embedding / RAG / HTTP work via `request_id`, `session_id`, and optional `trace_id` when context is set (`app.request_context`).
- **HTTP hints**: optional `method`, `path`, `status` for ASGI routes when context or `extra=` supplies them.
- **No noise**: omit `error` when there is no exception (no `"error": null`).

## Sinks

| Sink | When |
|------|------|
| **stderr** | Always (INFO and above for the configured handler). |

## Base record (every line)

All keys below are **always present** on normal log lines.

| Field | Type | Meaning |
|-------|------|---------|
| `ts` | string | ISO-8601 timestamp in **`America/New_York`** (from `record.created`). |
| `level` | string | Python log level name (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). |
| `request_id` | string | From request context, or `"-"` if unset. On `/v1/rag/query`, from `X-Request-Id` when sent, otherwise a server-generated UUID for that request. |
| `session_id` | string | From request context when `request_id` is set; otherwise `"-"` (avoids orphan session ids). On `/v1/rag/query`, from `X-Session-Id` when sent, otherwise a server-generated UUID for that request. |
| `trace_id` | string | From request context when set, else `"-"`. On `/v1/rag/query`, sourced from the optional `X-Trace-Id` request header; forwarded to the embedding API as `X-Trace-Id` when present. |
| `method` | string | HTTP method from context, or `"-"`. |
| `path` | string | HTTP path from context, or `"-"`. |
| `status` | string | HTTP status from context, or from `logger.info(..., extra={"status": "200"})`, or `"-"`. |
| `message` | string | Human-oriented log text (`record.getMessage()`). |

## Optional: `error`

| Field | Type | When present |
|-------|------|----------------|
| `error` | string | Only when the log call includes exception info (`exc_info=True` or inside `except` with `logger.exception(...)`). Value is the formatted traceback string. |

## Optional: extension fields (`extra=`)

If the `LogRecord` has any of these attributes (via `logger.info(..., extra={...})`), they are **copied onto the JSON object** as top-level keys. They are omitted when not supplied.

Defined allowlist in code (`_EXTRA_JSON_FIELDS`):

- `duration_ms` (often total RAG wall time; mirrors `latency_total_ms` on `complete_rag_answer done` lines)
- `latency_total_ms`, `latency_embed_ms`, `latency_retrieve_ms`, `latency_chunk_rerank_ms`, `latency_chat_ms`, `latency_follow_up_chat_ms`, `latency_follow_up_rerank_ms`
- `backend`
- `gpu`
- `reason`
- `upstream_status`
- `error_type`
- `error_message`
- `missing`
- `follow_up_empty_reason` (stable code when `follow_up_questions` resolves to `[]`; see `docs/follow-up-questions.md`)
- `follow_up_raw_reply` — full assistant content from the follow-up generation chat call (newlines escaped, not truncated). Present only on `follow_up_questions_ok` lines.
- `follow_up_candidates_full` — array of all parsed/de-duped candidate questions before rerank.
- `follow_up_candidates_count` — `len(follow_up_candidates_full)`.
- `follow_up_ranked` — array of final questions returned to the client (rerank top-N).
- `follow_up_ranked_count` — `len(follow_up_ranked)`.

To add new structured fields for dashboards or alerts, extend that tuple in `logging_config.py` and pass them through `extra=`.

## Example lines

**Info during retrieval (no exception, no `error` key):**

```json
{"ts": "2026-05-02T19:33:18.326075-04:00", "level": "INFO", "request_id": "req-abc123", "session_id": "ses-xyz789", "trace_id": "trace-001", "method": "POST", "path": "/v1/rag/query", "status": "200", "message": "query_chunks start collection=taixing_knowledge_dev k=50 dense_limit=50 cached_vec=True"}
```

**RAG request finished (optional latency fields on `complete_rag_answer done`):**

```json
{"ts": "2026-05-02T19:33:19.100000-04:00", "level": "INFO", "request_id": "req-abc123", "session_id": "ses-xyz789", "trace_id": "trace-001", "method": "POST", "path": "/v1/rag/query", "status": "200", "message": "complete_rag_answer done k_used=10 follow_up_questions=3 latency_total_ms=842", "duration_ms": 842, "latency_total_ms": 842, "latency_embed_ms": 120, "latency_retrieve_ms": 90, "latency_chunk_rerank_ms": 200, "latency_chat_ms": 350, "latency_follow_up_chat_ms": 60, "latency_follow_up_rerank_ms": 22}
```

**With exception (includes `error`):**

```json
{"ts": "2026-05-02T12:00:00.000000-05:00", "level": "ERROR", "request_id": "-", "session_id": "-", "trace_id": "-", "method": "-", "path": "-", "status": "-", "message": "upstream failed", "error": "Traceback (most recent call last):\n  ..."}
```

## Related code

- Formatter and filter: [`app/logging_config.py`](../app/logging_config.py) — `_JsonFormatter`, `_RequestContextFilter`, `setup_logging`.
- Context setters: [`app/request_context.py`](../app/request_context.py).
