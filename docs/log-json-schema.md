# Log JSON schema (design)

Structured logs for **layer-rag-query** are emitted by the stdlib logger `layer_rag.query`, configured in [`app/logging_config.py`](../app/logging_config.py). This document describes the **on-the-wire JSON** shape (one UTF-8 JSON object per line).

## Goals

- **Machine-parseable**: single-line JSON suitable for `jq`, Loki, Elasticsearch, or log agents.
- **Correlation**: tie log lines to embedding / RAG / HTTP work via `request_id` and `session_id` when context is set (`app.request_context`).
- **HTTP hints**: optional `method`, `path`, `status` for ASGI routes when context or `extra=` supplies them.
- **No noise**: omit `error` when there is no exception (no `"error": null`).

## Sinks

| Sink | When |
|------|------|
| **stderr** | Always (INFO and above for the configured handler). |
| **Grafana Loki** | Optional, when `tb-loki-central-logger` can build auth from env (e.g. `GRAFANA_CLOUD_API_KEY` set). Same JSON line is pushed; Loki adds stream labels (see below). |

## Base record (every line)

All keys below are **always present** on normal log lines.

| Field | Type | Meaning |
|-------|------|---------|
| `ts` | string | ISO-8601 timestamp in **`America/New_York`** (from `record.created`). |
| `level` | string | Python log level name (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). |
| `request_id` | string | From request context, or `"-"` if unset. |
| `session_id` | string | From request context when `request_id` is set; otherwise `"-"` (avoids orphan session ids). |
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

To add new structured fields for dashboards or alerts, extend that tuple in `logging_config.py` and pass them through `extra=`.

## Loki stream labels (static)

When Loki is enabled, each push uses base labels:

| Label | Typical value |
|-------|-----------------|
| `service` | `layer-rag-query` |
| `component` | `query` |
| `env` | From `ENV` env var, default `dev` |
| `version` | Package `__version__` |

Per-line label **`logger`** is set to the Python logger name (e.g. `layer_rag.query`).

## Environment notes

- **Loki**: See README / Grafana docs for `GRAFANA_CLOUD_*` variables used by `basic_auth_from_env()`.
- **Proxy**: If Loki push fails through an HTTP proxy with errors such as 403, set `LOKI_IGNORE_SYSTEM_PROXY=1` (truthy) to use a client that bypasses system proxy settings.
- **LogQL by level**: now that `level` is in each JSON line, queries like `{service="layer-rag-query",component="query"} | json | __error__="" | level="ERROR"` work.

## Example lines

**Info during retrieval (no exception, no `error` key):**

```json
{"ts": "2026-05-02T19:33:18.326075-04:00", "level": "INFO", "request_id": "req-abc123", "session_id": "ses-xyz789", "method": "POST", "path": "/v1/rag/query", "status": "200", "message": "query_chunks start collection=taixing_knowledge_dev k=50 dense_limit=50 cached_vec=True"}
```

**RAG request finished (optional latency fields on `complete_rag_answer done`):**

```json
{"ts": "2026-05-02T19:33:19.100000-04:00", "level": "INFO", "request_id": "req-abc123", "session_id": "ses-xyz789", "method": "POST", "path": "/v1/rag/query", "status": "200", "message": "complete_rag_answer done k_used=10 follow_up_questions=3 latency_total_ms=842", "duration_ms": 842, "latency_total_ms": 842, "latency_embed_ms": 120, "latency_retrieve_ms": 90, "latency_chunk_rerank_ms": 200, "latency_chat_ms": 350, "latency_follow_up_chat_ms": 60, "latency_follow_up_rerank_ms": 22}
```

**With exception (includes `error`):**

```json
{"ts": "2026-05-02T12:00:00.000000-05:00", "level": "ERROR", "request_id": "-", "session_id": "-", "method": "-", "path": "-", "status": "-", "message": "upstream failed", "error": "Traceback (most recent call last):\n  ..."}
```

## Related code

- Formatter and filter: [`app/logging_config.py`](../app/logging_config.py) — `_JsonFormatter`, `_RequestContextFilter`, `setup_logging`.
- Context setters: [`app/request_context.py`](../app/request_context.py).
