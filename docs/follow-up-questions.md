# Follow-up questions (RAG response)

After the main RAG answer and citations are produced, the pipeline can attach **`follow_up_questions`**: short suggested questions for the user to ask next. Implementation lives in [`app/rag_answer.py`](../app/rag_answer.py); HTTP/MCP wiring is in [`main.py`](../main.py).

## Response shape

`POST /v1/rag/query` and the MCP tool `answer_from_inference` return JSON with:

| Field | Type | Meaning |
|-------|------|---------|
| `answer` | string | Model reply with optional inline `[n]` citations. |
| `citations` | array | Passages actually cited in `answer` (`cite_id`, `chunk_id`, `source`, `text`). |
| `follow_up_questions` | array of strings | Always present; may be `[]` if disabled, on parse failure, or when generation fails. |

## Pipeline

1. **Main RAG** (unchanged): retrieve → optional chunk rerank → chat → citations.
2. **Context summary**: From the same chunk slice used for the final chat turn, build a compact bullet list (`source` + truncated text, ~3.5k chars max) via `_context_summary_for_followups`.
3. **Generate candidates**: Second call to `INFERENCE_URL` / `v1/chat/completions` asks the model for **only** a JSON array of strings. Count bounds: `min_gen = max(3, follow_up_candidates - 3)` through `max_gen = follow_up_candidates` (defaults: 5–8 when `follow_up_candidates == 8`).
4. **Rerank**: [`app/http/rerank.py`](../app/http/rerank.py) `rerank_texts` scores each candidate string against the **original user question** (`query` = `question`), returns all indices ordered by score, then the code keeps the first **`follow_up_final`** (default **3**).

Token budget for the generator: `min(512, max(256, max_tokens))` where `max_tokens` is the same cap used for the main RAG chat for that request.

## HTTP request body (optional fields)

| Field | Default | Constraints |
|-------|---------|-------------|
| `include_follow_up_questions` | `true` | Set `false` to skip extra chat + rerank. |
| `follow_up_candidates` | `8` | Integer in **3–12** (LLM asked for between `max(3, N-3)` and `N` questions). |
| `follow_up_final` | `3` | Integer **1–8** and **must be ≤ `follow_up_candidates`**. |

Validation errors (e.g. `follow_up_final > follow_up_candidates`) return **422** on HTTP with Pydantic detail.

## CLI (`python -m app.rag_answer`)

- `--no-follow-ups` — same as disabling follow-ups.
- `--follow-up-candidates N` — must satisfy `complete_rag_answer` validation (3–12).
- `--follow-up-final M` — must be ≤ candidates.

Printed JSON includes `follow_up_questions`.

## Fallbacks

- **JSON parse failure** after generation → `follow_up_questions: []` (warning logged).
- **Rerank HTTP / transport error** → first `follow_up_final` candidates in **generation order** (warning logged).
- **Generation exception** → `[]`.

## Cost and latency

Each enabled request adds **one extra chat completion** and **one rerank** call (documents = candidate question strings). Uses the same `INFERENCE_*` and `RERANK_*` settings as the rest of RAG.

## Related

- [README.md](../README.md) — curl example and body overview.
- [log-json-schema.md](log-json-schema.md) — structured logging for the same service.
