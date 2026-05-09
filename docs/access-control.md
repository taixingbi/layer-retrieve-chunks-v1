# Per-request access control

`POST /v1/rag/query` filters Qdrant retrieval against a per-request identity carried
in HTTP headers. The filter runs on the **dense leg** of [`query_chunks`](../app/retrieval.py)
and cascades to the BM25 fallback (which today re-ranks the already-filtered dense pool).

## Wire contract

### Request headers

| Header | Required | Parsing | Default if missing or empty |
|--------|----------|---------|------------------------------|
| `X-User-Id` | no | trimmed string | `"-"` |
| `X-User-Roles` | no | comma-separated; per-item `strip`; drop empties; preserve case | `["anyuser"]` |
| `X-User-Groups` | no | comma-separated; per-item `strip`; drop empties; preserve case | `[]` |
| `X-User-Teams` | no | comma-separated; per-item `strip`; drop empties; preserve case | `[]` |

The four fields **must not** appear in the JSON body. Sending any of `user_id`,
`user_roles`, `user_groups`, or `user_teams` in the body returns **400** (consistent
with the existing correlation-header rule in [smoke-tests.md](smoke-tests.md#correlation-headers)).

This server **trusts** these headers verbatim. Authentication (signing,
JWT verification, mTLS) is the responsibility of an upstream gateway — typically
the same one that injects `X-User-*` after authn. If you do not have such a
gateway, do not expose this endpoint to untrusted callers.

### Response headers

On `200` (JSON or SSE) the server echoes `X-User-Id` alongside `X-Request-Id` /
`X-Session-Id` / `X-Trace-Id`. SSE additionally includes `user_id` in the `meta`
event payload (see [streaming.md](streaming.md#event-types)).

## Semantics

| Identity | Filter applied | Public chunks (`access.roles ∋ "anyuser"`) | Tagged chunks (overlap on any dimension) | Untagged chunks (no `payload.access`) |
|----------|----------------|--------------------------------------------|------------------------------------------|---------------------------------------|
| `admin` role (case-insensitive) | none (bypass) | yes | yes | yes |
| Authenticated non-admin | should-overlap on `roles` OR `groups` OR `teams` | only if explicitly tagged with one of the user's values | yes (when at least one dimension overlaps) | no (deny by default) |
| Anonymous (no `X-User-*` headers) | `roles=["anyuser"]`, no groups, no teams | yes | only if a chunk's `access.roles` contains `"anyuser"` | no |

**Match rule:** *ANY-OVERLAP across dimensions.* The user matches a chunk when at
least one of `user.roles ∩ access.roles`, `user.groups ∩ access.groups`, or
`user.teams ∩ access.teams` is non-empty. Empty user dimensions contribute zero
`should` clauses (they don't widen visibility).

**Deny by default for untagged chunks:** the filter is encoded with `MatchAny`
against `payload.access.{roles,groups,teams}`. `MatchAny` cannot match when the
field is absent, so chunks without an `access` payload are returned only to admins.

**Public chunks:** to make a chunk visible to anonymous callers, tag it with
`access.roles ∋ "anyuser"`. The pipeline's anonymous default
(`X-User-Roles` = `["anyuser"]`) intersects that set.

## Qdrant filter shape

Built by [`build_qdrant_access_filter`](../app/access.py) and forwarded to
`AsyncQdrantClient.query_points(..., query_filter=...)`. Example for
`X-User-Roles: hr,recruiter`, `X-User-Groups: engineering`,
`X-User-Teams: rag-platform`:

```python
Filter(should=[
    FieldCondition(key="access.roles",  match=MatchAny(any=["hr", "recruiter"])),
    FieldCondition(key="access.groups", match=MatchAny(any=["engineering"])),
    FieldCondition(key="access.teams",  match=MatchAny(any=["rag-platform"])),
])
```

For an `admin` user the filter is `None` (no-op). For a non-admin user with zero
populated dimensions (defensive — should be unreachable because the parser
applies the `anyuser` default), the filter is a deny-everything sentinel:

```python
Filter(must=[FieldCondition(key="__deny_all__", match=MatchAny(any=[]))])
```

## Example payload

This is the chunk shape the filter operates on (only the relevant `access` block
is shown; everything else is opaque to the filter):

```json
{
  "id": "c5d60ac7-00f8-51df-b22f-9c2e531b4f47",
  "payload": {
    "text": "Q: What is Taixing Bi's current role?\nA: AI Infrastructure Engineer",
    "access": {
      "roles":  ["admin", "hr", "recruiter", "engineer"],
      "groups": ["engineering"],
      "teams":  ["rag-platform"]
    }
  }
}
```

For the example user above (`hr,recruiter` / `engineering` / `rag-platform`) the
chunk matches on **all three** dimensions — it is returned. An anonymous request
(`anyuser` only) does **not** match: `"anyuser"` is absent from `access.roles`,
so this chunk is private.

## Examples

Authenticated user with three populated dimensions:

```bash
curl -N -sS -X POST 'http://127.0.0.1:8000/v1/rag/query' \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -H "X-Request-Id: req-stream-1" \
  -H "X-Session-Id: ses-stream-1" \
  -H "X-Trace-Id: trace-stream-1" \
  -H "X-User-Id: taixing" \
  -H "X-User-Roles: hr,recruiter" \
  -H "X-User-Groups: engineering" \
  -H "X-User-Teams: rag-platform" \
  -d '{
    "question": "what is taixing visa",
    "collection_base": "taixing_knowledge",
    "k": 5,
    "k_max": 50
  }'
```

Admin (no filter applied; sees everything in the collection):

```bash
curl -sS -X POST 'http://127.0.0.1:8000/v1/rag/query' \
  -H "Content-Type: application/json" \
  -H "X-User-Id: alice" \
  -H "X-User-Roles: admin" \
  -d '{"question":"...","collection_base":"taixing_knowledge","k":5,"k_max":50}'
```

Anonymous (only chunks tagged `access.roles ∋ "anyuser"` are visible):

```bash
curl -sS -X POST 'http://127.0.0.1:8000/v1/rag/query' \
  -H "Content-Type: application/json" \
  -d '{"question":"...","collection_base":"taixing_knowledge","k":5,"k_max":50}'
```

## Observability

`query_chunks start` log lines now carry two structured fields you can pivot on
(see [log-json-schema.md](log-json-schema.md#optional-extension-fields-extra)):

- `access_filter_applied` — `true` for non-admin requests, `false` for admins.
- `access_filter_should_count` — number of `should` clauses in the filter
  (1, 2, or 3 depending on which user dimensions are populated; `0` for admin).

`complete_rag_answer start` lines log the user identity inline (`user={"id":..., "n_roles":..., "n_groups":..., "n_teams":..., "admin":...}`) and emit `user_roles`,
`user_groups`, `user_teams` as structured fields. Every line on the request also
carries the `user_id` base field (default `"-"`).

## Out of scope

- **Sparse / BM25-in-Qdrant filtering.** Today the BM25 leg runs in-process over
  the (already access-filtered) dense candidate set, so the filter cascades for
  free. If a future change moves BM25 into Qdrant as a sparse `query_points`
  call, the same `query_filter` argument must be threaded into that call too.
- **Filter symmetry for an external `lexical_retriever` callback.** The HTTP
  path doesn't pass one; if a programmatic caller of `query_chunks` provides a
  `lexical_retriever`, that callback is responsible for honoring the user's
  identity (the `user` argument is available in the same call).
- **Schema validation of `payload.access` at ingest.** This module enforces what
  is already in the index. Sanitizing `access` shape (string-typed lists, no
  nulls, lower-case role names if you want a canonical store) is an ingest-side
  concern.
- **Per-collection / per-tenant overrides.** Single-tenant filter for now.
- **Authentication.** This endpoint trusts the `X-User-*` headers verbatim;
  authn (token verification, mTLS, signing) belongs in a gateway in front.
