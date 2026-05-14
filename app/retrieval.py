"""
Hybrid chunk retrieval: dense (vector) + BM25 + RRF fusion over Qdrant.
"""
from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter
from rank_bm25 import BM25Okapi

from app.access import RagUser, build_qdrant_access_filter
from app.config import (
    get_env,
    TOP_K_DENSE,
    RRF_K,
)
from app.http.embed import embed_text
from app.logging_config import logger
from app.qdrant.client import create_async_client, resolve_connection_params
from app.request_context import bind_request_context

_TOKEN_RE = re.compile(r"\b\w+\b")
LexicalRetriever = Callable[[str, int], Awaitable[list[dict]]]


def _payload_source(payload: dict) -> str:
    """Ingest uses ``source``; some corpora use ``source_file``."""
    return (payload.get("source_file") or payload.get("source") or "") if payload else ""


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase, alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


async def _search_dense(
    client: AsyncQdrantClient,
    query: str,
    collection_name: str,
    k: int,
    *,
    request_id: str,
    session_id: str,
    trace_id: str | None = None,
    query_vector: list[float] | None = None,
    query_filter: Filter | None = None,
    conversation_id: str | None = None,
) -> list[dict]:
    """Dense vector search via Qdrant. Pass ``query_vector`` to skip re-embedding the query.

    ``query_filter`` is forwarded to :meth:`AsyncQdrantClient.query_points` and is the
    sole hook for access control on the dense leg. Pass ``None`` for "no filter"
    (e.g. admin bypass).
    ``conversation_id`` is forwarded to :func:`embed_text` when a dense embed is needed."""
    if query_vector is not None:
        vector = query_vector
    else:
        vector = await embed_text(
            query,
            request_id=request_id,
            session_id=session_id,
            trace_id=trace_id,
            conversation_id=conversation_id,
        )
    response = await client.query_points(
        collection_name=collection_name,
        query=vector,
        limit=k,
        query_filter=query_filter,
    )
    hits: list[dict] = []
    for rank, hit in enumerate(response.points, start=1):
        payload = hit.payload or {}
        meta = {k: v for k, v in payload.items() if k != "text"}
        hits.append({
            "chunk_id": str(hit.id),
            "text": payload.get("text", ""),
            "source": _payload_source(payload),
            "metadata": meta,
            "dense_rank": rank,
            "dense_score": hit.score,
        })
    return hits


def _search_bm25(
    query: str,
    hits: list[dict],
) -> None:
    """BM25 over hits. Adds bm25_rank and bm25_score to each hit in place."""
    if not hits:
        return
    corpus = [h["text"] for h in hits]
    tokenized_corpus = [_tokenize(t) for t in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    indexed = [(i, scores[i]) for i in range(len(hits))]
    indexed.sort(key=lambda x: x[1], reverse=True)
    for bm25_rank, (idx, _) in enumerate(indexed, start=1):
        hits[idx]["bm25_rank"] = bm25_rank
        hits[idx]["bm25_score"] = float(scores[idx])


def _chunks_for_log(chunks: list[dict], *, text_max: int = 240) -> str:
    """JSON for logging: rank, ids, scores, source, truncated text."""
    slim: list[dict] = []
    for c in chunks:
        text = c.get("text") or ""
        if len(text) > text_max:
            text = text[:text_max] + "…"
        slim.append({
            "rank": c.get("rank"),
            "chunk_id": c.get("chunk_id"),
            "score": c.get("score"),
            "source": c.get("source"),
            "text": text,
        })
    return json.dumps(slim, ensure_ascii=False)


def _qdrant_collection_name(collection_name: str) -> str:
    """``collection_name`` + ``_`` + ``ENV`` from ``.env``."""
    e = get_env().strip()
    return f"{collection_name}_{e}" if e else collection_name


def _fuse_rrf(
    dense_hits: list[dict],
    bm25_hits: list[dict],
    k_final: int,
    rrf_k: int = RRF_K,
) -> list[dict]:
    """Reciprocal Rank Fusion: merge dense + BM25 by chunk_id."""
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    def add_list(hits: list[dict], rank_key: str) -> None:
        for h in hits:
            cid = h.get("chunk_id", "")
            if not cid:
                continue
            rank = h.get(rank_key, 0)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            if cid not in doc_map:
                doc_map[cid] = dict(h)
            doc_map[cid]["rrf_score"] = rrf_scores[cid]

    add_list(dense_hits, "dense_rank")
    add_list(bm25_hits, "bm25_rank")

    merged = list(doc_map.values())
    merged.sort(key=lambda x: float(x.get("rrf_score", 0.0)), reverse=True)
    return merged[:k_final]


def _prepare_bm25_hits(
    query: str,
    dense_hits: list[dict],
    lexical_hits: list[dict] | None = None,
) -> list[dict]:
    """
    Build lexical-ranked hits for fusion.

    If ``lexical_hits`` is omitted, use BM25-over-dense fallback (not true hybrid).
    If provided, expects each hit to carry ``chunk_id`` and either:
    - ``bm25_rank``, or
    - ``bm25_score`` (sorted descending then ranked).
    """
    if lexical_hits is None:
        fallback = [dict(h) for h in dense_hits]
        _search_bm25(query, fallback)
        return fallback

    out = [dict(h) for h in lexical_hits if h.get("chunk_id")]
    if not out:
        return []

    if all(h.get("bm25_rank") for h in out):
        out.sort(key=lambda x: int(x.get("bm25_rank", 0)))
        return out

    out.sort(key=lambda x: float(x.get("bm25_score", 0.0)), reverse=True)
    for rank, h in enumerate(out, start=1):
        h["bm25_rank"] = rank
        h["bm25_score"] = float(h.get("bm25_score", 0.0))
    return out


def _fusion_pack_and_log_line(
    query: str,
    dense_hits: list[dict],
    lexical_hits: list[dict] | None,
    k: int,
    rrf_k: int,
) -> tuple[list[dict], str]:
    """RRF fusion + ranked rows + log JSON (CPU-bound; one ``to_thread``)."""
    bm25_hits = _prepare_bm25_hits(query, dense_hits, lexical_hits)
    merged = _fuse_rrf(dense_hits, bm25_hits, k_final=k, rrf_k=rrf_k)
    out = [
        {
            "rank": rank,
            "chunk_id": h.get("chunk_id", ""),
            "score": h.get("rrf_score", 0.0),
            "text": h.get("text", ""),
            "source": h.get("source_file") or h.get("source", ""),
            "metadata": h.get("metadata", {}),
            "scores": {
                "rrf_score": h.get("rrf_score", 0.0),
                "dense_score": h.get("dense_score", 0.0),
                "bm25_score": h.get("bm25_score", 0.0),
            },
        }
        for rank, h in enumerate(merged, start=1)
    ]
    return out, _chunks_for_log(out)


async def query_chunks(
    query: str,
    collection_name: str,
    k: int = 10,
    *,
    request_id: str,
    session_id: str,
    trace_id: str | None = None,
    top_k_dense: int = TOP_K_DENSE,
    rrf_k: int = RRF_K,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    client: AsyncQdrantClient | None = None,
    query_vector: list[float] | None = None,
    qdrant_limit_override: int | None = None,
    lexical_retriever: LexicalRetriever | None = None,
    user: RagUser | None = None,
    conversation_id: str | None = None,
) -> list[dict]:
    """
    Hybrid retrieval: dense + lexical + RRF fusion.

    Args:
        query: Search query text.
        collection_name: Qdrant collection **base** name; resolved name is
            ``{collection_name}_{ENV}`` using ``ENV`` from ``.env`` (``dev`` / ``qa`` / ``prod``, etc.).
            Use ``ENV=`` empty in ``.env`` for no suffix.
        k: Number of chunks to return after RRF.
        request_id: Embedding ``X-Request-Id`` (required, non-empty).
        session_id: Embedding ``X-Session-Id`` (required, non-empty).
        top_k_dense: Default dense recall limit (before ``qdrant_limit_override``).
        rrf_k: RRF constant ``k`` in ``1 / (k + rank)``.
        qdrant_url: Qdrant URL override (else ``QDRANT_URL`` from ``.env``).
        qdrant_api_key: Qdrant API key override (else ``QDRANT_API_KEY`` from ``.env``).
        client: Use this ``AsyncQdrantClient`` (caller manages lifecycle; skips creating one).
        query_vector: If set, use for dense search and do not call the embedding API for this query.
        qdrant_limit_override: If set, Qdrant dense ``limit`` is ``max(top_k_dense, override)``.
        lexical_retriever: Optional independent lexical retriever ``(query, k) -> list[dict]``.
            Each returned hit should include ``chunk_id`` and either ``bm25_rank`` or
            ``bm25_score``. When omitted, BM25 is computed only on dense candidates
            (fallback mode, not full-corpus lexical retrieval).
        user: Per-request identity (see :class:`app.access.RagUser`). When set and
            non-admin, an access ``Filter`` is built via
            :func:`app.access.build_qdrant_access_filter` and applied to the dense
            leg only. The BM25 fallback runs over the (already-filtered) dense pool,
            so it cascades for free; an out-of-process ``lexical_retriever`` is the
            caller's responsibility to filter symmetrically.
        conversation_id: Optional thread id forwarded to the embedding API JSON body when
            non-empty (same contract as chat/rerank gateways).

    1. Dense: embed query (unless ``query_vector``) → Qdrant vector search
    2. Lexical: use ``lexical_retriever`` results, or BM25-over-dense fallback
    3. RRF: fuse both rankings, return top k
    """
    with bind_request_context(
        request_id,
        session_id,
        trace_id=trace_id,
        user_id=user.id if user else None,
        conversation_id=(conversation_id or "").strip() or None,
    ):
        qdrant_filter = build_qdrant_access_filter(user)
        # `should_count` is the visible knob for "did we actually filter, and how
        # many dimensions did we union?" It's None for admin (no-op) and 0 if a
        # non-admin user somehow has zero dimensions (the deny-everything sentinel).
        if qdrant_filter is None:
            access_filter_should_count: int | None = None
        else:
            access_filter_should_count = len(qdrant_filter.should or [])

        async def _run(ac: AsyncQdrantClient) -> list[dict]:
            coll = _qdrant_collection_name(collection_name)
            dense_limit = (
                max(top_k_dense, qdrant_limit_override)
                if qdrant_limit_override is not None
                else top_k_dense
            )
            logger.info(
                "query_chunks start collection=%s k=%s dense_limit=%s cached_vec=%s "
                "access_filter_applied=%s access_filter_should_count=%s",
                coll,
                k,
                dense_limit,
                query_vector is not None,
                qdrant_filter is not None,
                access_filter_should_count if access_filter_should_count is not None else "-",
                extra={
                    "access_filter_applied": qdrant_filter is not None,
                    "access_filter_should_count": (
                        access_filter_should_count
                        if access_filter_should_count is not None
                        else 0
                    ),
                },
            )

            dense_hits = await _search_dense(
                ac,
                query,
                coll,
                k=dense_limit,
                request_id=request_id,
                session_id=session_id,
                trace_id=trace_id,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                conversation_id=conversation_id,
            )
            if not dense_hits:
                logger.info(
                    "query_chunks done dense_hits=0 returned=0 access_filter_applied=%s",
                    qdrant_filter is not None,
                )
                return []

            lexical_hits: list[dict] | None = None
            if lexical_retriever is not None:
                lexical_hits = await lexical_retriever(query, dense_limit)

            out, log_out = await asyncio.to_thread(
                _fusion_pack_and_log_line,
                query,
                dense_hits,
                lexical_hits,
                k,
                rrf_k,
            )
            # BM25 always participates: either from ``lexical_retriever`` hits or
            # BM25-over-dense fallback (``lexical_hits is None``). Log pool size, not len(None).
            if lexical_hits is not None:
                lexical_n = len(lexical_hits)
                lexical_mode = "retriever"
            else:
                lexical_n = len(dense_hits)
                lexical_mode = "bm25_on_dense"
            logger.info(
                "query_chunks ok dense_candidates=%s lexical_candidates=%s lexical_mode=%s returned=%s out=%s",
                len(dense_hits),
                lexical_n,
                lexical_mode,
                len(out),
                log_out,
            )
            return out

        if client is not None:
            return await _run(client)

        u, key = resolve_connection_params(qdrant_url, qdrant_api_key)
        ac = create_async_client(u, key)
        try:
            return await _run(ac)
        finally:
            await ac.close()
