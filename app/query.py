"""
Hybrid query chunks: dense (vector) + BM25 + RRF fusion.
"""
import json
import re

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from app.config import (
    get_env,
    get_qdrant_api_key,
    get_qdrant_url,
    TOP_K_DENSE,
    RRF_K,
)
from app.embed import embed_text
from app.logging_config import logger
from app.request_context import bind_request_context

_TOKEN_RE = re.compile(r"\b\w+\b")

_client: QdrantClient | None = None


def _make_client(url: str | None = None, api_key: str | None = None) -> QdrantClient:
    """Create QdrantClient from url/api_key or config."""
    u = url if url is not None else get_qdrant_url()
    k = api_key if api_key is not None else get_qdrant_api_key()
    return QdrantClient(
        url=u,
        api_key=k or None,
        check_compatibility=False,
    )


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = _make_client()
    return _client


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase, alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


def _search_dense(
    client: QdrantClient,
    query: str,
    collection_name: str,
    k: int,
    *,
    request_id: str,
    session_id: str,
    query_vector: list[float] | None = None,
) -> list[dict]:
    """Dense vector search via Qdrant. Pass ``query_vector`` to skip re-embedding the query."""
    if query_vector is not None:
        vector = query_vector
    else:
        vector = embed_text(query, request_id=request_id, session_id=session_id)
    response = client.query_points(
        collection_name=collection_name,
        query=vector,
        limit=k,
    )
    hits = []
    for rank, hit in enumerate(response.points, start=1):
        payload = hit.payload or {}
        meta = {key: val for key, val in payload.items() if key != "text"}
        hits.append({
            "chunk_id": str(hit.id),
            "text": payload.get("text", ""),
            "source": payload.get("source_file", ""),
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


def query_chunks(
    query: str,
    collection_name: str,
    k: int = 10,
    *,
    request_id: str,
    session_id: str,
    top_k_dense: int = TOP_K_DENSE,
    rrf_k: int = RRF_K,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    client: QdrantClient | None = None,
    query_vector: list[float] | None = None,
    qdrant_limit_override: int | None = None,
) -> list[dict]:
    """
    Hybrid retrieval: dense + BM25 + RRF fusion.

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
        client: Use this ``QdrantClient`` (skips URL / api_key overrides).
        query_vector: If set, use for dense search and do not call the embedding API for this query.
        qdrant_limit_override: If set, Qdrant dense ``limit`` is ``max(top_k_dense, override)``.

    1. Dense: embed query (unless ``query_vector``) → Qdrant vector search
    2. BM25: rank those hits with BM25
    3. RRF: fuse both rankings, return top k
    """
    with bind_request_context(request_id, session_id):
        if client is None:
            if qdrant_url is not None or qdrant_api_key is not None:
                client = _make_client(url=qdrant_url, api_key=qdrant_api_key)
            else:
                client = _get_client()

        coll = _qdrant_collection_name(collection_name)
        dense_limit = (
            max(top_k_dense, qdrant_limit_override)
            if qdrant_limit_override is not None
            else top_k_dense
        )
        logger.info(
            "query_chunks start collection=%s k=%s dense_limit=%s cached_vec=%s",
            coll,
            k,
            dense_limit,
            query_vector is not None,
        )

        # Run dense search
        dense_hits = _search_dense(
            client,
            query,
            coll,
            k=dense_limit,
            request_id=request_id,
            session_id=session_id,
            query_vector=query_vector,
        )
        if not dense_hits:
            logger.info("query_chunks done dense_hits=0 returned=0")
            return []

        # BM25 over dense candidates (adds bm25_rank, bm25_score to each)
        _search_bm25(query, dense_hits)

        # RRF fuse: each doc contributes from dense_rank and bm25_rank
        merged = _fuse_rrf(dense_hits, dense_hits, k_final=k, rrf_k=rrf_k)

        out = [
            {
                "rank": rank,
                "chunk_id": h.get("chunk_id", ""),
                "score": h.get("rrf_score", 0.0),
                "text": h.get("text", ""),
                "source": h.get("source_file", h.get("source", "")),
                "metadata": h.get("metadata", {}),
                "scores": {
                    "rrf_score": h.get("rrf_score", 0.0),
                    "dense_score": h.get("dense_score", 0.0),
                    "bm25_score": h.get("bm25_score", 0.0),
                },
            }
            for rank, h in enumerate(merged, start=1)
        ]
        logger.info(
            "query_chunks ok dense_candidates=%s returned=%s out=%s",
            len(dense_hits),
            len(out),
            _chunks_for_log(out),
        )
        return out
