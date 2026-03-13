"""
Hybrid query chunks: dense (vector) + BM25 + RRF fusion.
"""
import re

from pydantic import BaseModel
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from config import COLLECTION_NAME, QDRANT_API_KEY, QDRANT_URL, TOP_K_DENSE, RRF_K
from embed import embed_text

_TOKEN_RE = re.compile(r"\b\w+\b")


class QueryRequest(BaseModel):
    query: str
    k: int = 10
    collection: str | None = None


_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY or None,
            check_compatibility=False,
        )
    return _client


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase, alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


def _search_dense(
    client: QdrantClient,
    query: str,
    collection_name: str,
    k: int,
) -> list[dict]:
    """Dense vector search via Qdrant."""
    vector = embed_text(query)
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
    k: int = 10,
    collection_name: str = COLLECTION_NAME,
    *,
    top_k_dense: int = TOP_K_DENSE,
    rrf_k: int = RRF_K,
) -> list[dict]:
    """
    Hybrid retrieval: dense + BM25 + RRF fusion.
    1. Dense: embed query → Qdrant vector search → top_k_dense
    2. BM25: rank those hits with BM25
    3. RRF: fuse both rankings, return top k
    """
    client = _get_client()

    # Run dense search
    dense_hits = _search_dense(client, query, collection_name, k=top_k_dense)
    if not dense_hits:
        return []

    # BM25 over dense candidates (adds bm25_rank, bm25_score to each)
    _search_bm25(query, dense_hits)

    # RRF fuse: each doc contributes from dense_rank and bm25_rank
    merged = _fuse_rrf(dense_hits, dense_hits, k_final=k, rrf_k=rrf_k)

    return [
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
