"""
FastMCP server: hybrid chunk retrieval + embeddings (stdio for Cursor / Claude).

Requires `.env` at repo root (same as `import app`). Install: ``pip install -e ".[mcp]"``

Run: ``python -m app.main`` or ``fastmcp run app/main.py:mcp``
"""
from __future__ import annotations

import uuid
from typing import Any

import httpx
from fastmcp import FastMCP
from pydantic import BaseModel, Field, ValidationError, model_validator
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.asyncio_util import run_async
from app.http.embed import embed_text as _embed_text_async
from app.logging_config import logger
from app.qdrant.client import create_async_client, resolve_connection_params
from app.rag_answer import complete_rag_answer
from app.request_context import bind_http_context
from app.retrieval import query_chunks as _query_chunks_async

_FORBIDDEN_RAG_BODY_KEYS = frozenset({"request_id", "session_id", "trace_id"})


def _correlation_from_headers(request: Request) -> tuple[str, str, str | None]:
    """Read ``X-Request-Id``, ``X-Session-Id``, ``X-Trace-Id`` (case-insensitive). Trace may be absent."""
    rid = (request.headers.get("x-request-id") or "").strip()
    sid = (request.headers.get("x-session-id") or "").strip()
    tid_raw = (request.headers.get("x-trace-id") or "").strip()
    tid: str | None = tid_raw if tid_raw else None
    return rid, sid, tid


class AnswerFromInferenceBody(BaseModel):
    question: str
    collection_base: str
    k: int = Field(default=5, ge=1)
    k_max: int = Field(default=50, ge=1)
    max_tokens: int | None = None
    expand_on_not_found: bool = True
    rerank_top_n: int | None = Field(default=None, ge=1)
    rerank_return_top_k: int | None = Field(default=None, ge=1)
    retrieve_fallback_n: int | None = Field(default=None, ge=0)
    final_context_top_k: int | None = Field(default=None, ge=1)
    use_reranker: bool = True
    include_follow_up_questions: bool = True
    follow_up_candidates: int = Field(default=8, ge=3, le=12)
    follow_up_final: int = Field(default=3, ge=1, le=8)
    include_retrieval_hits: bool = False
    debug: bool = False
    trace_retrieval: bool = False
    return_retrieval_hits: bool = False

    @model_validator(mode="after")
    def _follow_up_final_lte_candidates(self) -> AnswerFromInferenceBody:
        if self.follow_up_final > self.follow_up_candidates:
            raise ValueError("follow_up_final must be <= follow_up_candidates")
        return self

    def wants_retrieval_hits(self) -> bool:
        """Support historical debug aliases for returning retrieval_hits."""
        return bool(
            self.include_retrieval_hits
            or self.debug
            or self.trace_retrieval
            or self.return_retrieval_hits
        )


def _answer_payload(
    *,
    answer: str,
    citations: list[dict],
    follow_up_questions: list[str],
    latency_ms: dict[str, int],
    retrieval_hits: list[dict],
    include_retrieval_hits: bool,
) -> dict[str, Any]:
    """Build stable HTTP/MCP response payload (conditionally including retrieval_hits)."""
    out: dict[str, Any] = {
        "answer": answer,
        "citations": citations,
        "follow_up_questions": follow_up_questions,
        "latency_ms": latency_ms,
    }
    if include_retrieval_hits:
        out["retrieval_hits"] = retrieval_hits
    return out


async def answer_from_inference_payload_async(
    body: AnswerFromInferenceBody,
    *,
    request_id: str,
    session_id: str,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Run RAG + chat (async). Raise ``ValueError`` or ``httpx.HTTPStatusError`` on failure."""
    if body.k_max < body.k:
        raise ValueError("k_max must be >= k")
    wants_hits = body.wants_retrieval_hits()
    answer, citations, follow_up_questions, latency_ms, retrieval_hits = await complete_rag_answer(
        body.question,
        body.collection_base,
        request_id,
        session_id,
        k=body.k,
        k_max=body.k_max,
        max_tokens=body.max_tokens,
        expand_on_not_found=body.expand_on_not_found,
        rerank_top_n=body.rerank_top_n,
        rerank_return_top_k=body.rerank_return_top_k,
        retrieve_fallback_n=body.retrieve_fallback_n,
        final_context_top_k=body.final_context_top_k,
        use_reranker=body.use_reranker,
        include_follow_up_questions=body.include_follow_up_questions,
        follow_up_candidates=body.follow_up_candidates,
        follow_up_final=body.follow_up_final,
        include_retrieval_hits=wants_hits,
        trace_id=trace_id,
    )
    return _answer_payload(
        answer=answer,
        citations=citations,
        follow_up_questions=follow_up_questions,
        latency_ms=latency_ms,
        retrieval_hits=retrieval_hits,
        include_retrieval_hits=wants_hits,
    )


mcp = FastMCP(
    "layer-rag-query",
    instructions="RAG tools: Qdrant hybrid search (dense + BM25 + RRF), embeddings, and optional "
    "full answers via INFERENCE_URL /v1/chat/completions (set in .env). "
    "Pass collection base; ENV suffix comes from .env. request_id and session_id are required for retrieval embedding calls.",
)


@mcp.tool
def retrieve_chunks(
    query: str,
    collection_base: str,
    request_id: str,
    session_id: str,
    k: int = 5,
) -> list[dict]:
    """Hybrid retrieval from Qdrant. collection_base is suffixed with ENV from .env (e.g. taixing_knowledge + dev → taixing_knowledge_dev)."""
    return run_async(
        _query_chunks_async(
            query,
            collection_base,
            k=k,
            request_id=request_id,
            session_id=session_id,
        )
    )


@mcp.tool
def embed_text(
    text: str,
    request_id: str,
    session_id: str,
) -> list[float]:
    """Embed a single string via the configured /v1/embeddings API. Returns the embedding vector."""
    return run_async(_embed_text_async(text, request_id=request_id, session_id=session_id))


@mcp.tool
def answer_from_inference(
    question: str,
    collection_base: str,
    request_id: str,
    session_id: str,
    k: int = 5,
    k_max: int = 50,
    max_tokens: int | None = None,
    expand_on_not_found: bool = True,
    rerank_top_n: int | None = None,
    rerank_return_top_k: int | None = None,
    retrieve_fallback_n: int | None = None,
    final_context_top_k: int | None = None,
    use_reranker: bool = True,
    include_follow_up_questions: bool = True,
    follow_up_candidates: int = 8,
    follow_up_final: int = 3,
    include_retrieval_hits: bool = False,
    debug: bool = False,
    trace_retrieval: bool = False,
    return_retrieval_hits: bool = False,
) -> dict[str, Any]:
    """Retrieve once (pool k_max), then chat; optional slice widen on NOT_FOUND. Set expand_on_not_found false for single-pass eval."""
    if follow_up_final > follow_up_candidates:
        raise ValueError("follow_up_final must be <= follow_up_candidates")
    wants_hits = include_retrieval_hits or debug or trace_retrieval or return_retrieval_hits
    answer, citations, follow_up_questions, latency_ms, retrieval_hits = run_async(
        complete_rag_answer(
            question,
            collection_base,
            request_id,
            session_id,
            k=k,
            k_max=k_max,
            max_tokens=max_tokens,
            expand_on_not_found=expand_on_not_found,
            rerank_top_n=rerank_top_n,
            rerank_return_top_k=rerank_return_top_k,
            retrieve_fallback_n=retrieve_fallback_n,
            final_context_top_k=final_context_top_k,
            use_reranker=use_reranker,
            include_follow_up_questions=include_follow_up_questions,
            follow_up_candidates=follow_up_candidates,
            follow_up_final=follow_up_final,
            include_retrieval_hits=wants_hits,
        )
    )
    return _answer_payload(
        answer=answer,
        citations=citations,
        follow_up_questions=follow_up_questions,
        latency_ms=latency_ms,
        retrieval_hits=retrieval_hits,
        include_retrieval_hits=wants_hits,
    )


@mcp.custom_route("/v1/rag/query", methods=["POST"])
async def answer_from_inference_http(request: Request) -> JSONResponse:
    """JSON body for ``curl`` when using FastMCP ``--transport http``.

    Correlation: ``X-Request-Id``, ``X-Session-Id``, ``X-Trace-Id`` (optional). If request or
    session id headers are missing or blank, new UUIDs are generated for this call only.
    Do not send ``request_id``, ``session_id``, or ``trace_id`` in the JSON body (400).
    """
    method = request.method
    path = request.url.path

    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"detail": "Invalid JSON"}, status_code=400)
    if not isinstance(data, dict):
        return JSONResponse({"detail": "JSON body must be an object"}, status_code=400)
    if _FORBIDDEN_RAG_BODY_KEYS & data.keys():
        return JSONResponse(
            {
                "detail": (
                    "request_id, session_id, and trace_id must not appear in the JSON body; "
                    "use X-Request-Id, X-Session-Id, and X-Trace-Id headers instead."
                )
            },
            status_code=400,
        )
    request_id, session_id, trace_id = _correlation_from_headers(request)
    if not request_id:
        request_id = str(uuid.uuid4())
    if not session_id:
        session_id = str(uuid.uuid4())
    try:
        body = AnswerFromInferenceBody.model_validate(data)
    except ValidationError as e:
        return JSONResponse({"detail": e.errors()}, status_code=422)
    try:
        # method/path/status for stderr JSON lines (matches ASGI access log when happy path).
        with bind_http_context(method, path, status="200"):
            out = await answer_from_inference_payload_async(
                body,
                request_id=request_id,
                session_id=session_id,
                trace_id=trace_id,
            )
    except ValueError as e:
        return JSONResponse({"detail": str(e)}, status_code=400)
    except httpx.HTTPStatusError as e:
        return JSONResponse(
            {"detail": e.response.text or str(e)},
            status_code=502,
        )
    hdrs: dict[str, str] = {
        "X-Request-Id": request_id,
        "X-Session-Id": session_id,
    }
    if trace_id:
        hdrs["X-Trace-Id"] = trace_id
    return JSONResponse(out, headers=hdrs)


@mcp.custom_route("/health", methods=["GET"], include_in_schema=False)
async def health(request: Request) -> JSONResponse:
    """Liveness: always 200 while the process is up."""
    with bind_http_context(request.method, request.url.path, status="200"):
        return JSONResponse({"status": "ok"})


@mcp.custom_route("/ready", methods=["GET"], include_in_schema=False)
async def ready(request: Request) -> JSONResponse:
    """Readiness: 200 when Qdrant responds to ``get_collections``, else 503."""
    url, api_key = resolve_connection_params()
    client = create_async_client(url, api_key)
    try:
        try:
            await client.get_collections()
        except Exception as e:
            with bind_http_context(request.method, request.url.path, status="503"):
                logger.warning(
                    "ready probe failed",
                    extra={"error_type": type(e).__name__, "error_message": str(e)},
                )
            return JSONResponse(
                {"status": "not_ready", "detail": type(e).__name__},
                status_code=503,
            )
        with bind_http_context(request.method, request.url.path, status="200"):
            return JSONResponse({"status": "ready"})
    finally:
        await client.close()


if __name__ == "__main__":
    mcp.run()
