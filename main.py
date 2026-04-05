"""
FastMCP server: hybrid chunk retrieval + embeddings (stdio for Cursor / Claude).

Requires `.env` at repo root (same as `import app`). Install: ``pip install -e ".[mcp]"``

Run: ``python main.py`` or ``fastmcp run main.py:mcp``
"""
from __future__ import annotations

from typing import Any

import httpx
from fastmcp import FastMCP
from pydantic import BaseModel, Field, ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse

from app import embed_text as _embed_text
from app import query_chunks as _query_chunks
from app.rag_answer import complete_rag_answer
from app.request_context import bind_http_context, bind_request_context


class AnswerFromInferenceBody(BaseModel):
    question: str
    collection_base: str
    request_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    k: int = Field(default=5, ge=1)
    k_max: int = Field(default=40, ge=1)
    max_tokens: int | None = None


def answer_from_inference_payload(body: AnswerFromInferenceBody) -> dict[str, Any]:
    """Run RAG + chat; raise ``ValueError`` or ``httpx.HTTPStatusError`` on failure."""
    if body.k_max < body.k:
        raise ValueError("k_max must be >= k")
    with bind_request_context(body.request_id, body.session_id):
        answer, citations = complete_rag_answer(
            body.question,
            body.collection_base,
            body.request_id,
            body.session_id,
            k=body.k,
            k_max=body.k_max,
            max_tokens=body.max_tokens,
        )
    return {"answer": answer, "citations": citations}


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
    with bind_request_context(request_id, session_id):
        return _query_chunks(
            query,
            collection_base,
            k=k,
            request_id=request_id,
            session_id=session_id,
        )


@mcp.tool
def embed_text(
    text: str,
    request_id: str,
    session_id: str,
) -> list[float]:
    """Embed a single string via the configured /v1/embeddings API. Returns the embedding vector."""
    with bind_request_context(request_id, session_id):
        return _embed_text(text, request_id=request_id, session_id=session_id)


@mcp.tool
def answer_from_inference(
    question: str,
    collection_base: str,
    request_id: str,
    session_id: str,
    k: int = 5,
    k_max: int = 40,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Retrieve chunks (retry with larger k up to k_max if the model returns empty or NOT_FOUND), then POST to INFERENCE_URL /v1/chat/completions. ``citations`` lists only chunks referenced via [n] in ``answer``."""
    with bind_request_context(request_id, session_id):
        answer, citations = complete_rag_answer(
            question,
            collection_base,
            request_id,
            session_id,
            k=k,
            k_max=k_max,
            max_tokens=max_tokens,
        )
    return {"answer": answer, "citations": citations}


@mcp.custom_route("/v1/rag/query", methods=["POST"])
async def answer_from_inference_http(request: Request) -> JSONResponse:
    """JSON body for ``curl`` when using FastMCP ``--transport http``."""
    method = request.method
    path = request.url.path

    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"detail": "Invalid JSON"}, status_code=400)
    try:
        body = AnswerFromInferenceBody.model_validate(data)
    except ValidationError as e:
        return JSONResponse({"detail": e.errors()}, status_code=422)
    try:
        # method/path/status for stderr + Loki JSON lines (matches ASGI access log when happy path).
        with bind_http_context(method, path, status="200"):
            out = answer_from_inference_payload(body)
    except ValueError as e:
        return JSONResponse({"detail": str(e)}, status_code=400)
    except httpx.HTTPStatusError as e:
        return JSONResponse(
            {"detail": e.response.text or str(e)},
            status_code=502,
        )
    return JSONResponse(out)


if __name__ == "__main__":
    mcp.run()
