"""Entry point: runs the query API server."""
import time
import uuid

import structlog
import uvicorn
from fastapi import FastAPI, Request

from config import COLLECTION_NAME, PORT
from logging_config import configure_logging
from query import QueryRequest, query_chunks

configure_logging()

app = FastAPI(
    title="RAG Query Chunks",
    description="Hybrid retrieval: dense + BM25 + RRF fusion",
    version="1.0.0",
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log request path, method, status, and duration."""
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    structlog.get_logger().info(
        "request",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration_ms, 2),
    )
    return response


@app.get("/health")
def health():
    """Health check for load balancers / k8s probes."""
    return {"status": "ok"}


@app.post("/query")
def post_query(req: QueryRequest):
    """POST /query with JSON body: {"query": "...", "k": 10, "collection": "rag_dev"}"""
    collection = req.collection or COLLECTION_NAME
    return query_chunks(req.query, k=req.k, collection_name=collection)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
