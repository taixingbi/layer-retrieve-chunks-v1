"""Entry point: runs the query API server."""
import uvicorn
from fastapi import FastAPI

from config import COLLECTION_NAME, PORT
from query import QueryRequest, query_chunks

app = FastAPI(
    title="RAG Query Chunks",
    description="Hybrid retrieval: dense + BM25 + RRF fusion",
    version="1.0.0",
)


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
