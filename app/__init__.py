"""
Layer RAG query: hybrid dense + BM25 + RRF retrieval and embeddings.

Public helpers ``embed_text``, ``embed_texts``, and ``query_chunks`` are **async**
(await them, or use ``app.asyncio_util.run_async`` from blocking code).
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("layer-rag-query")
except PackageNotFoundError:
    __version__ = "0.0.0"

from app.http.embed import embed_text, embed_texts
from app.retrieval import query_chunks

__all__ = ["__version__", "query_chunks", "embed_text", "embed_texts"]
