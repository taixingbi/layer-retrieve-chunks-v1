"""
Layer RAG query: hybrid dense + BM25 + RRF retrieval and embeddings.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("layer-rag-query")
except PackageNotFoundError:
    __version__ = "0.0.0"

from app.embed import embed_text, embed_texts
from app.query import query_chunks

__all__ = ["__version__", "query_chunks", "embed_text", "embed_texts"]
