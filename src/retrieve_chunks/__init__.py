"""
RAG retrieve chunks: hybrid dense + BM25 + RRF fusion.
"""
from retrieve_chunks.config import configure
from retrieve_chunks.embed import embed_text, embed_texts
from retrieve_chunks.query import query_chunks

__all__ = ["query_chunks", "embed_text", "embed_texts", "configure"]
