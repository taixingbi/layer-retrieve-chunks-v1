"""
Configuration for query (Qdrant URL, API key, collection, embeddings).
Loads .env if present; values can be overridden by environment variables.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://192.168.86.173:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# Embedding (local v1/embeddings API)
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://192.168.86.173:8001")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# Query defaults
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_dev")
TOP_K_DENSE = int(os.getenv("TOP_K_DENSE", "20"))
RRF_K = int(os.getenv("RRF_K", "60"))

# Server
PORT = int(os.getenv("PORT", "8000"))

# Vector size (BAAI/bge-m3 outputs 1024)
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "1024"))
