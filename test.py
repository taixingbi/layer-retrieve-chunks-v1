"""Basic tests for retrieve_chunks package."""
import sys
from pathlib import Path

# Allow running without pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import retrieve_chunks

retrieve_chunks.configure(
    env="dev",
    qdrant_url="http://192.168.86.173:6333",
    embedding_url="http://192.168.86.173:8001",
    embedding_model="BAAI/bge-m3",
    collection_name="taixing_knowledge",
)
chunks = retrieve_chunks.query_chunks("who is taixing?", k=3)
print(chunks)