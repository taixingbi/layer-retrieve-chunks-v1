"""HTTP clients for embedding, chat-completions, and rerank APIs."""

from app.http.embed import embed_text, embed_texts
from app.http.inference import chat_complete
from app.http.rerank import rerank_texts

__all__ = ["embed_text", "embed_texts", "chat_complete", "rerank_texts"]
