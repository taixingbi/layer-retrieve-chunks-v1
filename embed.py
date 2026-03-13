"""Generate embeddings via local v1/embeddings API."""
import httpx

from config import EMBEDDING_MODEL, EMBEDDING_URL, VECTOR_SIZE

_http_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _http_client
    if _http_client is None:
        _http_client = httpx.Client(timeout=60.0)
    return _http_client


def embed_text(text: str) -> list[float]:
    """Embed a single text."""
    return embed_texts([text])[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts via local v1/embeddings API. Reuses HTTP client and batches when possible."""
    if not texts:
        return []

    url = f"{EMBEDDING_URL.rstrip('/')}/v1/embeddings"
    model = EMBEDDING_MODEL or "BAAI/bge-m3"
    client = _get_client()

    payload = {"model": model, "input": texts if len(texts) > 1 else texts[0]}
    resp = client.post(url, json=payload)
    if resp.status_code == 200:
        data = resp.json()
        embeddings = [e["embedding"] for e in sorted(data["data"], key=lambda x: x["index"])]
    else:
        embeddings = []
        for text in texts:
            r = client.post(url, json={"model": model, "input": text})
            r.raise_for_status()
            embeddings.append(r.json()["data"][0]["embedding"])

    for emb in embeddings:
        if len(emb) != VECTOR_SIZE:
            raise ValueError(
                f"Embedding dim {len(emb)} != VECTOR_SIZE {VECTOR_SIZE}. "
                "Set VECTOR_SIZE in .env to match your embedding model."
            )
    return embeddings
