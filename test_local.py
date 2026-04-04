"""Local smoke test: `.env` + `uv pip install -e .` or `pip install -e .`, then `python test_local.py`."""
from app import query_chunks
from app.config import get_env

COLL = "taixing_knowledge"
REQUEST_ID = "request_id_1"
SESSION_ID = "session_id_1"

_env = get_env().strip()
_resolved = f"{COLL}_{_env}" if _env else COLL
print(f"Qdrant collection: {_resolved!r} (ENV={_env!r})")

chunks = query_chunks(
    "who is taixing's visa status",
    COLL,
    k=5,
    request_id=REQUEST_ID,
    session_id=SESSION_ID,
)
print(f"retrieved {len(chunks)} chunk(s)")
for c in chunks:
    preview = (c.get("text") or "")[:120].replace("\n", " ")
    print(f"  rank={c.get('rank')} score={c.get('score', 0):.4f} text={preview!r}...")
