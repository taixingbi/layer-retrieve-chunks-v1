#!/usr/bin/env python3
"""
RAG: ``query_chunks`` → ranked passages → ``/v1/chat/completions`` (OpenAI-compatible).

Run from repo root (requires ``.env``). Defaults match a local vLLM/LMDeploy-style server:

  INFERENCE_URL=http://192.168.86.179:30080 \\
  INFERENCE_MODEL=Qwen/Qwen2.5-7B-Instruct \\
  python -m app.rag_answer "where is jersey city" -c taixing_knowledge

Embedding tracing for retrieval uses ``RAG_REQUEST_ID`` / ``RAG_SESSION_ID`` if set,
else random UUIDs per run.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid

import httpx

from app.config import get_inference_max_tokens, get_inference_model, get_inference_url
from app.logging_config import logger
from app.query import query_chunks
from app.request_context import bind_request_context

_NOT_FOUND_REPLY = "NOT_FOUND"

_chat_http: httpx.Client | None = None


def _chat_http_client() -> httpx.Client:
    global _chat_http
    if _chat_http is None:
        _chat_http = httpx.Client()
    return _chat_http


def _build_numbered_context(
    chunks: list[dict],
    max_chars: int = 12_000,
) -> tuple[str, list[dict]]:
    """
    Build context with passage numbers [1], [2], … for citations.

    Returns (context string, citation list: cite_id, chunk_id, source, text full passage).
    """
    parts: list[str] = []
    citations: list[dict] = []
    size = 0
    cite_id = 0
    for c in chunks:
        text = (c.get("text") or "").strip()
        if not text:
            continue
        src = (c.get("source") or "").strip() or "(unknown source)"
        cite_id += 1
        block = f"[{cite_id}] {src}\n{text}"
        if size + len(block) + 2 > max_chars:
            break
        parts.append(block)
        citations.append(
            {
                "cite_id": cite_id,
                "chunk_id": c.get("chunk_id", ""),
                "source": src,
                "text": text,
            }
        )
        size += len(block) + 2
    return "\n\n".join(parts), citations


def _citations_used_in_answer(answer: str, citations: list[dict]) -> list[dict]:
    """
    Keep only passages the model cited via [n] in the answer (n must exist in ``citations``).

    Ignores bracket numbers that are not valid ``cite_id``s (e.g. accidental matches).
    Order follows first appearance in ``answer``.
    """
    if not citations or not answer:
        return []
    by_id = {c["cite_id"]: c for c in citations}
    used: list[dict] = []
    seen: set[int] = set()
    for m in re.finditer(r"\[(\d+)\]", answer):
        cid = int(m.group(1))
        if cid in by_id and cid not in seen:
            seen.add(cid)
            used.append(by_id[cid])
    return used


def chat_complete(
    *,
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    timeout: float = 120.0,
) -> str:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    client = _chat_http_client()
    r = client.post(
        url,
        json={"model": model, "messages": messages, "max_tokens": max_tokens},
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    try:
        text = (data["choices"][0]["message"]["content"] or "").strip()
        logger.info(
            "chat_complete ok url=%s model=%s reply_chars=%s",
            url,
            model,
            len(text),
        )
        return text
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected chat response shape: {data!r}") from e


def _answer_needs_more_context(answer: str) -> bool:
    """True when we should retrieve more chunks (empty reply or explicit NOT_FOUND)."""
    if not answer:
        return True
    return answer.strip() == _NOT_FOUND_REPLY


def complete_rag_answer(
    question: str,
    collection_base: str,
    request_id: str,
    session_id: str,
    *,
    k: int = 5,
    k_max: int = 40,
    max_tokens: int | None = None,
) -> tuple[str, list[dict]]:
    """
    ``query_chunks`` → numbered context → ``POST .../v1/chat/completions``.
    Uses ``get_inference_url`` / ``get_inference_model`` / ``get_inference_max_tokens`` (from ``.env`` via ``app.config``).

    Returns ``(answer, citations)`` where ``citations`` lists only passages the model
    referenced with ``[n]`` in ``answer`` (each item: ``cite_id``, ``chunk_id``, ``source``, ``text``).

    If the model returns an empty string or exactly ``NOT_FOUND``, retries with larger ``k``
    (doubling capped by ``k_max``) until a substantive answer or ``k`` can no longer increase.
    """
    if max_tokens is None:
        max_tokens = get_inference_max_tokens()
    infer_base = get_inference_url()
    model = get_inference_model()

    if k < 1:
        raise ValueError("k must be at least 1.")
    if k_max < k:
        raise ValueError("k_max must be >= k.")

    system_msg = {
        "role": "system",
        "content": (
            "You are a precise assistant. Answer ONLY from the provided context. "
            "Each passage in the context starts with [n] SOURCE then the text; when a sentence "
            "uses facts from passage n, end that sentence (or clause) with a citation like [n]. "
            "You may cite multiple passages when needed, e.g. [1][2]. "
            "If the answer is not present in the context, reply with exactly: "
            f"\"{_NOT_FOUND_REPLY}\" and nothing else."
        ),
    }

    with bind_request_context(request_id, session_id):
        logger.info(
            "complete_rag_answer start collection_base=%s k=%s k_max=%s",
            collection_base,
            k,
            k_max,
        )
        current_k = k
        last_answer = ""
        last_citations: list[dict] = []
        while True:
            chunks = query_chunks(
                question,
                collection_base,
                k=current_k,
                request_id=request_id,
                session_id=session_id,
            )
            if not chunks:
                raise ValueError("No chunks retrieved for this query.")

            context, last_citations = _build_numbered_context(chunks)
            messages = [
                system_msg,
                {
                    "role": "user",
                    "content": (
                        f"Context:\n---\n{context}\n---\n\nQuestion: {question}"
                    ),
                },
            ]
            last_answer = chat_complete(
                base_url=infer_base,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )
            if not _answer_needs_more_context(last_answer):
                logger.info("complete_rag_answer done k_used=%s", current_k)
                return last_answer, _citations_used_in_answer(last_answer, last_citations)

            next_k = min(current_k * 2, k_max)
            if next_k <= current_k:
                logger.info(
                    "complete_rag_answer done needs_more_context k_used=%s",
                    current_k,
                )
                return last_answer, _citations_used_in_answer(last_answer, last_citations)
            logger.info(
                "complete_rag_answer retry larger k %s -> %s",
                current_k,
                next_k,
            )
            current_k = next_k


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Answer a question using hybrid retrieval + inference chat API.",
    )
    p.add_argument("question", help="User question")
    p.add_argument(
        "--collection",
        "-c",
        default="taixing_knowledge",
        help="Qdrant collection base (suffix from ENV in .env)",
    )
    p.add_argument("-k", type=int, default=5, help="Initial chunks to retrieve")
    p.add_argument(
        "--k-max",
        type=int,
        default=40,
        help="Upper bound on k when retrying after empty or NOT_FOUND (double k until this cap)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=get_inference_max_tokens(),
    )
    args = p.parse_args(argv)

    rid = os.environ.get("RAG_REQUEST_ID") or str(uuid.uuid4())
    sid = os.environ.get("RAG_SESSION_ID") or str(uuid.uuid4())

    try:
        answer, citations = complete_rag_answer(
            args.question,
            args.collection,
            rid,
            sid,
            k=args.k,
            k_max=args.k_max,
            max_tokens=args.max_tokens,
        )
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    except httpx.HTTPStatusError as e:
        print(e.response.text, file=sys.stderr)
        return 2
    out = {"answer": answer, "citations": citations}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

