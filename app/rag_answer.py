#!/usr/bin/env python3
"""
RAG: ``app.retrieval.query_chunks`` → ranked passages → ``/v1/chat/completions`` (OpenAI-compatible).

Run from repo root (requires ``.env``). Defaults match a local vLLM/LMDeploy-style server:

  INFERENCE_URL=http://192.168.86.179:30080 \\
  INFERENCE_MODEL=Qwen/Qwen2.5-7B-Instruct \\
  python -m app.rag_answer "where is jersey city" -c taixing_knowledge

Embedding tracing for retrieval uses ``RAG_REQUEST_ID`` / ``RAG_SESSION_ID`` if set,
else random UUIDs per run.
"""
from __future__ import annotations

import argparse
from typing import Any
import json
import os
import re
import sys
import time
import uuid

import httpx

from app.config import (
    get_final_context_top_k,
    get_inference_max_tokens,
    get_inference_model,
    get_inference_url,
    get_rerank_model,
    get_rerank_return_top_k,
    get_rerank_top_n,
    get_rerank_url,
    get_retrieve_fallback_n,
)
from app.asyncio_util import run_async
from app.http.embed import embed_text
from app.http.inference import chat_complete
from app.http.rerank import rerank_texts
from app.logging_config import logger
from app.retrieval import query_chunks
from app.request_context import bind_request_context

_NOT_FOUND_REPLY = "NOT_FOUND"
_FOLLOW_UP_GEN_MAX_TOKENS_CAP = 512


def _elapsed_ms(since: float) -> int:
    """Wall time in milliseconds from ``time.perf_counter()`` mark ``since``."""
    return int(round((time.perf_counter() - since) * 1000))


def _preview_for_log(text: str, *, max_chars: int = 400) -> str:
    """One-line safe truncation for stderr JSON (newlines escaped)."""
    s = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
    if len(s) > max_chars:
        return s[:max_chars] + "…"
    return s


def _context_summary_for_followups(chunks: list[dict], *, max_chars: int = 3500) -> str:
    """Compact lines (source + truncated text) for follow-up generation prompts."""
    lines: list[str] = []
    size = 0
    for c in chunks:
        src = (c.get("source") or "").strip() or "(unknown)"
        text = (c.get("text") or "").strip().replace("\n", " ")
        if len(text) > 220:
            text = text[:220] + "…"
        line = f"- {src}: {text}"
        if size + len(line) + 1 > max_chars:
            break
        lines.append(line)
        size += len(line) + 1
    return "\n".join(lines) if lines else "(no context)"


def _parse_follow_up_json(raw: str) -> list[str]:
    """Parse model output into a list of non-empty question strings."""
    text = raw.strip()
    if not text:
        return []
    if "```" in text:
        low = text.lower()
        if "```json" in low:
            start = low.find("```json") + 7
            end = text.find("```", start)
            if end != -1:
                text = text[start:end].strip()
        else:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end != -1:
                text = text[start:end].strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        i0 = text.find("[")
        i1 = text.rfind("]")
        if i0 == -1 or i1 <= i0:
            return []
        try:
            data = json.loads(text[i0 : i1 + 1])
        except json.JSONDecodeError:
            return []
    if not isinstance(data, list):
        return []
    out: list[str] = []
    for x in data:
        if isinstance(x, str) and (s := x.strip()):
            out.append(s)
    return out


async def _generate_follow_up_candidates(
    *,
    question: str,
    answer: str,
    context_summary: str,
    infer_base: str,
    model: str,
    min_count: int,
    max_count: int,
    max_tokens: int,
) -> list[str]:
    """One chat call: JSON array of between ``min_count`` and ``max_count`` question strings."""
    sys = (
        "You write only valid JSON: a single array of strings. No markdown, no keys, no commentary. "
        f"Each string is one short follow-up question (under 120 characters). "
        f"Produce between {min_count} and {max_count} distinct questions. "
        "Questions must be answerable from the same knowledge domain as the context summary; "
        "they may extend or refine the user's topic. Do not repeat the original question verbatim."
    )
    user = (
        f"Original question:\n{question}\n\n"
        f"Answer that was given:\n{answer}\n\n"
        f"Context summary (retrieved passages):\n{context_summary}\n\n"
        f"Output a JSON array of {min_count} to {max_count} follow-up questions."
    )
    raw = await chat_complete(
        base_url=infer_base,
        model=model,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
    )
    candidates = _parse_follow_up_json(raw)
    if not candidates and raw.strip():
        logger.warning(
            "follow_up parse produced no questions reply_chars=%s raw_preview=%s",
            len(raw),
            _preview_for_log(raw),
        )
    return candidates


async def _rerank_follow_up_strings(
    *,
    question: str,
    candidates: list[str],
    rerank_url: str,
    rerank_model: str,
    top_n: int,
) -> list[str]:
    """Rerank candidate questions; return top ``top_n`` strings in score order."""
    if not candidates:
        return []
    n = min(top_n, len(candidates))
    try:
        rows = await rerank_texts(
            base_url=rerank_url,
            model=rerank_model,
            query=question,
            documents=candidates,
            top_n=n,
        )
    except Exception as e:
        logger.warning("follow_up rerank failed reason=%s", str(e))
        return candidates[:n]
    out: list[str] = []
    seen: set[int] = set()
    for row in rows:
        idx = row.get("index", -1)
        if not isinstance(idx, int) or idx < 0 or idx >= len(candidates) or idx in seen:
            continue
        seen.add(idx)
        out.append(candidates[idx])
        if len(out) >= n:
            break
    if len(out) < n:
        for i, s in enumerate(candidates):
            if i in seen:
                continue
            out.append(s)
            if len(out) >= n:
                break
    return out if out else candidates[:n]


async def _follow_up_questions(
    *,
    question: str,
    answer: str,
    chunks_used: list[dict],
    follow_up_candidates: int,
    follow_up_final: int,
    infer_base: str,
    model: str,
    max_tokens_main: int,
    rerank_url: str,
    rerank_model: str,
) -> tuple[list[str], int, int]:
    """Returns ``(questions, chat_ms, rerank_ms)``; times are zero when skipped or on failure."""
    if not chunks_used:
        return [], 0, 0
    min_gen = max(3, follow_up_candidates - 3)
    max_gen = follow_up_candidates
    if min_gen > max_gen:
        min_gen = max_gen
    summary = _context_summary_for_followups(chunks_used)
    gen_budget = min(_FOLLOW_UP_GEN_MAX_TOKENS_CAP, max(256, max_tokens_main))
    gen_t0 = time.perf_counter()
    try:
        candidates = await _generate_follow_up_candidates(
            question=question,
            answer=answer,
            context_summary=summary,
            infer_base=infer_base,
            model=model,
            min_count=min_gen,
            max_count=max_gen,
            max_tokens=gen_budget,
        )
    except Exception as e:
        logger.warning("follow_up generation failed reason=%s", str(e))
        return [], _elapsed_ms(gen_t0), 0
    gen_ms = _elapsed_ms(gen_t0)
    if not candidates:
        return [], gen_ms, 0
    rr_t0 = time.perf_counter()
    ranked = await _rerank_follow_up_strings(
        question=question,
        candidates=candidates,
        rerank_url=rerank_url,
        rerank_model=rerank_model,
        top_n=follow_up_final,
    )
    return ranked, gen_ms, _elapsed_ms(rr_t0)


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


def _answer_needs_more_context(answer: str) -> bool:
    """True when we should retrieve more chunks (empty reply or explicit NOT_FOUND)."""
    if not answer:
        return True
    return answer.strip() == _NOT_FOUND_REPLY


def _with_citations(answer: str, citations: list[dict]) -> tuple[str, list[dict]]:
    return answer, _citations_used_in_answer(answer, citations)


def _retrieval_hits_payload(
    chunks_full: list[dict],
    reranked: list[dict] | None,
) -> list[dict]:
    """
    Slim rows for eval/debug: RRF-ordered retrieve stage, then optional rerank stage
    (1-based ``rank`` within each ``stage``). No passage text.
    """
    hits: list[dict] = []
    for i, c in enumerate(chunks_full, start=1):
        hits.append({
            "stage": "retrieve",
            "rank": int(c.get("rank") or i),
            "chunk_id": str(c.get("chunk_id") or ""),
            "source": str(c.get("source") or ""),
            "score": float(c.get("score", 0.0)),
        })
    if reranked:
        for i, c in enumerate(reranked, start=1):
            hits.append({
                "stage": "rerank",
                "rank": int(c.get("rerank_rank") or i),
                "chunk_id": str(c.get("chunk_id") or ""),
                "source": str(c.get("source") or ""),
                "score": float(c.get("rerank_score", 0.0)),
            })
    return hits


def _merge_rerank_with_retrieve_fallback(
    reranked: list[dict],
    chunks_full: list[dict],
    *,
    fallback_n: int,
) -> list[dict]:
    """
    Rerank-ordered list first, then up to ``fallback_n`` chunks from RRF ``chunks_full`` order
    whose ``chunk_id`` was missing from the rerank output (reranker miss hedge).
    """
    if fallback_n <= 0 or not reranked:
        return [dict(c) for c in reranked]
    seen = {str(c.get("chunk_id") or "") for c in reranked}
    seen.discard("")
    out: list[dict] = [dict(c) for c in reranked]
    added = 0
    for c in chunks_full:
        if added >= fallback_n:
            break
        cid = str(c.get("chunk_id") or "")
        if not cid or cid in seen:
            continue
        out.append(dict(c))
        seen.add(cid)
        added += 1
    return out


async def _rerank_chunks(
    *,
    question: str,
    chunks: list[dict],
    rerank_url: str,
    rerank_model: str,
    rerank_return_top_k: int,
) -> list[dict]:
    if not chunks:
        return []
    rows = await rerank_texts(
        base_url=rerank_url,
        model=rerank_model,
        query=question,
        documents=[(c.get("text") or "") for c in chunks],
        top_n=min(rerank_return_top_k, len(chunks)),
    )
    out: list[dict] = []
    for rank, row in enumerate(rows, start=1):
        idx = row.get("index", -1)
        if not isinstance(idx, int) or idx < 0 or idx >= len(chunks):
            continue
        doc = dict(chunks[idx])
        doc["rerank_score"] = float(row.get("score", 0.0))
        doc["rerank_rank"] = rank
        out.append(doc)
    return out


async def complete_rag_answer(
    question: str,
    collection_base: str,
    request_id: str,
    session_id: str,
    *,
    k: int = 5,
    k_max: int = 50,
    max_tokens: int | None = None,
    expand_on_not_found: bool = True,
    rerank_top_n: int | None = None,
    rerank_return_top_k: int | None = None,
    retrieve_fallback_n: int | None = None,
    final_context_top_k: int | None = None,
    use_reranker: bool = True,
    include_follow_up_questions: bool = True,
    follow_up_candidates: int = 8,
    follow_up_final: int = 3,
    include_retrieval_hits: bool = False,
) -> tuple[str, list[dict], list[str], dict[str, int], list[dict]]:
    """
    ``query_chunks`` → numbered context → ``POST .../v1/chat/completions``.
    Uses ``get_inference_url`` / ``get_inference_model`` / ``get_inference_max_tokens`` (from ``.env`` via ``app.config``).

    Returns ``(answer, citations, follow_up_questions, latency_ms, retrieval_hits)`` where ``citations`` lists only passages the model
    referenced with ``[n]`` in ``answer`` (each item: ``cite_id``, ``chunk_id``, ``source``, ``text``).
    ``follow_up_questions`` is empty when disabled or on failure; otherwise up to ``follow_up_final`` strings.
    ``latency_ms`` maps phase names to integer milliseconds (``total``, ``embed``, ``retrieve``, ``chunk_rerank``,
    ``chat``, ``follow_up_chat``, ``follow_up_rerank``); unused phases are ``0``.
    ``retrieval_hits`` is empty unless ``include_retrieval_hits`` is true; then each item has
    ``stage`` (``retrieve`` or ``rerank``), ``rank``, ``chunk_id``, ``source``, ``score`` (RRF vs rerank scale; not comparable across stages).

    One hybrid retrieval at ``k_max`` (query embedded once, Qdrant limit
    ``max(TOP_K_DENSE, k_max)``). NOT_FOUND / empty retries only **widen the local slice**
    within the merged candidate pool (no second embed, no second Qdrant round).

    Rerank keeps ``rerank_return_top_k`` rows from ``/v1/rerank``; then up to ``retrieve_fallback_n``
    raw-retrieval chunks are appended if absent from that list. The chat context uses at most
    ``final_context_top_k`` passages per turn while widening.

    Set ``expand_on_not_found=False`` for a single chat call at the initial ``k`` (typical for eval).
    """
    if max_tokens is None:
        max_tokens = get_inference_max_tokens()
    if rerank_top_n is None:
        rerank_top_n = get_rerank_top_n()
    if rerank_return_top_k is None:
        rerank_return_top_k = get_rerank_return_top_k()
    if retrieve_fallback_n is None:
        retrieve_fallback_n = get_retrieve_fallback_n()
    if final_context_top_k is None:
        final_context_top_k = get_final_context_top_k()
    infer_base = get_inference_url()
    model = get_inference_model()
    rerank_base = get_rerank_url()
    rerank_model = get_rerank_model()

    if k < 1:
        raise ValueError("k must be at least 1.")
    if k_max < k:
        raise ValueError("k_max must be >= k.")
    if rerank_top_n < 1:
        raise ValueError("rerank_top_n must be >= 1.")
    if rerank_return_top_k < 1:
        raise ValueError("rerank_return_top_k must be >= 1.")
    if retrieve_fallback_n < 0:
        raise ValueError("retrieve_fallback_n must be >= 0.")
    if final_context_top_k < 1:
        raise ValueError("final_context_top_k must be >= 1.")
    if use_reranker and rerank_return_top_k < final_context_top_k:
        raise ValueError(
            "rerank_return_top_k must be >= final_context_top_k when use_reranker is true "
            "so the widen pool can reach the configured max context size."
        )
    if follow_up_final < 1:
        raise ValueError("follow_up_final must be at least 1.")
    if follow_up_candidates < 3 or follow_up_candidates > 12:
        raise ValueError("follow_up_candidates must be between 3 and 12.")
    if follow_up_final > follow_up_candidates:
        raise ValueError("follow_up_final must be <= follow_up_candidates.")

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
        wall_t0 = time.perf_counter()
        logger.info(
            "complete_rag_answer start collection_base=%s k=%s k_max=%s rerank_top_n=%s "
            "rerank_return_top_k=%s retrieve_fallback_n=%s final_context_top_k=%s "
            "use_reranker=%s follow_ups=%s cand=%s final=%s",
            collection_base,
            k,
            k_max,
            rerank_top_n,
            rerank_return_top_k,
            retrieve_fallback_n,
            final_context_top_k,
            use_reranker,
            include_follow_up_questions,
            follow_up_candidates,
            follow_up_final,
        )
        t_embed = time.perf_counter()
        query_vector = await embed_text(
            question, request_id=request_id, session_id=session_id
        )
        embed_ms = _elapsed_ms(t_embed)
        retrieve_pool = max(k_max, rerank_top_n)
        t_ret = time.perf_counter()
        chunks_full = await query_chunks(
            question,
            collection_base,
            k=retrieve_pool,
            request_id=request_id,
            session_id=session_id,
            query_vector=query_vector,
            qdrant_limit_override=retrieve_pool,
        )
        retrieve_ms = _elapsed_ms(t_ret)
        if not chunks_full:
            raise ValueError("No chunks retrieved for this query.")

        candidate_chunks = chunks_full[: min(rerank_top_n, len(chunks_full))]
        chunk_rerank_ms = 0
        reranked_for_hits: list[dict] | None = None
        if use_reranker:
            try:
                t_rr = time.perf_counter()
                reranked = await _rerank_chunks(
                    question=question,
                    chunks=candidate_chunks,
                    rerank_url=rerank_base,
                    rerank_model=rerank_model,
                    rerank_return_top_k=rerank_return_top_k,
                )
                chunk_rerank_ms = _elapsed_ms(t_rr)
                if reranked:
                    candidate_chunks = _merge_rerank_with_retrieve_fallback(
                        reranked,
                        chunks_full,
                        fallback_n=retrieve_fallback_n,
                    )
                    reranked_for_hits = reranked
                    fallback_added = max(0, len(candidate_chunks) - len(reranked))
                else:
                    fallback_added = 0
                logger.info(
                    "complete_rag_answer rerank applied candidates=%s rerank_kept=%s "
                    "fallback_added=%s merged_pool=%s",
                    len(chunks_full),
                    len(reranked),
                    fallback_added,
                    len(candidate_chunks),
                )
            except Exception as e:
                logger.warning(
                    "complete_rag_answer rerank fallback reason=%s",
                    str(e),
                )

        current_k = min(k, len(candidate_chunks), final_context_top_k)
        last_answer = ""
        last_citations: list[dict] = []
        chunks_for_followups: list[dict] = []
        chat_ms_total = 0
        while True:
            chunks_for_followups = candidate_chunks[:current_k]

            context, last_citations = _build_numbered_context(chunks_for_followups)
            messages = [
                system_msg,
                {
                    "role": "user",
                    "content": (
                        f"Context:\n---\n{context}\n---\n\nQuestion: {question}"
                    ),
                },
            ]
            t_chat = time.perf_counter()
            last_answer = await chat_complete(
                base_url=infer_base,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )
            chat_ms_total += _elapsed_ms(t_chat)
            if not _answer_needs_more_context(last_answer):
                logger.info("complete_rag_answer chat ok k_used=%s", current_k)
                break

            if not expand_on_not_found:
                logger.info(
                    "complete_rag_answer done single_pass k_used=%s (expand_on_not_found=False)",
                    current_k,
                )
                break

            next_k = min(current_k * 2, final_context_top_k, len(candidate_chunks))
            if next_k <= current_k:
                logger.info(
                    "complete_rag_answer done needs_more_context k_used=%s",
                    current_k,
                )
                break
            logger.info(
                "complete_rag_answer widen context slice %s -> %s (same retrieval pool)",
                current_k,
                next_k,
            )
            current_k = next_k

        answer_out, citations_out = _with_citations(last_answer, last_citations)
        follow_ups: list[str] = []
        follow_up_chat_ms = 0
        follow_up_rerank_ms = 0
        if include_follow_up_questions:
            follow_ups, follow_up_chat_ms, follow_up_rerank_ms = await _follow_up_questions(
                question=question,
                answer=answer_out,
                chunks_used=chunks_for_followups,
                follow_up_candidates=follow_up_candidates,
                follow_up_final=follow_up_final,
                infer_base=infer_base,
                model=model,
                max_tokens_main=max_tokens,
                rerank_url=rerank_base,
                rerank_model=rerank_model,
            )
        total_ms = _elapsed_ms(wall_t0)
        latency_ms: dict[str, int] = {
            "total": total_ms,
            "embed": embed_ms,
            "retrieve": retrieve_ms,
            "chunk_rerank": chunk_rerank_ms,
            "chat": chat_ms_total,
            "follow_up_chat": follow_up_chat_ms,
            "follow_up_rerank": follow_up_rerank_ms,
        }
        logger.info(
            "complete_rag_answer done k_used=%s follow_up_questions=%s latency_total_ms=%s",
            current_k,
            len(follow_ups),
            total_ms,
            extra={
                "duration_ms": total_ms,
                "latency_total_ms": total_ms,
                "latency_embed_ms": embed_ms,
                "latency_retrieve_ms": retrieve_ms,
                "latency_chunk_rerank_ms": chunk_rerank_ms,
                "latency_chat_ms": chat_ms_total,
                "latency_follow_up_chat_ms": follow_up_chat_ms,
                "latency_follow_up_rerank_ms": follow_up_rerank_ms,
            },
        )
        retrieval_hits: list[dict] = []
        if include_retrieval_hits:
            retrieval_hits = _retrieval_hits_payload(chunks_full, reranked_for_hits)
        return answer_out, citations_out, follow_ups, latency_ms, retrieval_hits


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
        default=50,
        help="Retrieval pool size (RRF top-k); also cap when widening context slice on NOT_FOUND",
    )
    p.add_argument(
        "--single-pass",
        action="store_true",
        help="One chat at initial k only (no widening slice after NOT_FOUND / empty).",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=get_inference_max_tokens(),
    )
    p.add_argument(
        "--rerank-top-n",
        type=int,
        default=get_rerank_top_n(),
        help="RRF candidate pool size sent to reranker (document count).",
    )
    p.add_argument(
        "--rerank-return-top-k",
        type=int,
        default=get_rerank_return_top_k(),
        help="Rerank API top_n: ranked rows to keep before retrieve fallback (>= final-top-k).",
    )
    p.add_argument(
        "--retrieve-fallback-n",
        type=int,
        default=get_retrieve_fallback_n(),
        help="After rerank, append up to N RRF-ordered chunks missing from rerank output.",
    )
    p.add_argument(
        "--final-top-k",
        type=int,
        default=get_final_context_top_k(),
        help="Max passages in one chat context (NOT_FOUND widen cap).",
    )
    p.add_argument(
        "--no-reranker",
        action="store_true",
        help="Skip reranker and use fused order directly.",
    )
    p.add_argument(
        "--no-follow-ups",
        action="store_true",
        help="Skip follow-up question generation (no extra chat + rerank).",
    )
    p.add_argument(
        "--follow-up-candidates",
        type=int,
        default=8,
        help="LLM generates this many follow-up candidates (3–12) before rerank.",
    )
    p.add_argument(
        "--follow-up-final",
        type=int,
        default=3,
        help="Return this many follow-up questions after rerank (<= candidates).",
    )
    p.add_argument(
        "--retrieval-hits",
        action="store_true",
        help="Include retrieval_hits (RRF retrieve stage + optional rerank stage) in JSON for eval/debug.",
    )
    args = p.parse_args(argv)

    rid = os.environ.get("RAG_REQUEST_ID") or str(uuid.uuid4())
    sid = os.environ.get("RAG_SESSION_ID") or str(uuid.uuid4())

    try:
        answer, citations, follow_up_questions, latency_ms, _retrieval_hits = run_async(
            complete_rag_answer(
                args.question,
                args.collection,
                rid,
                sid,
                k=args.k,
                k_max=args.k_max,
                max_tokens=args.max_tokens,
                expand_on_not_found=not args.single_pass,
                rerank_top_n=args.rerank_top_n,
                rerank_return_top_k=args.rerank_return_top_k,
                retrieve_fallback_n=args.retrieve_fallback_n,
                final_context_top_k=args.final_top_k,
                use_reranker=not args.no_reranker,
                include_follow_up_questions=not args.no_follow_ups,
                follow_up_candidates=args.follow_up_candidates,
                follow_up_final=args.follow_up_final,
                include_retrieval_hits=args.retrieval_hits,
            )
        )
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    except httpx.HTTPStatusError as e:
        print(e.response.text, file=sys.stderr)
        return 2
    out: dict[str, Any] = {
        "answer": answer,
        "citations": citations,
        "follow_up_questions": follow_up_questions,
        "latency_ms": latency_ms,
    }
    if args.retrieval_hits:
        out["retrieval_hits"] = _retrieval_hits
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

