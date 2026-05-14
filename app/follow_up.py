"""Follow-up question generation: extra LLM call + reranker over candidate strings.

Public surface: ``generate_follow_ups`` — invoked by ``app.rag_answer.complete_rag_answer``
after the main RAG answer is produced. All failures are logged and degrade to ``[]`` so
the primary RAG response shape stays stable.
"""
from __future__ import annotations

import json
import logging
import time

from app.http.inference import chat_complete
from app.http.rerank import rerank_texts
from app.logging_config import logger

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


def _sanitize_for_log(text: str) -> str:
    """Single-line safe form (newlines escaped) for stderr JSON; no truncation."""
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")


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


_FOLLOW_UP_OBJECT_KEYS = ("follow_up_questions", "questions", "follow_ups")


def _coerce_to_strings(value: object) -> list[str]:
    """Pull question strings from supported shapes: list, or dict with one of the known keys."""
    if isinstance(value, list):
        return [x for x in value if isinstance(x, str)]
    if isinstance(value, dict):
        for key in _FOLLOW_UP_OBJECT_KEYS:
            v = value.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, str)]
        for v in value.values():
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                return list(v)
    return []


def _parse_follow_up_json(raw: str) -> tuple[list[str], str | None]:
    """
    Parse model output into a list of non-empty question strings.

    Accepts the canonical shape ``{"follow_up_questions": ["…", "…"]}`` and
    legacy bare arrays. Recovers from common LLM glitches: code fences,
    concatenated top-level JSON values (``["Q1"]["Q2"]["Q3"]``), and
    array-fragment slices.

    When the returned list is empty, the second element is a stable machine
    reason for logging (``None`` when the list is non-empty).
    """
    text = raw.strip()
    if not text:
        return [], "empty_model_reply"
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
    if not text.strip():
        return [], "empty_after_code_fence_strip"

    # Greedy scan: collect 1..N concatenated top-level JSON values, tolerating
    # whitespace and stray commas between them. Handles ``["Q1"]["Q2"]["Q3"]``
    # and ``{"follow_up_questions":[…]}``.
    decoder = json.JSONDecoder()
    values: list[object] = []
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n:
            break
        try:
            value, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            break
        values.append(value)
        i = end

    if not values:
        i0 = text.find("[")
        i1 = text.rfind("]")
        if i0 == -1 or i1 <= i0:
            return [], "json_invalid_no_array_slice"
        try:
            values = [json.loads(text[i0 : i1 + 1])]
        except json.JSONDecodeError:
            return [], "json_invalid_bracket_slice_failed"

    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        for s in _coerce_to_strings(value):
            t = s.strip()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
    if not out:
        if len(values) == 1 and not isinstance(values[0], (list, dict)):
            return [], f"parsed_not_list:{type(values[0]).__name__}"
        return [], "parsed_list_no_non_empty_strings"
    return out, None


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
    request_id: str,
    session_id: str,
    trace_id: str | None = None,
    conversation_id: str | None = None,
) -> tuple[list[str], str]:
    """One chat call: returns ``(candidates, raw)`` where ``candidates`` is the parsed
    ``{"follow_up_questions": [...]}`` list (between ``min_count`` and ``max_count``
    distinct strings) and ``raw`` is the unparsed assistant content (for logging)."""
    sys = (
        "Return ONLY one valid JSON object with a single key "
        '"follow_up_questions" whose value is an array of strings. '
        "No markdown, no commentary, no extra keys, no multiple objects. "
        f"Produce between {min_count} and {max_count} distinct questions, "
        "each under 120 characters. The user just received the answer below "
        "for their question; generate questions that DRILL DOWN into the same "
        "specific sub-topic as the answer (clarifications, edge cases, next "
        "logical steps, related sub-aspects). Stay tightly on the topic the "
        "user actually asked about. Do NOT pivot to other facts that appear "
        "only in the context summary but were not part of the answer. Each "
        "question must be answerable from the same retrieved passages. "
        "Do not repeat the original question verbatim."
    )
    user = (
        f"Original question:\n{question}\n\n"
        f"Answer that was given:\n{answer}\n\n"
        f"Context summary (retrieved passages):\n{context_summary}\n\n"
        f'Return EXACTLY this shape with {min_count}-{max_count} entries:\n'
        '{"follow_up_questions": ["question 1", "question 2", "question 3"]}'
    )
    raw = await chat_complete(
        base_url=infer_base,
        model=model,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        request_id=request_id,
        session_id=session_id,
        trace_id=trace_id,
        conversation_id=conversation_id,
    )
    candidates, empty_reason = _parse_follow_up_json(raw)
    if not candidates:
        preview = _preview_for_log(raw) if raw.strip() else "-"
        lvl = logging.WARNING if empty_reason != "empty_model_reply" else logging.INFO
        logger.log(
            lvl,
            "follow_up_questions_empty reason=%s reply_chars=%s raw_preview=%s",
            empty_reason,
            len(raw),
            preview,
            extra={"follow_up_empty_reason": empty_reason},
        )
    return candidates, raw


async def _rerank_follow_up_strings(
    *,
    question: str,
    answer: str,
    candidates: list[str],
    rerank_url: str,
    rerank_model: str,
    top_n: int,
    request_id: str,
    session_id: str,
    trace_id: str | None = None,
    conversation_id: str | None = None,
) -> list[str]:
    """Rerank candidate questions; return top ``top_n`` strings in score order.

    The reranker query concatenates ``question`` + ``answer`` so the cross-encoder
    has the full semantic target (terms from the answer body are usually what
    distinguishes drill-downs from off-topic candidates)."""
    if not candidates:
        return []
    n = min(top_n, len(candidates))
    rerank_query = f"{question}\n\n{answer}".strip() if answer else question
    try:
        rows = await rerank_texts(
            base_url=rerank_url,
            model=rerank_model,
            query=rerank_query,
            documents=candidates,
            top_n=n,
            request_id=request_id,
            session_id=session_id,
            trace_id=trace_id,
            conversation_id=conversation_id,
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


async def generate_follow_ups(
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
    request_id: str,
    session_id: str,
    trace_id: str | None = None,
    conversation_id: str | None = None,
) -> tuple[list[str], int, int]:
    """Returns ``(questions, chat_ms, rerank_ms)``; times are zero when skipped or on failure."""
    if not chunks_used:
        logger.info(
            "follow_up_questions_empty reason=no_chunks_used",
            extra={"follow_up_empty_reason": "no_chunks_used"},
        )
        return [], 0, 0
    min_gen = max(3, follow_up_candidates - 3)
    max_gen = follow_up_candidates
    if min_gen > max_gen:
        min_gen = max_gen
    summary = _context_summary_for_followups(chunks_used)
    gen_budget = min(_FOLLOW_UP_GEN_MAX_TOKENS_CAP, max(256, max_tokens_main))
    gen_t0 = time.perf_counter()
    try:
        candidates, raw = await _generate_follow_up_candidates(
            question=question,
            answer=answer,
            context_summary=summary,
            infer_base=infer_base,
            model=model,
            min_count=min_gen,
            max_count=max_gen,
            max_tokens=gen_budget,
            request_id=request_id,
            session_id=session_id,
            trace_id=trace_id,
            conversation_id=conversation_id,
        )
    except Exception as e:
        logger.warning(
            "follow_up_questions_empty reason=generation_failed detail=%s",
            str(e),
            extra={"follow_up_empty_reason": "generation_failed", "error_message": str(e)},
        )
        return [], _elapsed_ms(gen_t0), 0
    gen_ms = _elapsed_ms(gen_t0)
    if not candidates:
        return [], gen_ms, 0
    rr_t0 = time.perf_counter()
    ranked = await _rerank_follow_up_strings(
        question=question,
        answer=answer,
        candidates=candidates,
        rerank_url=rerank_url,
        rerank_model=rerank_model,
        top_n=follow_up_final,
        request_id=request_id,
        session_id=session_id,
        trace_id=trace_id,
        conversation_id=conversation_id,
    )
    rr_ms = _elapsed_ms(rr_t0)
    if ranked:
        logger.info(
            "follow_up_questions_ok cand=%s ranked=%s reply_chars=%s",
            len(candidates),
            len(ranked),
            len(raw),
            extra={
                "follow_up_raw_reply": _sanitize_for_log(raw),
                "follow_up_candidates_full": list(candidates),
                "follow_up_candidates_count": len(candidates),
                "follow_up_ranked": list(ranked),
                "follow_up_ranked_count": len(ranked),
                "latency_follow_up_chat_ms": gen_ms,
                "latency_follow_up_rerank_ms": rr_ms,
            },
        )
    return ranked, gen_ms, rr_ms
