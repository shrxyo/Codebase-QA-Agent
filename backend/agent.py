import json
import logging
import re
import time
from typing import Generator, Iterable

import llm_client
from indexer import RepoIndexer

SYSTEM_PROMPT = (
    "You are Codebase-QA-Agent, a code analysis assistant. Answer questions about a GitHub repository "
    "using retrieved code chunks. Trace actual call flows through the code. Be precise and concise. "
    "If context is insufficient, say so clearly.\n\n"
    # Task 5 baseline: with only "cite using [filename:lines] notation", 23/24
    # answers named files in prose ("see `x.py` lines 5-9"), which can't be
    # verified. An explicit format, an example and a ban on prose references fix that.
    "Citations: back every statement about the code with an inline citation in square "
    "brackets: the file path exactly as shown in the context headers, a colon, then the "
    "line number or range, e.g. [app/models/user.py:12-30] or [app/models/user.py:42]. "
    "Only cite lines that appear in the provided context. Whenever you mention a file, "
    "include a bracket citation; never write 'see file.py lines 5-9' in prose. "
    "Use a plain hyphen in ranges."
)

MAX_HISTORY_MESSAGES = 6
MAX_TOKENS = 1500

logger = logging.getLogger(__name__)


def _ms_since(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


# ---------------------------------------------------------------------------
# Citation verification (pure functions: no I/O, shared with eval_answers.py)
# ---------------------------------------------------------------------------

# The model writes typographic dashes and narrow no-break spaces, even inside
# file names ("docker‑compose.yml" with U+2011), so normalize before parsing.
_NORMALIZE = {
    **dict.fromkeys(map(ord, "‐‑‒–—―−﹣－"), "-"),
    **dict.fromkeys(map(ord, "    "), " "),
    **dict.fromkeys(map(ord, "`*\"'"), None),  # markdown/quotes inside the brackets
}
_BRACKET_RE = re.compile(r"\[([^\[\]\n]{1,300})\]")
_RANGE = r"L?\d+(?:\s*-\s*L?\d+)?"
# A file reference inside a bracket: path, separator, one or more line ranges.
# Accepts "a/b.py:10-20", "b.py:10", "b.py#L10-L20", "b.py lines 10-20",
# "b.py, line 7", "b.py:10-20, 30-40".
_REF_RE = re.compile(
    r"(?P<path>(?:\.{1,2}/)*\.?[\w@+-]+(?:[./][\w@+-]+)*)"
    r"\s*(?::|#|,?\s+lines?\s*|,?\s+(?=L\d))\s*"
    rf"(?P<ranges>{_RANGE}(?:\s*,\s*{_RANGE})*)",
    re.IGNORECASE,
)
_EXTENSIONLESS = {"Dockerfile", "Makefile", "Procfile", "LICENSE"}
_MAX_LISTED = 5


def _parse_range(text: str) -> tuple[int, int]:
    nums = [int(n) for n in re.findall(r"\d+", text)]
    return (nums[0], nums[-1])


def _looks_like_path(path: str) -> bool:
    return "." in path.strip(".") or "/" in path or path in _EXTENSIONLESS


def parse_citations(answer: str) -> list[dict]:
    """Find every [file:lines] citation in an answer.

    Returns [{"raw", "file", "start", "end"}], one per line range, where "raw"
    is the exact bracket text (so the UI can find it in the answer). Brackets
    that aren't file references ([DONE], ["key"], [str, int],
    [filename:lines]) are ignored. Repeats of the same file+range are kept
    once: citing the same lines twice is one claim, not two.
    """
    out, seen = [], set()
    for bracket in _BRACKET_RE.finditer(answer):
        inner = bracket.group(1).translate(_NORMALIZE)
        for ref in _REF_RE.finditer(inner):
            path = ref.group("path")
            if not _looks_like_path(path):
                continue
            for part in ref.group("ranges").split(","):
                start, end = _parse_range(part)
                key = (path, start, end)
                if key in seen:
                    continue
                seen.add(key)
                out.append({"raw": bracket.group(0), "file": path, "start": start, "end": end})
    return out


def _resolve_file(path: str, known_files: Iterable[str]) -> list[str]:
    """Indexed files the cited path can refer to.

    Exact match first; otherwise a whole-component suffix match, because the
    model often drops leading folders ("requests/auth.py" for
    "src/requests/auth.py", "main.py" for "backend/main.py").
    """
    while path.startswith(("./", "../")):
        path = path.split("/", 1)[1]
    path = path.lstrip("/")
    known = list(known_files)
    if path in known:
        return [path]
    return [f for f in known if f.endswith("/" + path)]


def verify_citations(answer: str, retrieved: list[dict], indexed_files: Iterable[str]) -> dict:
    """Check each [file:lines] citation against the index and the retrieval.

    A citation is valid only if its file is in the index AND its line range
    overlaps a chunk that was retrieved for this answer, i.e. the model could
    actually have read those lines. `retrieved` is [{"file", "lines": "a-b"}].

    Returns {"score", "reason", "total", "valid", "citations": [...]}. With no
    citations the score is 0: an answer that cites nothing is unverified, not
    perfect.
    """
    chunks: dict[str, list[tuple[int, int]]] = {}
    for src in retrieved:
        chunks.setdefault(src["file"], []).append(_parse_range(src["lines"]))
    known = set(indexed_files) | set(chunks)

    results = []
    for c in parse_citations(answer):
        lines = f"{c['start']}-{c['end']}" if c["end"] != c["start"] else str(c["start"])
        candidates = _resolve_file(c["file"], known)
        if c["start"] < 1 or c["end"] < c["start"]:
            valid, why, resolved = False, "malformed line range", None
        elif not candidates:
            valid, why, resolved = False, "file not in the index", None
        else:
            hit = next(
                (f for f in candidates
                 for lo, hi in chunks.get(f, [])
                 if c["start"] <= hi and lo <= c["end"]),
                None,
            )
            resolved = hit or candidates[0]
            if hit:
                valid, why = True, None
            elif any(f in chunks for f in candidates):
                valid, why = False, "lines outside the retrieved chunks"
            else:
                valid, why = False, "file not retrieved for this answer"
        results.append({
            "raw": c["raw"], "file": c["file"], "resolved": resolved,
            "lines": lines, "valid": valid, "why": why,
        })

    total = len(results)
    n_valid = sum(r["valid"] for r in results)
    if total == 0:
        reason = "No [file:lines] citations in the answer, so nothing could be verified."
    elif n_valid == total:
        subject = "The citation points" if total == 1 else f"All {total} citations point"
        reason = f"{subject} at code retrieved for this answer."
    else:
        bad = [f"{r['file']}:{r['lines']} ({r['why']})" for r in results if not r["valid"]]
        more = f"; +{len(bad) - _MAX_LISTED} more" if len(bad) > _MAX_LISTED else ""
        reason = f"{n_valid} of {total} citations verified. Not verified: {'; '.join(bad[:_MAX_LISTED])}{more}."
    return {
        "score": round(n_valid / total, 3) if total else 0.0,
        "reason": reason,
        "total": total,
        "valid": n_valid,
        "citations": results,
    }


def stream_answer(
    indexer: RepoIndexer,
    question: str,
    history: list[dict],
) -> Generator[str, None, None]:
    # All timings are measured from when the server starts producing this
    # response, so TTFT is what the user waits before the first word appears
    # (retrieval + prompt building + model latency, including any reasoning
    # the model does before it emits visible text).
    t_start = time.perf_counter()
    chunks = indexer.query(question, n_results=8)
    retrieval_ms = _ms_since(t_start)

    sources = [{"file": c["file"], "lines": c["lines"]} for c in chunks]
    yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

    context_parts = []
    for c in chunks:
        context_parts.append(f"### {c['file']} (lines {c['lines']})\n```{c['language']}\n{c['content']}\n```")
    context_block = "\n\n".join(context_parts)

    trimmed_history = history[-MAX_HISTORY_MESSAGES:]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(trimmed_history)
    messages.append({
        "role": "user",
        "content": f"Relevant code context:\n\n{context_block}\n\nQuestion: {question}",
    })

    client = llm_client.get_llm_client()
    model = llm_client.get_model_name()

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=MAX_TOKENS,
        stream=True,
        **llm_client.get_request_options(),
    )

    ttft_ms = None
    answer_parts = []
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            if ttft_ms is None:
                ttft_ms = _ms_since(t_start)
            answer_parts.append(delta.content)
            yield f"data: {json.dumps({'type': 'text', 'content': delta.content})}\n\n"

    # Verify citations once the whole answer exists (a citation can be split
    # across tokens). A bug here must never cost the user the answer itself.
    try:
        check = verify_citations("".join(answer_parts), sources, indexer.indexed_files())
        confidence = {
            "type": "confidence",
            "score": check["score"],
            "reason": check["reason"],
            "citations": check["citations"],
        }
        yield f"data: {json.dumps(confidence)}\n\n"
    except Exception:
        logger.exception("Citation verification failed")

    timings = {"retrieval_ms": retrieval_ms, "ttft_ms": ttft_ms, "total_ms": _ms_since(t_start)}
    logger.info("Chat timings %s", timings)
    yield f"data: {json.dumps({'type': 'timings', **timings})}\n\n"
    yield "data: [DONE]\n\n"
