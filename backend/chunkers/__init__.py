"""Chunkers turn a file into retrieval chunks.

Every chunk is a dict with: content, file, lines ("start-end", 1-based,
inclusive), language, node_type. The line chunker below is the default for
every language and the fallback for the Python AST chunker.
"""
from __future__ import annotations

CHUNK_SIZE = 60
CHUNK_OVERLAP = 10
MIN_CHUNK_CHARS = 20


def make_chunk(lines: list[str], start: int, end: int, file: str, language: str, node_type: str) -> dict | None:
    """Chunk for 1-based inclusive lines start..end, or None if it's near-empty."""
    content = "".join(lines[start - 1:end])
    if len(content.strip()) < MIN_CHUNK_CHARS:
        return None
    return {
        "content": content,
        "file": file,
        "lines": f"{start}-{end}",
        "language": language,
        "node_type": node_type,
    }


def chunk_lines(lines: list[str], file: str, language: str, first: int = 1, last: int | None = None) -> list[dict]:
    """Fixed windows of CHUNK_SIZE lines overlapping by CHUNK_OVERLAP.

    first/last (1-based, inclusive) restrict it to part of the file; used by the
    Python chunker for a single statement too long to keep whole.
    """
    last = len(lines) if last is None else last
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    start = first
    while start <= last:
        end = min(start + CHUNK_SIZE - 1, last)
        chunk = make_chunk(lines, start, end, file, language, "chunk")
        if chunk:
            chunks.append(chunk)
        if end == last:
            break
        start += step
    return chunks
