"""Function-aware chunking for Python, using the standard-library ast module.

Why: the embedder (all-MiniLM-L6-v2) reads only the first 256 word pieces of a
chunk, about 28 lines of code. With fixed 60-line windows a definition often
sits past that cutoff, so retrieval finds where a name is *used* instead of
where it is *defined*. Cutting at def/class boundaries puts every definition
at the start of its own chunk, with its decorators and docstring.

Rules:
  - In a module or class body, each def/class gets its own chunk, starting at
    its first decorator (plus any # comment lines directly above it).
  - Other statements (imports, constants, class attributes) are packed into
    chunks of at most MAX_LINES lines.
  - A def/class longer than MAX_LINES is split at statement boundaries; its
    first piece keeps the signature and docstring. A single statement longer
    than MAX_LINES falls back to line windows.
  - Adjacent pieces shorter than SMALL_LINES are merged (up to SMALL_LINES), so
    a file of two-line exception classes doesn't become dozens of tiny chunks.
  - If the file doesn't parse, the whole file falls back to 60-line chunks.
Every line of the file lands in exactly one chunk (no overlap, nothing lost).
"""
from __future__ import annotations

import ast

from chunkers import CHUNK_SIZE, chunk_lines, make_chunk

MAX_LINES = CHUNK_SIZE  # same cap as the line chunker, so an A/B only moves the cut points
SMALL_LINES = 25

_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _node_type(node: ast.AST) -> str:
    return "class" if isinstance(node, ast.ClassDef) else "function"


def _start(node: ast.stmt) -> int:
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno] + [d.lineno for d in decorators])


def _leading_comments(lines: list[str], start: int, floor: int) -> int:
    """Move start up over # comment lines directly above it (not past floor)."""
    while start - 1 >= floor and lines[start - 2].lstrip().startswith("#"):
        start -= 1
    return start


class _Splitter:
    def __init__(self, lines: list[str]):
        self.lines = lines
        # (start, end, node_type), 1-based inclusive, contiguous and in order.
        self.pieces: list[tuple[int, int, str]] = []

    def body(self, start: int, end: int, children: list[ast.stmt], kind: str, defs_alone: bool) -> None:
        """Cover lines start..end, whose statements are `children`.

        defs_alone: in a module/class body every def/class is its own piece.
        In a function body, statements are only packed by size.
        """
        cur = start
        for child in children:
            cs, ce = _start(child), child.end_lineno
            if defs_alone and isinstance(child, _DEFS):
                cs = _leading_comments(self.lines, cs, cur)
                if cs > cur:
                    self.pieces.append((cur, cs - 1, kind))
                self.node(child, cs)
                cur = ce + 1
                continue
            if ce - cur + 1 <= MAX_LINES:
                continue  # still fits in the current piece
            if cs > cur:
                self.pieces.append((cur, cs - 1, kind))
                cur = cs
            if ce - cur + 1 > MAX_LINES:  # one statement too big on its own
                self.pieces.append((cur, ce, "chunk"))
                cur = ce + 1
        if cur <= end:
            self.pieces.append((cur, end, kind))

    def node(self, node: ast.stmt, start: int) -> None:
        end = node.end_lineno
        kind = _node_type(node)
        if end - start + 1 <= MAX_LINES:
            self.pieces.append((start, end, kind))
        else:
            self.body(start, end, node.body, kind, defs_alone=isinstance(node, ast.ClassDef))


def _absorb_blank(pieces: list[tuple[int, int, str]], lines: list[str]) -> list[tuple[int, int, str]]:
    """Attach whitespace-only pieces (gaps between defs) to the piece before,
    so a merged chunk never starts with blank lines in front of a definition."""
    out: list[tuple[int, int, str]] = []
    for start, end, kind in pieces:
        if out and not "".join(lines[start - 1:end]).strip():
            ps, _, pk = out[-1]
            out[-1] = (ps, end, pk)
        else:
            out.append((start, end, kind))
    return out


def _merge_small(pieces: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    merged: list[tuple[int, int, str]] = []
    for start, end, kind in pieces:
        if merged:
            ps, pe, pk = merged[-1]
            if (pe - ps + 1) < SMALL_LINES and (end - start + 1) < SMALL_LINES and end - ps + 1 <= SMALL_LINES:
                merged[-1] = (ps, end, pk if pk == kind else "group")
                continue
        merged.append((start, end, kind))
    return merged


def chunk_python(lines: list[str], file: str, language: str = "py") -> list[dict]:
    """Chunks for one Python file (lines as from readlines(), keepends)."""
    try:
        tree = ast.parse("".join(lines))
    except (SyntaxError, ValueError):  # ValueError: source contains null bytes
        return chunk_lines(lines, file, language)
    if not lines:
        return []

    splitter = _Splitter(lines)
    splitter.body(1, len(lines), tree.body, "module", defs_alone=True)

    chunks = []
    for start, end, kind in _merge_small(_absorb_blank(splitter.pieces, lines)):
        if kind == "chunk":  # oversized single statement: line windows
            chunks.extend(chunk_lines(lines, file, language, start, end))
            continue
        chunk = make_chunk(lines, start, end, file, language, kind)
        if chunk:
            chunks.append(chunk)
    return chunks
