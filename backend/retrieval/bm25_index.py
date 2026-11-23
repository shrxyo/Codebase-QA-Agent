"""Keyword (BM25) search over a repo's chunks.

Embeddings only read the first 256 word pieces of a chunk (all-MiniLM-L6-v2),
which is about half of a 60-line chunk. BM25 scores the whole chunk, and it
matches exact identifiers, so it covers what the embedder can't see.

The index is saved as JSON (chunk ids + tokens) next to the ChromaDB data and
rebuilt in memory on startup. Rebuilding BM25 from tokens takes milliseconds,
so there's no need to pickle the model itself.
"""
from __future__ import annotations

import json
import os
import re

from rank_bm25 import BM25Okapi

# Runs of letters/digits; everything else (_ . / - : etc.) is a separator.
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
# Identifiers kept whole as well as split: foo_bar, a.b.c, path/to/x, kebab-case.
_IDENT_RE = re.compile(r"[A-Za-z0-9]+(?:[_.\-/][A-Za-z0-9]+)+")
# camelCase / PascalCase / acronym boundaries: "HTTPAdapter" -> HTTP, Adapter.
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

# Question filler only. Kept short on purpose: words like "file", "function",
# "index" or "error" can be meaningful in a codebase, so they stay.
STOPWORDS = frozenset("""
a an and are as at be by can do does for from how i in is it its me of on or
should so that the this to was what when where which who why will with you
""".split())


def tokenize(text: str) -> list[str]:
    """Lowercased tokens with code identifiers split into their parts.

    "HTTPAdapter.send_request" -> http, adapter, send, request,
    plus the whole identifier "httpadapter.send_request" and "send_request"
    style compounds, so an exact-name query also matches as one rare term.
    """
    tokens: list[str] = []
    for word in _WORD_RE.findall(text):
        parts = _CAMEL_RE.findall(word) or [word]
        tokens.extend(p.lower() for p in parts)
        if len(parts) > 1:
            tokens.append(word.lower())
    for ident in _IDENT_RE.findall(text):
        tokens.append(ident.lower())
        # Also the snake_case compounds inside a dotted/slashed path.
        pieces = re.split(r"[./\-]", ident)
        if len(pieces) > 1:
            tokens.extend(p.lower() for p in pieces if "_" in p)
    return [t for t in tokens if t not in STOPWORDS]


class BM25Index:
    """BM25 over chunks, addressed by the same ids as the ChromaDB collection."""

    def __init__(self, ids: list[str], tokens: list[list[str]]):
        if len(ids) != len(tokens):
            raise ValueError("ids and tokens must have the same length")
        self.ids = ids
        self._tokens = tokens
        # BM25Okapi divides by the corpus size, so it can't be built empty.
        self._bm25 = BM25Okapi(tokens) if ids else None

    @classmethod
    def from_documents(cls, ids: list[str], documents: list[str]) -> "BM25Index":
        return cls(ids, [tokenize(d) for d in documents])

    def search(self, query: str, n: int) -> list[tuple[str, float]]:
        """Top-n (id, score), best first. Chunks sharing no query term are dropped."""
        terms = tokenize(query)
        if self._bm25 is None or not terms or n <= 0:
            return []
        scores = self._bm25.get_scores(terms)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [(self.ids[i], float(scores[i])) for i in order[:n] if scores[i] > 0]

    def to_dict(self) -> dict:
        return {"version": 1, "ids": self.ids, "tokens": self._tokens}

    @classmethod
    def from_dict(cls, data: dict) -> "BM25Index":
        return cls(data["ids"], data["tokens"])

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.to_dict(), f)
        os.replace(tmp, path)  # atomic: a crash never leaves half a file

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        with open(path) as f:
            return cls.from_dict(json.load(f))
