"""Reciprocal Rank Fusion (RRF).

Merges several ranked lists into one using only each item's rank:

    score(d) = sum over lists L containing d of  1 / (k + rank_L(d))     (rank from 1)

Why ranks and not scores: BM25 scores are unbounded and depend on query length,
while vector distances live on a different scale entirely. Adding them would
need per-corpus normalisation and a tuned weight; RRF needs neither. k = 60 is
the constant from the original paper (Cormack et al., 2009): it damps the
difference between rank 1 and rank 2 so one retriever can't dominate just by
being confident about its top hit.
"""
from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TypeVar

T = TypeVar("T", bound=Hashable)

RRF_K = 60


def reciprocal_rank_fusion(rankings: Sequence[Sequence[T]], k: int = RRF_K) -> list[tuple[T, float]]:
    """Fuse ranked lists (best first) into one list of (item, score), best first.

    Pure function. Duplicates inside one list count only at their best rank.
    Ties are broken by the item's best rank in any list, then by the order in
    which the lists were given, so the output is deterministic.
    """
    if k < 0:
        raise ValueError("k must be >= 0")

    scores: dict[T, float] = {}
    best_rank: dict[T, tuple[int, int]] = {}  # item -> (rank, list index)
    for list_idx, ranking in enumerate(rankings):
        seen: set[T] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen:
                continue
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            if item not in best_rank or (rank, list_idx) < best_rank[item]:
                best_rank[item] = (rank, list_idx)

    return sorted(scores.items(), key=lambda kv: (-kv[1], best_rank[kv[0]]))
