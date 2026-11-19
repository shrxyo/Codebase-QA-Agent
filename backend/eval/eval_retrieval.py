"""Retrieval eval: file-level Recall@k and MRR over the golden dataset.

Calls RepoIndexer directly (no HTTP). Each repo is indexed once from a checkout
of its pinned commit into a separate ChromaDB directory (eval/.index), so the
eval never touches the app's own index and reruns skip re-embedding.

Metric definitions (k counts CHUNK positions, because the agent is given the
top 8 chunks; a chunk is relevant if its file is one of the expected files):
  Recall@k  share of a question's expected files that appear in the top k
            chunks, averaged over questions
  MRR       1 / rank of the first relevant chunk (0 if none in the top 8),
            averaged over questions

Usage (from backend/):
  python -m eval.eval_retrieval                      # embedding-only baseline
  python -m eval.eval_retrieval --method hybrid      # also: bm25
  python -m eval.eval_retrieval --method embedding --reindex
  python -m eval.eval_retrieval --source codebase-qa-agent=..   # unpushed commit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Callable

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(EVAL_DIR, ".index")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")

# indexer reads CHROMA_PERSIST_DIR at import time, and Docker sets it to the
# app's index, so override it (not setdefault) before importing.
os.environ["CHROMA_PERSIST_DIR"] = INDEX_DIR

import indexer as indexer_module  # noqa: E402
from indexer import RepoIndexer  # noqa: E402

from eval.golden_dataset import QUESTIONS, REPOS, GoldenQuestion, checkout  # noqa: E402

K_VALUES = (1, 3, 5, 8)
MAX_K = max(K_VALUES)

# --method registry. A retriever takes (indexer, question, n) and returns the
# top-n chunks, best first, each a dict with at least a "file" key.
# "embedding" is RepoIndexer's "vector" method (name kept so old result files
# still line up).
Retriever = Callable[[RepoIndexer, str, int], list[dict]]
RETRIEVERS: dict[str, Retriever] = {
    "embedding": lambda idx, q, n: idx.query(q, n_results=n, method="vector"),
    "bm25": lambda idx, q, n: idx.query(q, n_results=n, method="bm25"),
    "hybrid": lambda idx, q, n: idx.query(q, n_results=n, method="hybrid"),
}


CHUNKERS = ("lines", "ast")


def eval_repo_url(repo: str, chunker: str = "lines") -> str:
    """Key the index on url@commit so a new pin never reuses a stale index.

    Function-aware chunks get their own index ("#ast"), so both variants can
    be compared without rebuilding. "lines" keeps the original key.
    """
    url = f"{REPOS[repo]['url']}@{REPOS[repo]['commit']}"
    return url if chunker == "lines" else f"{url}#{chunker}"


def get_indexer(repo: str, source: str | None = None, reindex: bool = False,
                chunker: str = "lines") -> tuple[RepoIndexer, dict]:
    """Return an indexer for the pinned commit, building the index if needed.

    Returns (indexer, stats). stats includes "embed_ms" only when it was built.
    """
    url = eval_repo_url(repo, chunker)
    if not reindex:
        for r in indexer_module.load_persisted_repos():
            if r["repo_url"] == url:
                return r["indexer"], r["stats"]

    idx = RepoIndexer(url, function_aware=(chunker == "ast"))
    with tempfile.TemporaryDirectory() as tmp:
        idx.repo_dir = checkout(repo, os.path.join(tmp, repo), source)
        stats = idx.chunk()
        t = time.perf_counter()
        idx.build_index()
        stats["embed_ms"] = round((time.perf_counter() - t) * 1000)
        idx.save_metadata(stats, summary={})
        idx.chunks = []
        idx.repo_dir = None
    return idx, stats


def score(q: GoldenQuestion, files: list[str]) -> dict:
    """Score one ranked list of chunk files (best first) against q."""
    expected = set(q.expected_files)
    recall = {
        k: len(expected & set(files[:k])) / len(expected) for k in K_VALUES
    }
    first = next((i + 1 for i, f in enumerate(files[:MAX_K]) if f in expected), None)
    return {
        "id": q.id,
        "repo": q.repo,
        "kind": q.kind,
        "question": q.question,
        "expected_files": list(q.expected_files),
        "retrieved_files": files,
        "first_relevant_rank": first,
        "rr": 1 / first if first else 0.0,
        "recall": {str(k): v for k, v in recall.items()},
    }


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    out = {"n": n}
    for k in K_VALUES:
        out[f"recall@{k}"] = sum(r["recall"][str(k)] for r in rows) / n if n else 0.0
    out["mrr"] = sum(r["rr"] for r in rows) / n if n else 0.0
    return out


def markdown_table(groups: dict[str, dict]) -> str:
    head = "| Group | n | " + " | ".join(f"Recall@{k}" for k in K_VALUES) + " | MRR |"
    sep = "|" + "---|" * (len(K_VALUES) + 3)
    lines = [head, sep]
    for name, m in groups.items():
        cells = " | ".join(f"{m[f'recall@{k}']:.2f}" for k in K_VALUES)
        lines.append(f"| {name} | {m['n']} | {cells} | {m['mrr']:.2f} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="File-level retrieval eval (Recall@k, MRR).")
    parser.add_argument("--method", choices=sorted(RETRIEVERS), default="embedding")
    parser.add_argument("--repo", choices=sorted(REPOS), action="append",
                        help="limit to one repo (repeatable); default: all")
    parser.add_argument("--chunker", choices=CHUNKERS, default="lines",
                        help="lines = 60-line windows; ast = function-aware Python chunks")
    parser.add_argument("--reindex", action="store_true", help="rebuild the eval index")
    parser.add_argument("--source", action="append", default=[], metavar="REPO=URL_OR_PATH",
                        help="fetch a repo from somewhere else, e.g. an unpushed local clone")
    args = parser.parse_args()

    retrieve = RETRIEVERS[args.method]
    sources = dict(s.split("=", 1) for s in args.source)
    repos = args.repo or list(REPOS)

    rows, index_stats = [], {}
    for repo in repos:
        idx, stats = get_indexer(repo, sources.get(repo), args.reindex, args.chunker)
        index_stats[repo] = stats
        for q in (q for q in QUESTIONS if q.repo == repo):
            t = time.perf_counter()
            hits = retrieve(idx, q.question, MAX_K)
            row = score(q, [h["file"] for h in hits])
            row["retrieval_ms"] = round((time.perf_counter() - t) * 1000, 1)
            rows.append(row)

    groups = {"all": aggregate(rows)}
    for repo in repos:
        groups[f"repo: {repo}"] = aggregate([r for r in rows if r["repo"] == repo])
    for kind in dict.fromkeys(r["kind"] for r in rows):
        groups[f"kind: {kind}"] = aggregate([r for r in rows if r["kind"] == kind])

    print(f"\n## Retrieval eval: method={args.method} chunker={args.chunker}\n")
    print(markdown_table(groups))
    lat = sorted(r["retrieval_ms"] for r in rows)
    print(f"\nRetrieval latency (in-process, n={len(lat)}): "
          f"p50 {lat[len(lat) // 2]:.1f} ms · max {lat[-1]:.1f} ms")
    print("\nMisses (no expected file in top 3):")
    for r in rows:
        if r["recall"]["3"] == 0:
            print(f"  {r['id']:12} first hit rank={r['first_relevant_rank']}  "
                  f"top3={r['retrieved_files'][:3]}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = args.method if args.chunker == "lines" else f"{args.method}-{args.chunker}"
    path = os.path.join(RESULTS_DIR, f"retrieval_{tag}_{ts}.json")
    with open(path, "w") as f:
        json.dump({
            "method": args.method,
            "chunker": args.chunker,
            "timestamp": ts,
            "repos": {r: REPOS[r] for r in repos},
            "index_stats": index_stats,
            "k_values": list(K_VALUES),
            "metrics": groups,
            "questions": rows,
        }, f, indent=2)
    print(f"\nSaved {os.path.relpath(path)}")


if __name__ == "__main__":
    sys.exit(main())
