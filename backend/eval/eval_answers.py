"""Answer eval: runs the real agent + LLM over the golden dataset.

Reports TTFT and total latency (p50/p95), using the server-side `timings`
event that agent.stream_answer emits (same numbers the UI shows), and
citation validity: the share of [file:lines] citations that point at a file
in the index and at lines retrieved for that answer (agent.verify_citations,
the same check behind the UI's confidence score).

Calls agent.stream_answer directly (no HTTP), with the eval index built by
eval_retrieval (pinned commits). Retrieval is whatever the agent uses, so
compare methods with eval_retrieval; this script is about the answer path.

Groq's free tier allows ~8k tokens/min and one question uses ~3.6k, so
questions are spaced --sleep seconds apart (default 35). Otherwise 429 retries
show up as fake multi-second TTFTs. 24 questions take ~15 minutes.

Usage (from backend/, ideally inside the Docker container to match the
latency baseline):
  python -m eval.eval_answers
  python -m eval.eval_answers --repo requests --limit 3 --sleep 0   # smoke test
  python -m eval.eval_answers --rescore eval/results/answers_<ts>.json  # no LLM calls
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from eval.eval_retrieval import RESULTS_DIR, get_indexer  # noqa: E402  (sets CHROMA_PERSIST_DIR)
from eval.golden_dataset import QUESTIONS, REPOS  # noqa: E402

import agent  # noqa: E402
import indexer as indexer_module  # noqa: E402
import llm_client  # noqa: E402


def validate_citations(answer: str, sources: list[dict], indexed_files) -> dict:
    """Score an answer's [file:lines] citations with the app's own checker.

    Same function the /chat stream uses for its `confidence` event, so the
    eval and the UI share one definition of "valid".
    """
    return agent.verify_citations(answer, sources, indexed_files)


def citation_summary(rows: list[dict]) -> dict:
    # Skip failed questions (e.g. 429s): no answer is not an uncited answer.
    checked = [r["citations"] for r in rows if r.get("citations") is not None and not r.get("error")]
    total = sum(c["total"] for c in checked)
    return {
        # Pooled: valid citations / all citations across the run.
        "citation_validity": sum(c["valid"] for c in checked) / total if total else None,
        "citations_checked": total,
        # Mean of the per-answer `confidence` score (0 for an answer with no
        # citations), i.e. what a user sees on average.
        "mean_confidence": sum(c["score"] for c in checked) / len(checked) if checked else None,
        "answers_with_citations": sum(1 for c in checked if c["total"]),
        "answers_checked": len(checked),
    }


def print_citation_summary(c: dict) -> None:
    if c["citation_validity"] is None:
        print(f"\nCitation validity: n/a (0 citations in {c['answers_checked']} answers)")
    else:
        print(f"\nCitation validity: {c['citation_validity']:.0%} of {c['citations_checked']} citations")
    print(f"Answers with >=1 citation: {c['answers_with_citations']}/{c['answers_checked']}"
          + (f" · mean confidence {c['mean_confidence']:.2f}" if c["mean_confidence"] is not None else ""))


def rescore(path: str) -> None:
    """Re-check the citations of a saved run without calling the LLM.

    Uses the saved answers + sources and the eval index's file lists, and
    writes the result back into the same JSON.
    """
    with open(path) as f:
        data = json.load(f)
    files = {repo: get_indexer(repo)[0].indexed_files() for repo in {r["repo"] for r in data["questions"]}}
    for r in data["questions"]:
        r["citations"] = validate_citations(r["answer"], r["sources"], files[r["repo"]])
        c = r["citations"]
        print(f"{r['id']:12} {c['valid']}/{c['total']}  {c['reason']}")
    data["summary"].update(citation_summary(data["questions"]))
    print_citation_summary(data["summary"])
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nUpdated {os.path.relpath(path)}")


def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolated percentile (same as numpy's default)."""
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * p / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def run_question(indexer, question: str) -> dict:
    """Drive the SSE generator and collect its events."""
    answer, sources, timings, error = [], [], None, None
    try:
        for line in agent.stream_answer(indexer, question, history=[]):
            if not line.startswith("data: "):
                continue
            raw = line[6:].strip()
            if raw == "[DONE]":
                break
            event = json.loads(raw)
            if event["type"] == "text":
                answer.append(event["content"])
            elif event["type"] == "sources":
                sources = event["sources"]
            elif event["type"] == "timings":
                timings = {k: v for k, v in event.items() if k != "type"}
    except Exception as e:  # e.g. Groq 429 after the SDK's retries
        error = f"{type(e).__name__}: {e}"
    return {"answer": "".join(answer), "sources": sources, "timings": timings, "error": error}


def fmt_ms(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v / 1000:.2f} s" if v >= 1000 else f"{v:.0f} ms"


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer eval: latency + citation validity.")
    parser.add_argument("--repo", choices=sorted(REPOS), action="append")
    parser.add_argument("--limit", type=int, help="max questions per repo")
    parser.add_argument("--sleep", type=float, default=35.0, help="seconds between questions")
    parser.add_argument("--rescore", metavar="RESULTS_JSON",
                        help="re-check citations in a saved run (no LLM calls) and exit")
    parser.add_argument("--source", action="append", default=[], metavar="REPO=URL_OR_PATH",
                        help="where to fetch a repo if its eval index must be built")
    args = parser.parse_args()
    if args.rescore:
        return rescore(args.rescore)

    sources = dict(s.split("=", 1) for s in args.source)
    repos = args.repo or list(REPOS)
    # Load the embedding model first, as the server does at startup, so the
    # first question's retrieval time doesn't include model loading.
    indexer_module.warm_up()
    rows = []
    for repo in repos:
        idx, _ = get_indexer(repo, sources.get(repo))
        files = idx.indexed_files()
        qs = [q for q in QUESTIONS if q.repo == repo][: args.limit]
        for q in qs:
            if rows and args.sleep:
                time.sleep(args.sleep)
            r = run_question(idx, q.question)
            r.update({"id": q.id, "repo": repo, "kind": q.kind, "question": q.question,
                      "citations": validate_citations(r["answer"], r["sources"], files)})
            rows.append(r)
            t = r["timings"] or {}
            c = r["citations"]
            print(f"{q.id:12} ttft={fmt_ms(t.get('ttft_ms')):>8} total={fmt_ms(t.get('total_ms')):>8}"
                  f"  citations={c['valid']}/{c['total']}"
                  + (f"  ERROR {r['error']}" if r["error"] else ""), flush=True)

    ok = [r for r in rows if r["timings"] and not r["error"]]
    ttft = [r["timings"]["ttft_ms"] for r in ok if r["timings"]["ttft_ms"] is not None]
    total = [r["timings"]["total_ms"] for r in ok]
    retrieval = [r["timings"]["retrieval_ms"] for r in ok]

    summary = {
        "n": len(rows),
        "errors": len(rows) - len(ok),
        "retrieval_ms": {"p50": percentile(retrieval, 50), "p95": percentile(retrieval, 95)},
        "ttft_ms": {"p50": percentile(ttft, 50), "p95": percentile(ttft, 95)},
        "total_ms": {"p50": percentile(total, 50), "p95": percentile(total, 95)},
        **citation_summary(rows),
    }

    print(f"\n## Answer eval: model={llm_client.get_model_name()}, n={len(rows)}, errors={summary['errors']}\n")
    print("| Metric | p50 | p95 |\n|---|---|---|")
    for key, label in (("retrieval_ms", "Retrieval"), ("ttft_ms", "TTFT"), ("total_ms", "Total")):
        print(f"| {label} | {fmt_ms(summary[key]['p50'])} | {fmt_ms(summary[key]['p95'])} |")
    print_citation_summary(summary)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(RESULTS_DIR, f"answers_{ts}.json")
    with open(path, "w") as f:
        json.dump({
            "timestamp": ts,
            "provider": os.getenv("LLM_PROVIDER", "ollama"),
            "model": llm_client.get_model_name(),
            "sleep_s": args.sleep,
            "repos": {r: REPOS[r] for r in repos},
            "summary": summary,
            "questions": rows,
        }, f, indent=2)
    print(f"\nSaved {os.path.relpath(path)}")


if __name__ == "__main__":
    sys.exit(main())
