import logging
import os
import re
import socket
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Load .env before importing modules that read env vars at import time.
load_dotenv()

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import agent  # noqa: E402
import db  # noqa: E402
import indexer as indexer_module  # noqa: E402
import tasks  # noqa: E402
from indexer import RepoIndexer  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Every replica is stateless: the repo registry, build status and BM25 tokens
# live in db.py (Postgres in compose), vectors in ChromaDB (a server in
# compose), and indexing runs in tasks.py (Celery workers in compose). Any
# replica can serve any request, so the load balancer needs no sticky sessions.
REPLICA = socket.gethostname()

# Query-only indexers for recently used repos, keyed by collection name. A
# rebuild publishes a *new* collection name, so a cached entry can never be
# stale: the per-request DB lookup just stops asking for the old key.
_INDEXER_CACHE_SIZE = 8
_indexers: "OrderedDict[str, RepoIndexer]" = OrderedDict()
_indexers_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    # Load the embedding model once, before serving traffic: /chat embeds
    # every question in-process.
    indexer_module.warm_up()
    if tasks.celery_app is None:
        # Single-instance mode: builds run in this process, so any build
        # still marked in progress died with the previous process.
        n = db.fail_in_progress("Indexing was interrupted by a server restart. Please try again.")
        if n:
            logger.info("Marked %d interrupted build(s) as failed", n)
    logger.info(
        "Startup complete on %s (chroma=%s, queue=%s)",
        REPLICA, indexer_module.CHROMA_HOST or "embedded",
        "celery" if tasks.celery_app else "in-process",
    )
    yield


app = FastAPI(title="Codebase-QA-Agent", lifespan=lifespan)

# The frontend sends no cookies or auth headers, so credentials stay off;
# "*" with credentials is rejected by browsers anyway. Set CORS_ORIGINS to a
# comma-separated list (e.g. the Vercel URL) to lock it down.
_cors = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors or ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Only public GitHub repos are in scope, and repo_url goes straight to
# `git clone`, so anything else (file:///etc, ssh://, internal hosts) would be
# a server-side request forgery hole. Owner/repo character rules are GitHub's.
_GITHUB_URL_RE = re.compile(
    r"https://github\.com/(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/"
    r"(?P<repo>[A-Za-z0-9._-]{1,100}?)(?:\.git)?/?"
)


def normalize_repo_url(raw: str) -> str:
    """Canonical https://github.com/owner/repo, or HTTP 400.

    Canonical form also means "…/repo", "…/repo/" and "…/repo.git" share one
    index and one build instead of three.
    """
    m = _GITHUB_URL_RE.fullmatch(raw.strip())
    if not m or m.group("repo") in (".", ".."):
        raise HTTPException(
            status_code=400,
            detail="Only public GitHub repositories are supported: https://github.com/<owner>/<repo>",
        )
    return f"https://github.com/{m.group('owner')}/{m.group('repo')}"


class IndexRequest(BaseModel):
    repo_url: str
    # Rebuild even if this repo is already indexed (e.g. to pick up new commits).
    force: bool = False


class ChatRequest(BaseModel):
    repo_url: str
    question: str
    history: list[dict] = []


@app.get("/health")
async def health():
    """Liveness: the process is up. `replica` shows which container answered."""
    return {"status": "ok", "replica": REPLICA}


@app.get("/ready")
def ready():
    """Readiness: every dependency this replica needs is reachable."""
    checks: dict[str, str] = {}
    try:
        db.ping()
        checks["db"] = "ok"
    except Exception as exc:
        checks["db"] = f"error: {type(exc).__name__}"
    try:
        indexer_module._get_chroma_client().heartbeat()
        checks["chroma"] = "ok"
    except Exception as exc:
        checks["chroma"] = f"error: {type(exc).__name__}"
    if tasks.celery_app is not None:
        try:
            with tasks.celery_app.connection_for_write() as conn:
                conn.ensure_connection(max_retries=1, timeout=2)
            checks["queue"] = "ok"
        except Exception as exc:
            checks["queue"] = f"error: {type(exc).__name__}"
    ok = all(v == "ok" for v in checks.values())
    return JSONResponse({"ready": ok, "replica": REPLICA, "checks": checks}, status_code=200 if ok else 503)


# Plain `def`: the DB calls are blocking. Indexing itself no longer runs in the
# request: it takes ~1 min locally and ~10 min on Render's free CPU, so the
# request only records the build and the client polls /index/status.
@app.post("/index")
def index_repo(body: IndexRequest):
    repo_url = normalize_repo_url(body.repo_url)
    row, started = db.request_build(repo_url, force=body.force)
    if started:
        tasks.enqueue(repo_url, row.build_id)
        logger.info("Queued build %s for %s", row.build_id, repo_url)

    # Already indexed: return it at once (a pre-indexed demo repo opens
    # instantly). With force, report the rebuild instead; the old index
    # keeps serving chat until the new one is published.
    if row.collection is not None and not body.force:
        return _repo_payload(row, cached=True)
    return JSONResponse(_repo_payload(db.get_repo(repo_url), cached=False), status_code=202)


@app.get("/index/status")
def index_status(repo_url: str = Query(...)):
    row = db.get_repo(normalize_repo_url(repo_url))
    if row is None:
        raise HTTPException(status_code=404, detail="No indexing job for this repository.")
    return _repo_payload(row, cached=False)


def _repo_payload(row: db.Repo, cached: bool) -> dict:
    """/index and /index/status body.

    status: "queued" | "indexing" | "indexed" | "failed" (the latest build).
    ready:  whether chat works now (true during a forced rebuild of an
            already indexed repo). timings_ms describe the build that made
            the serving index.
    """
    stats = row.stats or {}
    serving_status = "indexed" if (cached or row.status == "ready") else row.status
    return {
        "status": serving_status,
        "ready": row.collection is not None,
        "error": row.error if row.status == "failed" else None,
        "cached": cached,
        "repo_url": row.repo_url,
        "files_indexed": stats.get("files"),
        "chunks_created": stats.get("chunks"),
        "languages": stats.get("languages"),
        "summary": row.summary,
        "timings_ms": row.timings_ms,
        # Identifies the latest build: concurrent /index calls get the same one.
        "build_id": row.build_id,
        "replica": REPLICA,
    }


def _get_indexer(repo_url: str, collection: str) -> RepoIndexer | None:
    with _indexers_lock:
        idx = _indexers.get(collection)
        if idx is not None:
            _indexers.move_to_end(collection)
            return idx

    # Miss: one BM25 row read plus a Chroma get_collection, ~tens of ms.
    # Done outside the lock so a slow load doesn't block other repos' chats.
    bm25 = db.get_bm25(repo_url, collection)
    if bm25 is None:
        return None  # superseded by a rebuild between the two reads
    try:
        idx = RepoIndexer.attach(repo_url, collection, bm25)
    except Exception:
        logger.warning("Collection %s for %s is gone", collection, repo_url)
        return None

    with _indexers_lock:
        _indexers[collection] = idx
        _indexers.move_to_end(collection)
        while len(_indexers) > _INDEXER_CACHE_SIZE:
            _indexers.popitem(last=False)
    return idx


def _serving_indexer(raw_url: str) -> RepoIndexer:
    repo_url = normalize_repo_url(raw_url)
    row = db.get_repo(repo_url)
    if row is None or row.collection is None:
        raise HTTPException(
            status_code=404,
            detail=f"Repository not indexed. POST /index first: {repo_url}",
        )
    indexer = _get_indexer(repo_url, row.collection)
    if indexer is None:
        # The index was swapped by a rebuild in the last few milliseconds.
        raise HTTPException(status_code=409, detail="This repository's index was just rebuilt. Please retry.")
    return indexer


@app.post("/chat")
def chat(body: ChatRequest):
    indexer = _serving_indexer(body.repo_url)

    # stream_answer is a sync generator; Starlette iterates it in a worker
    # thread, so the blocking retrieval + LLM stream don't block the event loop.
    return StreamingResponse(
        agent.stream_answer(indexer, body.question, body.history),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Replica": REPLICA},
    )


@app.get("/repos")
def list_repos():
    return [
        {
            "repo_url": r.repo_url,
            "files_indexed": (r.stats or {}).get("files"),
            "chunks_created": (r.stats or {}).get("chunks"),
            "languages": (r.stats or {}).get("languages"),
        }
        for r in db.list_ready()
    ]


class RetrievalRequest(BaseModel):
    repo_url: str
    question: str
    method: str = indexer_module.DEFAULT_RETRIEVAL_METHOD
    n_results: int = 8


@app.post("/debug/retrieval")
def debug_retrieval(body: RetrievalRequest):
    """What /chat would retrieve, without calling the LLM (DEBUG_MODE only).

    Used to inspect retrieval and to load-test the replicas: the hosted LLM's
    rate limit (8k tokens/min) makes /chat useless as a throughput benchmark.
    """
    if os.getenv("DEBUG_MODE", "false").lower() != "true":
        raise HTTPException(status_code=404, detail="Not Found")
    if body.method not in indexer_module.RETRIEVAL_METHODS or not 1 <= body.n_results <= 50:
        raise HTTPException(status_code=400, detail="Bad method or n_results")
    indexer = _serving_indexer(body.repo_url)
    t = time.perf_counter()
    hits = indexer.query(body.question, n_results=body.n_results, method=body.method)
    return {
        "replica": REPLICA,
        "retrieval_ms": round((time.perf_counter() - t) * 1000),
        "chunks": [{"file": h["file"], "lines": h["lines"], "score": h.get("score"),
                    "distance": h.get("distance")} for h in hits],
    }
