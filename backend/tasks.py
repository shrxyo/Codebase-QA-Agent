"""Background indexing: clone -> chunk -> embed (|| summary) -> publish.

Indexing a medium repo takes ~1 min locally and ~10 min on Render's free CPU,
far too long to hold an HTTP request open (clients and proxies time out, and
a replica restart loses the work). So /index only records a build in the DB
and hands it off here; the client polls GET /index/status.

Two runners, one pipeline (run_build):
- REDIS_URL set (docker compose): a Celery task on Redis, run by separate
  worker containers. The 370 MiB embedding peak then never lands on an API
  replica, and indexing capacity scales by adding workers.
- REDIS_URL unset (single instance, e.g. Render free tier): a one-thread pool
  inside the API process. Same code, no extra services.

Run a worker:  celery -A tasks worker --pool=solo --concurrency=1 -Q index
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import db
import indexer as indexer_module
import summarizer
from indexer import RepoIndexer

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "").strip()

# Redis redelivers an unacknowledged job after this long, which is how a job
# survives its worker being killed. Must be longer than the slowest build, or
# a still-running job gets a duplicate (harmless: see run_build, but wasted).
VISIBILITY_TIMEOUT_SECONDS = int(os.getenv("CELERY_VISIBILITY_TIMEOUT", "900"))


def _ms_since(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


def _user_error(exc: Exception, repo_url: str) -> str:
    if isinstance(exc, subprocess.CalledProcessError):
        return f"Failed to clone repository. Check that the URL is correct and the repo is public: {repo_url}"
    if isinstance(exc, subprocess.TimeoutExpired):
        return "Clone timed out after 5 minutes. The repository may be too large or unreachable."
    return str(exc) or type(exc).__name__


def run_build(repo_url: str, build_id: str) -> None:
    """Build one index and publish it, unless a newer build superseded it.

    Safe to run twice for the same build (Redis redelivery after a worker
    crash): each attempt writes its own collection, and db.finish_build lets
    only the first attempt to finish publish; the loser deletes its copy.
    """
    if not db.mark_indexing(repo_url, build_id):
        logger.info("Build %s for %s was superseded; skipping", build_id, repo_url)
        return

    collection = indexer_module.build_collection_name(repo_url, uuid.uuid4().hex)
    idx = RepoIndexer(repo_url, collection_name=collection, persist_bm25_file=False)
    timings: dict[str, int] = {}
    t_start = time.perf_counter()
    try:
        t = time.perf_counter()
        idx.clone()
        timings["clone"] = _ms_since(t)

        t = time.perf_counter()
        stats = idx.chunk()
        timings["chunk"] = _ms_since(t)

        # The summary only needs the README, file listing and entry points,
        # not the embeddings, so run the LLM call while we embed. It's timed
        # inside the worker thread because it overlaps with "embed".
        def timed_summary():
            t0 = time.perf_counter()
            result = summarizer.generate_repo_summary(idx, idx.repo_dir, repo_url)
            timings["summary"] = _ms_since(t0)
            return result

        with ThreadPoolExecutor(max_workers=1) as pool:
            summary_future = pool.submit(timed_summary)
            t = time.perf_counter()
            idx.build_index()
            timings["embed"] = _ms_since(t)
            summary = summary_future.result()

        # Also stamp the collection itself, so it is self-describing.
        idx.save_metadata(stats, summary)
        timings["total"] = _ms_since(t_start)
        bm25 = idx.bm25_payload()
    except Exception as exc:
        logger.exception("Build %s for %s failed", build_id, repo_url)
        idx.delete()
        db.fail_build(repo_url, build_id, _user_error(exc, repo_url))
        return
    finally:
        idx.remove_clone()

    previous = db.finish_build(repo_url, build_id, collection, stats, summary, timings, bm25)
    if previous is False:
        logger.info("Build %s for %s finished after being superseded; discarding", build_id, repo_url)
        indexer_module.delete_collection(collection)
        return
    # Garbage-collect every other collection of this repo: the previous
    # serving one, plus any left half-built by an attempt whose worker was
    # killed (found by the worker-crash test: they were never deleted).
    # Readers switch on their next DB lookup; one already holding the old
    # collection gets a 409 and the client retries. Trade-off: a duplicate
    # attempt of this same build still embedding loses its collection, which
    # is fine because it could no longer publish anyway.
    for name in indexer_module.build_collections_for_repo(repo_url):
        if name != collection:
            indexer_module.delete_collection(name)

    logger.info(
        "Indexed %s (%d files, %d chunks) collection=%s timings_ms=%s",
        repo_url, stats["files"], stats["chunks"], collection, timings,
    )


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

celery_app = None
_local_pool: ThreadPoolExecutor | None = None

if REDIS_URL:
    from celery import Celery
    from celery.signals import worker_ready

    celery_app = Celery("codebaseqa", broker=REDIS_URL)
    celery_app.conf.update(
        task_default_queue="index",
        task_ignore_result=True,  # status lives in Postgres, not a result backend
        # Ack only after the build finishes, so a killed worker's job goes back
        # on the queue instead of being lost.
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        # One job at a time per worker: indexing peaks at ~370 MiB.
        worker_prefetch_multiplier=1,
        broker_transport_options={"visibility_timeout": VISIBILITY_TIMEOUT_SECONDS},
        broker_connection_retry_on_startup=True,
    )

    @celery_app.task(name="tasks.build_repo")
    def build_repo_task(repo_url: str, build_id: str) -> None:
        run_build(repo_url, build_id)

    @worker_ready.connect
    def _warm_worker(**_):
        db.init()
        indexer_module.warm_up()
        logger.info("Index worker ready")


def enqueue(repo_url: str, build_id: str) -> None:
    """Hand a build to whichever runner is configured."""
    global _local_pool
    if celery_app is not None:
        try:
            build_repo_task.delay(repo_url, build_id)
        except Exception as exc:
            db.fail_build(repo_url, build_id, f"Could not queue the indexing job: {exc}")
            raise
        return
    if _local_pool is None:
        # One build at a time, same as one worker: two concurrent embeddings
        # would double the memory peak on a 512 MB instance.
        _local_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="index")
    _local_pool.submit(run_build, repo_url, build_id)
