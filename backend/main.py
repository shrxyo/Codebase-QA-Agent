import logging
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Load .env before importing modules that read env vars at import time.
load_dotenv()

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import agent  # noqa: E402
import indexer as indexer_module  # noqa: E402
import summarizer  # noqa: E402
from indexer import RepoIndexer  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# repo_url -> {"indexer": RepoIndexer, "stats": dict, "summary": dict}
_indexed_repos: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the embedding model once, before serving traffic, and reload any
    # repos indexed before the last restart.
    indexer_module.warm_up()
    for repo in indexer_module.load_persisted_repos():
        _indexed_repos[repo["repo_url"]] = repo
    logger.info("Startup complete: %d repo(s) restored from disk", len(_indexed_repos))
    yield


app = FastAPI(title="RepoSage", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ms_since(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


class IndexRequest(BaseModel):
    repo_url: str


class ChatRequest(BaseModel):
    repo_url: str
    question: str
    history: list[dict] = []


@app.get("/health")
async def health():
    return {"status": "ok"}


# Plain `def` on purpose: cloning, embedding and the summary LLM call are all
# blocking. FastAPI runs sync endpoints in a worker thread, so other requests
# (/health, /chat for another repo) keep being served while this one runs.
@app.post("/index")
def index_repo(body: IndexRequest):
    repo_url = body.repo_url.strip()

    # Clean up any previously indexed version of the same repo
    if repo_url in _indexed_repos:
        _indexed_repos.pop(repo_url)["indexer"].delete()

    indexer = RepoIndexer(repo_url)
    timings: dict[str, int] = {}
    t_start = time.perf_counter()
    try:
        t = time.perf_counter()
        indexer.clone()
        timings["clone"] = _ms_since(t)

        t = time.perf_counter()
        stats = indexer.chunk()
        timings["chunk"] = _ms_since(t)

        # The summary only needs the README, file listing and entry points,
        # not the embeddings, so run the LLM call while we embed. It's timed
        # inside the worker thread because it overlaps with "embed".
        def timed_summary():
            t0 = time.perf_counter()
            result = summarizer.generate_repo_summary(indexer, indexer.repo_dir, repo_url)
            timings["summary"] = _ms_since(t0)
            return result

        with ThreadPoolExecutor(max_workers=1) as pool:
            summary_future = pool.submit(timed_summary)
            t = time.perf_counter()
            indexer.build_index()
            timings["embed"] = _ms_since(t)
            summary = summary_future.result()

        indexer.save_metadata(stats, summary)
        timings["total"] = _ms_since(t_start)
    except subprocess.CalledProcessError:
        indexer.delete()
        raise HTTPException(
            status_code=400,
            detail=f"Failed to clone repository. Check that the URL is correct and the repo is public: {repo_url}",
        )
    except subprocess.TimeoutExpired:
        indexer.delete()
        raise HTTPException(
            status_code=400,
            detail="Clone timed out after 5 minutes. The repository may be too large or unreachable.",
        )
    except Exception as exc:
        indexer.delete()
        raise HTTPException(status_code=400, detail=str(exc))

    # The cloned files aren't needed for chat; everything is in ChromaDB now.
    indexer.remove_clone()
    _indexed_repos[repo_url] = {"indexer": indexer, "stats": stats, "summary": summary}

    logger.info(
        "Indexed %s (%d files, %d chunks) timings_ms=%s",
        repo_url, stats["files"], stats["chunks"], timings,
    )

    return {
        "status": "indexed",
        "repo_url": repo_url,
        "files_indexed": stats["files"],
        "chunks_created": stats["chunks"],
        "languages": stats["languages"],
        "summary": summary,
        "timings_ms": timings,
    }


@app.post("/chat")
async def chat(body: ChatRequest):
    repo_url = body.repo_url.strip()

    if repo_url not in _indexed_repos:
        raise HTTPException(
            status_code=404,
            detail=f"Repository not indexed. POST /index first: {repo_url}",
        )

    indexer = _indexed_repos[repo_url]["indexer"]

    # stream_answer is a sync generator; Starlette iterates it in a worker
    # thread, so the blocking retrieval + LLM stream don't block the event loop.
    return StreamingResponse(
        agent.stream_answer(indexer, body.question, body.history),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/repos")
async def list_repos():
    return [
        {
            "repo_url": url,
            "files_indexed": entry["stats"]["files"],
            "chunks_created": entry["stats"]["chunks"],
            "languages": entry["stats"]["languages"],
        }
        for url, entry in _indexed_repos.items()
    ]
