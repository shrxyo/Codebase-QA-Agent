import os
import subprocess

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from indexer import RepoIndexer

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

app = FastAPI(title="RepoSage")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# repo_url -> {"indexer": RepoIndexer, "stats": dict}
_indexed_repos: dict[str, dict] = {}


class IndexRequest(BaseModel):
    repo_url: str


@app.get("/health")
async def health():
    client = chromadb.Client()
    embed_fn = embedding_functions.DefaultEmbeddingFunction()
    col = client.create_collection(name="health_check", embedding_function=embed_fn)

    col.upsert(
        ids=["test_chunk"],
        documents=["def hello(): return 'world'"],
        metadatas=[{"file": "test.py", "lines": "1-1", "language": "py"}],
    )

    results = col.query(query_texts=["hello function"], n_results=1)
    retrieved = len(results["documents"][0])

    client.delete_collection("health_check")

    return {
        "status": "ok",
        "rag_pipeline": "operational",
        "retrieved_chunks": retrieved,
    }


@app.post("/index")
async def index_repo(body: IndexRequest):
    repo_url = body.repo_url.strip()

    # Clean up any previously indexed version of the same repo
    if repo_url in _indexed_repos:
        _indexed_repos[repo_url]["indexer"].cleanup()
        del _indexed_repos[repo_url]

    indexer = RepoIndexer()
    try:
        stats = indexer.index_repo(repo_url)
    except subprocess.CalledProcessError:
        indexer.cleanup()
        raise HTTPException(
            status_code=400,
            detail=f"Failed to clone repository. Check that the URL is correct and the repo is public: {repo_url}",
        )
    except subprocess.TimeoutExpired:
        indexer.cleanup()
        raise HTTPException(
            status_code=400,
            detail="Clone timed out after 60 seconds. The repository may be too large or unreachable.",
        )
    except Exception as exc:
        indexer.cleanup()
        raise HTTPException(status_code=400, detail=str(exc))

    _indexed_repos[repo_url] = {"indexer": indexer, "stats": stats}

    return {
        "status": "indexed",
        "repo_url": repo_url,
        "files_indexed": stats["files"],
        "chunks_created": stats["chunks"],
        "languages": stats["languages"],
    }


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
