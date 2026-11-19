import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".go", ".rs", ".java", ".rb", ".cpp",
    ".c", ".h", ".cs", ".php", ".swift",
    ".kt", ".md", ".txt", ".yaml", ".yml",
    ".toml", ".json",
}

SKIP_DIRS = {
    "node_modules", ".git", "__pycache__",
    "dist", "build", ".next", "venv",
}

CHUNK_SIZE = 60
CHUNK_OVERLAP = 10
MIN_CHUNK_CHARS = 20
UPSERT_BATCH_SIZE = 100

MAX_REPO_SIZE_MB = 200
CLONE_TIMEOUT_SECONDS = 300

# Where ChromaDB stores its data on disk. Indexed repos survive restarts as
# long as this directory survives (a Docker volume locally; note that Render's
# free tier has no persistent disk, so there it lasts until the next deploy).
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")

_embed_fn = None
_chroma_client = None


def _check_github_repo_size(url: str) -> None:
    """Raises ValueError if the GitHub repo exceeds MAX_REPO_SIZE_MB."""
    match = re.search(r"github\.com[/:]([^/]+)/([^/\.]+?)(?:\.git)?$", url)
    if not match:
        return  # Not a GitHub URL — skip check, let clone proceed

    owner, repo = match.group(1), match.group(2)
    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "RepoSage/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception:
        return  # Network or API error — don't block the clone

    size_kb = data.get("size", 0)
    size_mb = size_kb / 1024
    if size_mb > MAX_REPO_SIZE_MB:
        raise ValueError(
            f"Repository is {size_mb:.0f} MB, which exceeds the {MAX_REPO_SIZE_MB} MB limit. "
            "Try a smaller repository."
        )


def _get_embed_fn():
    global _embed_fn
    if _embed_fn is None:
        _embed_fn = embedding_functions.DefaultEmbeddingFunction()
    return _embed_fn


def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            # Don't send usage telemetry from a demo server.
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client


def _collection_name(repo_url: str) -> str:
    # Deterministic, so re-indexing a repo replaces its collection and a
    # restarted server can find it again.
    return "repo_" + hashlib.sha1(repo_url.encode()).hexdigest()[:16]


def warm_up() -> None:
    """Load the embedding model and run it once.

    Called at app startup so the first /index request doesn't pay for loading
    the ONNX model. The model files themselves are downloaded at Docker build
    time (see Dockerfile), so this only reads them from local disk.
    """
    _get_embed_fn()(["warm up"])
    _get_chroma_client()


def load_persisted_repos() -> list[dict]:
    """Return every fully indexed repo found in the on-disk ChromaDB store.

    Each item: {"repo_url", "indexer", "stats", "summary"}. Collections from an
    indexing run that never finished (no "ready" flag) are deleted.
    """
    client = _get_chroma_client()
    repos = []
    for col in client.list_collections():
        meta = col.metadata or {}
        if not meta.get("ready"):
            logger.info("Deleting incomplete collection %s", col.name)
            client.delete_collection(col.name)
            continue

        indexer = RepoIndexer(meta["repo_url"])
        indexer._collection = client.get_collection(
            name=col.name, embedding_function=_get_embed_fn()
        )
        repos.append({
            "repo_url": meta["repo_url"],
            "indexer": indexer,
            "stats": json.loads(meta["stats"]),
            "summary": json.loads(meta["summary"]),
        })
    return repos


class RepoIndexer:
    """Indexes one repository into its own ChromaDB collection.

    Indexing is split into steps so main.py can time each one and run the
    summary LLM call in parallel with embedding:
        clone() -> chunk() -> build_index() -> save_metadata() -> remove_clone()
    """

    def __init__(self, repo_url: str):
        self.repo_url = repo_url
        self.repo_dir: str | None = None
        self.chunks: list[dict] = []
        self._temp_dir: str | None = None
        self._collection = None

    def clone_and_chunk(self) -> dict:
        self.clone()
        return self.chunk()

    def clone(self) -> None:
        """Shallow-clone the repo into a temp dir (after a size check)."""
        _check_github_repo_size(self.repo_url)

        self._temp_dir = tempfile.mkdtemp()
        self.repo_dir = os.path.join(self._temp_dir, "repo")

        subprocess.run(
            ["git", "clone", "--depth=1", self.repo_url, self.repo_dir],
            check=True,
            timeout=CLONE_TIMEOUT_SECONDS,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def chunk(self) -> dict:
        """Walk the clone and split supported files into line-based chunks."""
        chunks: list[dict] = []
        languages_seen: set[str] = set()

        for root, dirs, files in os.walk(self.repo_dir):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    continue

                abs_path = os.path.join(root, filename)
                rel_path = os.path.relpath(abs_path, self.repo_dir)

                try:
                    with open(abs_path, encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                except OSError:
                    continue

                language = ext.lstrip(".")
                languages_seen.add(language)
                chunks.extend(self._chunk_lines(lines, rel_path, language))

        self.chunks = chunks

        return {
            "files": self._count_unique_files(chunks),
            "chunks": len(chunks),
            "languages": sorted(languages_seen),
        }

    def build_index(self) -> None:
        """Embed all chunks into this repo's collection, replacing any old one."""
        client = _get_chroma_client()
        name = _collection_name(self.repo_url)
        try:
            client.delete_collection(name)
        except Exception:
            pass  # didn't exist

        # "ready" stays False until save_metadata() runs, so a crash mid-index
        # leaves a collection that load_persisted_repos() will discard.
        self._collection = client.create_collection(
            name=name,
            embedding_function=_get_embed_fn(),
            metadata={"repo_url": self.repo_url, "ready": False},
        )

        for batch_start in range(0, len(self.chunks), UPSERT_BATCH_SIZE):
            batch = self.chunks[batch_start: batch_start + UPSERT_BATCH_SIZE]
            self._collection.upsert(
                ids=[f"{c['file']}::{c['lines']}" for c in batch],
                documents=[c["content"] for c in batch],
                metadatas=[
                    {"file": c["file"], "lines": c["lines"], "language": c["language"]}
                    for c in batch
                ],
            )

    def save_metadata(self, stats: dict, summary: dict) -> None:
        """Store stats + summary on the collection and mark it complete.

        ChromaDB metadata values must be scalars, so dicts are stored as JSON.
        """
        self._collection.modify(metadata={
            "repo_url": self.repo_url,
            "ready": True,
            "stats": json.dumps(stats),
            "summary": json.dumps(summary),
        })

    def query(self, question: str, n_results: int = 8) -> list[dict]:
        if self._collection is None:
            return []

        actual_n = min(n_results, self._collection.count())
        if actual_n == 0:
            return []

        results = self._collection.query(
            query_texts=[question],
            n_results=actual_n,
        )

        hits = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            hits.append({
                "content": doc,
                "file": meta["file"],
                "lines": meta["lines"],
                "language": meta["language"],
                "distance": distance,
            })
        return hits

    def remove_clone(self) -> None:
        """Delete the cloned files. Everything needed for chat is in ChromaDB."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        self._temp_dir = None
        self.repo_dir = None
        self.chunks = []

    def delete(self) -> None:
        """Remove the clone and this repo's collection."""
        self.remove_clone()
        try:
            _get_chroma_client().delete_collection(_collection_name(self.repo_url))
        except Exception:
            pass
        self._collection = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chunk_lines(self, lines: list[str], file: str, language: str) -> list[dict]:
        chunks = []
        total = len(lines)
        step = CHUNK_SIZE - CHUNK_OVERLAP
        start = 0

        while start < total:
            end = min(start + CHUNK_SIZE, total)
            content = "".join(lines[start:end])

            if len(content.strip()) >= MIN_CHUNK_CHARS:
                chunks.append({
                    "content": content,
                    "file": file,
                    "lines": f"{start + 1}-{end}",
                    "language": language,
                    "node_type": "chunk",
                })

            if end == total:
                break
            start += step

        return chunks

    def _count_unique_files(self, chunks: list[dict]) -> int:
        return len({c["file"] for c in chunks})
