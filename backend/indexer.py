import os
import shutil
import subprocess
import tempfile
import uuid

import chromadb
from chromadb.utils import embedding_functions

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

_embed_fn = None


def _get_embed_fn():
    global _embed_fn
    if _embed_fn is None:
        _embed_fn = embedding_functions.DefaultEmbeddingFunction()
    return _embed_fn


class RepoIndexer:
    def __init__(self):
        self.repo_dir: str | None = None
        self.chunks: list[dict] = []
        self._temp_dir: str | None = None
        self._collection = None

    def index_repo(self, url: str) -> dict:
        self._temp_dir = tempfile.mkdtemp()
        self.repo_dir = os.path.join(self._temp_dir, "repo")

        subprocess.run(
            ["git", "clone", "--depth=1", url, self.repo_dir],
            check=True,
            timeout=60,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

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
        self._build_collection(chunks)

        return {
            "files": self._count_unique_files(chunks),
            "chunks": len(chunks),
            "languages": sorted(languages_seen),
        }

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

    def cleanup(self) -> None:
        self._collection = None
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        self._temp_dir = None
        self.repo_dir = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_collection(self, chunks: list[dict]) -> None:
        client = chromadb.Client()
        collection_name = f"repo_{uuid.uuid4().hex}"
        self._collection = client.create_collection(
            name=collection_name,
            embedding_function=_get_embed_fn(),
        )

        for batch_start in range(0, len(chunks), UPSERT_BATCH_SIZE):
            batch = chunks[batch_start: batch_start + UPSERT_BATCH_SIZE]
            self._collection.upsert(
                ids=[f"{c['file']}::{c['lines']}" for c in batch],
                documents=[c["content"] for c in batch],
                metadatas=[
                    {"file": c["file"], "lines": c["lines"], "language": c["language"]}
                    for c in batch
                ],
            )

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
