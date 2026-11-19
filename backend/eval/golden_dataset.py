"""Golden dataset for retrieval and answer evals.

24 questions over 2 repos, each pinned to a commit so results are reproducible:
  - psf/requests: a small, well-known Python library (64 files, ~335 chunks)
  - this repo (Codebase-QA-Agent), pinned before the eval harness existed

Every question has the file(s) a correct answer needs (`expected_files`, paths
relative to the repo root, the same form RepoIndexer stores in chunk metadata)
and `evidence`: (file, substring) pairs that prove why each file is expected.
`python -m eval.golden_dataset --verify` checks every piece of evidence against
a fresh checkout at the pinned commit, so the labels can't silently go stale.

Question kinds (3 per kind per repo):
  definition  "where is X defined", phrased WITHOUT the identifier
  flow        "how does Y work", usually needs 1-3 files
  config      "which setting / env var / constant controls Z"
  identifier  uses the exact identifier (function, constant), which is where
              embedding-only retrieval is expected to be weakest
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass

REPOS: dict[str, dict[str, str]] = {
    "requests": {
        "url": "https://github.com/psf/requests",
        "commit": "3c7baaa05ab4aa49cc819ce3a819d260de19c157",  # 2026-10-26
    },
    "codebase-qa-agent": {
        "url": "https://github.com/shrxyo/Codebase-QA-Agent",
        "commit": "918a8540eb440a6a9826554a734f1493c2c7545d",  # latency instrumentation
    },
}

KINDS = ("definition", "flow", "config", "identifier")


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    repo: str
    kind: str
    question: str
    expected_files: tuple[str, ...]
    evidence: tuple[tuple[str, str], ...]


Q = GoldenQuestion

QUESTIONS: list[GoldenQuestion] = [
    # ------------------------------------------------------------------ requests
    Q("req-def-1", "requests", "definition",
      "Where is the case-insensitive dictionary that stores HTTP headers implemented?",
      ("src/requests/structures.py",),
      (("src/requests/structures.py", "class CaseInsensitiveDict"),)),
    Q("req-def-2", "requests", "definition",
      "Where are the exception classes for connect and read timeouts defined?",
      ("src/requests/exceptions.py",),
      (("src/requests/exceptions.py", "class ConnectTimeout"),
       ("src/requests/exceptions.py", "class ReadTimeout"))),
    Q("req-def-3", "requests", "definition",
      "Where are the human-readable names for HTTP status codes, like codes.ok or codes.not_found, defined?",
      ("src/requests/status_codes.py",),
      (("src/requests/status_codes.py", '200: ("ok"'),
       ("src/requests/status_codes.py", '"not_found"'))),

    Q("req-flow-1", "requests", "flow",
      "How does a call to requests.get() end up being sent over the network by the transport adapter?",
      ("src/requests/api.py", "src/requests/sessions.py", "src/requests/adapters.py"),
      (("src/requests/api.py", "with sessions.Session() as session"),
       ("src/requests/sessions.py", "adapter = self.get_adapter(url=request.url)"),
       ("src/requests/adapters.py", "class HTTPAdapter(BaseAdapter)"))),
    Q("req-flow-2", "requests", "flow",
      "How are redirects followed, and when is the Authorization header dropped during a redirect?",
      ("src/requests/sessions.py",),
      (("src/requests/sessions.py", "def resolve_redirects("),
       ("src/requests/sessions.py", 'del headers["Authorization"]'))),
    Q("req-flow-3", "requests", "flow",
      "How does HTTP Digest authentication retry the request after the server responds with a 401 challenge?",
      ("src/requests/auth.py",),
      (("src/requests/auth.py", "class HTTPDigestAuth"),
       ("src/requests/auth.py", "def handle_401("))),

    Q("req-cfg-1", "requests", "config",
      "What is the default maximum number of redirects a session will follow, and where is it set?",
      ("src/requests/models.py", "src/requests/sessions.py"),
      (("src/requests/models.py", "DEFAULT_REDIRECT_LIMIT: int = 30"),
       ("src/requests/sessions.py", "self.max_redirects = DEFAULT_REDIRECT_LIMIT"))),
    Q("req-cfg-2", "requests", "config",
      "Which environment variables control the CA certificate bundle used for TLS verification?",
      ("src/requests/sessions.py",),
      (("src/requests/sessions.py", 'os.environ.get("REQUESTS_CA_BUNDLE")'),
       ("src/requests/sessions.py", 'os.environ.get("CURL_CA_BUNDLE")'))),
    Q("req-cfg-3", "requests", "config",
      "What are the default connection pool size and retry count for the HTTP adapter?",
      ("src/requests/adapters.py",),
      (("src/requests/adapters.py", "DEFAULT_POOLSIZE = 10"),
       ("src/requests/adapters.py", "DEFAULT_RETRIES = 0"))),

    Q("req-id-1", "requests", "identifier",
      "What does should_strip_auth check?",
      ("src/requests/sessions.py",),
      (("src/requests/sessions.py", "def should_strip_auth("),)),
    Q("req-id-2", "requests", "identifier",
      "What is ITER_CHUNK_SIZE and which method uses it?",
      ("src/requests/models.py",),
      (("src/requests/models.py", "ITER_CHUNK_SIZE: int = 512"),
       ("src/requests/models.py", "chunk_size: int = ITER_CHUNK_SIZE"))),
    Q("req-id-3", "requests", "identifier",
      "What does dispatch_hook do?",
      ("src/requests/hooks.py",),
      (("src/requests/hooks.py", "def dispatch_hook("),)),

    # --------------------------------------------------------- codebase-qa-agent
    Q("cqa-def-1", "codebase-qa-agent", "definition",
      "Where is the HTTP endpoint that answers chat questions defined?",
      ("backend/main.py",),
      (("backend/main.py", '@app.post("/chat")'),)),
    Q("cqa-def-2", "codebase-qa-agent", "definition",
      "Which frontend component shows how long indexing took on the repo summary card?",
      ("frontend/src/components/RepoSummaryCard.jsx",),
      (("frontend/src/components/RepoSummaryCard.jsx", "indexed in {formatMs(stats.timings.total)}"),)),
    Q("cqa-def-3", "codebase-qa-agent", "definition",
      "Where is the system prompt that tells the model to cite files defined?",
      ("backend/agent.py",),
      (("backend/agent.py", "Always cite files using [filename:lines] notation"),)),

    Q("cqa-flow-1", "codebase-qa-agent", "flow",
      "How does indexing run the repo summary at the same time as embedding?",
      ("backend/main.py",),
      (("backend/main.py", "with ThreadPoolExecutor(max_workers=1) as pool"),)),
    Q("cqa-flow-2", "codebase-qa-agent", "flow",
      "How is a chat answer streamed from the server to the browser, from retrieval to the final [DONE] event?",
      ("backend/agent.py", "frontend/src/lib/api.js"),
      (("backend/agent.py", "def stream_answer("),
       ("frontend/src/lib/api.js", "if (raw === '[DONE]')"))),
    Q("cqa-flow-3", "codebase-qa-agent", "flow",
      "How does the server get back the repos it had already indexed after it restarts?",
      ("backend/indexer.py", "backend/main.py"),
      (("backend/indexer.py", "def load_persisted_repos()"),
       ("backend/main.py", "indexer_module.load_persisted_repos()"))),

    Q("cqa-cfg-1", "codebase-qa-agent", "config",
      "Which environment variable chooses between the local and the hosted LLM provider?",
      ("backend/llm_client.py",),
      (("backend/llm_client.py", 'os.getenv("LLM_PROVIDER", "ollama")'),)),
    Q("cqa-cfg-2", "codebase-qa-agent", "config",
      "Where is the directory that ChromaDB saves the index to configured, both in code and in Docker?",
      ("backend/indexer.py", "docker-compose.yml"),
      (("backend/indexer.py", 'CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR"'),
       ("docker-compose.yml", "CHROMA_PERSIST_DIR: /data/chroma"))),
    Q("cqa-cfg-3", "codebase-qa-agent", "config",
      "How is the backend configured for deployment on Render, and is the reranker turned on there?",
      ("render.yaml",),
      (("render.yaml", "plan: free"),
       ("render.yaml", "key: RERANKER_ENABLED"))),

    Q("cqa-id-1", "codebase-qa-agent", "identifier",
      "What does get_request_options return, and for which provider?",
      ("backend/llm_client.py",),
      (("backend/llm_client.py", "def get_request_options()"),
       ("backend/llm_client.py", '"reasoning_effort": "low"'))),
    Q("cqa-id-2", "codebase-qa-agent", "identifier",
      "What thresholds does _complexity_label use?",
      ("backend/summarizer.py",),
      (("backend/summarizer.py", "def _complexity_label("),)),
    Q("cqa-id-3", "codebase-qa-agent", "identifier",
      "What is MAX_REPO_SIZE_MB and where is it enforced?",
      ("backend/indexer.py",),
      (("backend/indexer.py", "MAX_REPO_SIZE_MB = 200"),
       ("backend/indexer.py", "if size_mb > MAX_REPO_SIZE_MB"))),
]


def questions_for(repo: str) -> list[GoldenQuestion]:
    return [q for q in QUESTIONS if q.repo == repo]


def checkout(repo: str, dest: str, source: str | None = None) -> str:
    """Fetch exactly the pinned commit of `repo` into `dest` (shallow).

    `source` overrides the clone URL, e.g. a local path for a commit that
    hasn't been pushed yet. Returns `dest`.
    """
    info = REPOS[repo]
    os.makedirs(dest, exist_ok=True)

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", dest, *args], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    git("init", "-q")
    git("fetch", "-q", "--depth=1", source or info["url"], info["commit"])
    git("checkout", "-q", "FETCH_HEAD")
    return dest


def verify(repo_dirs: dict[str, str]) -> list[str]:
    """Check the dataset against checkouts. Returns a list of problems."""
    problems = []
    seen_ids = set()
    for q in QUESTIONS:
        if q.id in seen_ids:
            problems.append(f"{q.id}: duplicate id")
        seen_ids.add(q.id)
        if q.kind not in KINDS:
            problems.append(f"{q.id}: unknown kind {q.kind!r}")
        if q.repo not in repo_dirs:
            continue
        evidenced = {f for f, _ in q.evidence}
        for f in q.expected_files:
            if f not in evidenced:
                problems.append(f"{q.id}: no evidence for expected file {f}")
        for f, needle in q.evidence:
            path = os.path.join(repo_dirs[q.repo], f)
            try:
                with open(path, encoding="utf-8") as fh:
                    text = fh.read()
            except OSError:
                problems.append(f"{q.id}: {f} does not exist")
                continue
            if needle not in text:
                problems.append(f"{q.id}: {f} does not contain {needle!r}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--verify", action="store_true",
                        help="check evidence against fresh checkouts at the pinned commits")
    parser.add_argument("--source", action="append", default=[], metavar="REPO=URL_OR_PATH",
                        help="override where a repo is fetched from (e.g. an unpushed local clone)")
    args = parser.parse_args()

    if not args.verify:
        for q in QUESTIONS:
            print(f"{q.id:12} {q.kind:10} {q.question}\n{'':24}-> {', '.join(q.expected_files)}")
        return

    sources = dict(s.split("=", 1) for s in args.source)
    with tempfile.TemporaryDirectory() as tmp:
        dirs = {r: checkout(r, os.path.join(tmp, r), sources.get(r)) for r in REPOS}
        problems = verify(dirs)
    for p in problems:
        print("FAIL", p)
    print(f"{len(QUESTIONS)} questions, {len(problems)} problems")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
