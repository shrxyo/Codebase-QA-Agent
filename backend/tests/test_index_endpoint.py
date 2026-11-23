"""/index, /index/status and /chat lookup against the shared repo registry.

Uses a throwaway SQLite DB (the same code path as Postgres) and replaces the
job runner, so nothing is cloned or embedded. No lifespan (TestClient isn't
used as a context manager), so no model load.
"""
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

import db
import main
import tasks

URL = "https://github.com/psf/requests"


@pytest.fixture
def client(tmp_path, monkeypatch):
    db.init(f"sqlite:///{tmp_path / 'test.db'}")
    queued = []
    monkeypatch.setattr(tasks, "enqueue", lambda url, build_id: queued.append((url, build_id)))
    return TestClient(main.app), queued


def make_ready(url=URL, collection="repo_abc_1"):
    row, _ = db.request_build(url)
    assert db.mark_indexing(url, row.build_id)
    db.finish_build(
        url, row.build_id, collection,
        stats={"files": 64, "chunks": 292, "languages": ["md", "py"]},
        summary={"name": "requests"},
        timings_ms={"embed": 594802, "total": 597774},
        bm25={"version": 1, "ids": ["a.py::1-60"], "tokens": [["a"]]},
    )
    return row.build_id


def test_already_indexed_repo_is_returned_without_rebuilding(client):
    c, queued = client
    make_ready()

    r = c.post("/index", json={"repo_url": f"  {URL} "})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "indexed" and body["cached"] is True and body["ready"] is True
    assert body["chunks_created"] == 292
    assert body["summary"] == {"name": "requests"}
    assert body["timings_ms"]["total"] == 597774  # timings of the original build
    assert queued == []


def test_new_repo_queues_one_build_and_reports_progress(client):
    c, queued = client
    r = c.post("/index", json={"repo_url": URL})
    assert r.status_code == 202
    assert r.json()["status"] == "queued" and r.json()["ready"] is False
    assert len(queued) == 1

    s = c.get("/index/status", params={"repo_url": URL})
    assert s.status_code == 200 and s.json()["status"] == "queued"

    # Chat isn't possible until the build is published.
    assert c.post("/chat", json={"repo_url": URL, "question": "hi"}).status_code == 404


def test_concurrent_index_requests_share_one_build(client):
    c, queued = client
    first = c.post("/index", json={"repo_url": URL})
    # Same repo, different spellings: one canonical key, so still one build.
    for spelling in (URL + ".git", URL + "/", URL):
        r = c.post("/index", json={"repo_url": spelling})
        assert r.status_code == 202
        assert r.json()["repo_url"] == URL
    assert len(queued) == 1
    assert first.json()["repo_url"] == URL


def test_force_rebuild_keeps_serving_the_old_index(client):
    c, queued = client
    old_build = make_ready()

    r = c.post("/index", json={"repo_url": URL, "force": True})
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "queued"
    assert body["ready"] is True  # chat still works during the rebuild
    assert len(queued) == 1 and queued[0][1] != old_build
    assert db.get_repo(URL).collection == "repo_abc_1"


def test_failed_build_reports_error_and_can_be_retried(client):
    c, queued = client
    c.post("/index", json={"repo_url": URL})
    db.fail_build(URL, queued[0][1], "Failed to clone repository.")

    s = c.get("/index/status", params={"repo_url": URL}).json()
    assert s["status"] == "failed" and s["error"] == "Failed to clone repository."

    assert c.post("/index", json={"repo_url": URL}).status_code == 202
    assert len(queued) == 2


def test_stale_build_is_taken_over(client, monkeypatch):
    c, queued = client
    c.post("/index", json={"repo_url": URL})
    assert c.post("/index", json={"repo_url": URL}).status_code == 202
    assert len(queued) == 1  # fresh: joined

    # Pretend the worker died long ago and the job was lost.
    monkeypatch.setattr(db, "STALE_BUILD_SECONDS", -1)
    c.post("/index", json={"repo_url": URL})
    assert len(queued) == 2


@pytest.mark.parametrize("bad", [
    "file:///etc/passwd",
    "ssh://git@github.com/psf/requests",
    "git@github.com:psf/requests.git",
    "http://github.com/psf/requests",
    "https://github.com.evil.com/psf/requests",
    "https://gitlab.com/psf/requests",
    "https://github.com/psf",
    "https://github.com/psf/requests/tree/main",
    "https://github.com/psf/..",
    "https://169.254.169.254/latest/meta-data",
    "--upload-pack=touch /tmp/x",
])
def test_only_public_github_urls_are_cloned(client, bad):
    c, queued = client
    r = c.post("/index", json={"repo_url": bad})
    assert r.status_code == 400
    assert queued == []


def test_status_for_unknown_repo_is_404(client):
    c, _ = client
    assert c.get("/index/status", params={"repo_url": URL}).status_code == 404


def test_chat_uses_cached_indexer_for_the_serving_collection(client, monkeypatch):
    c, _ = client
    make_ready()
    attached = []

    class FakeIdx:
        pass

    def fake_attach(url, collection, bm25):
        attached.append(collection)
        return FakeIdx()

    monkeypatch.setattr(main.RepoIndexer, "attach", staticmethod(fake_attach))
    monkeypatch.setattr(main, "_indexers", main.OrderedDict())
    monkeypatch.setattr(main.agent, "stream_answer", lambda idx, q, h: iter(["data: [DONE]\n\n"]))

    for _ in range(3):
        r = c.post("/chat", json={"repo_url": URL, "question": "hi"})
        assert r.status_code == 200
    assert attached == ["repo_abc_1"]  # loaded once, then served from the cache


def test_repos_lists_only_ready(client):
    c, _ = client
    make_ready()
    c.post("/index", json={"repo_url": "https://github.com/pallets/flask"})
    assert [r["repo_url"] for r in c.get("/repos").json()] == [URL]


def test_debug_retrieval_is_off_unless_debug_mode(client, monkeypatch):
    c, _ = client
    monkeypatch.setenv("DEBUG_MODE", "false")
    r = c.post("/debug/retrieval", json={"repo_url": URL, "question": "x"})
    assert r.status_code == 404 and r.json()["detail"] == "Not Found"


def test_concurrent_index_responses_share_build_id(client):
    c, queued = client
    ids = {c.post("/index", json={"repo_url": URL}).json()["build_id"] for _ in range(3)}
    assert ids == {queued[0][1]}
