"""/index returns an already-indexed repo instead of re-embedding it.

No lifespan (TestClient isn't used as a context manager), so no model load.
"""
import pytest
from fastapi.testclient import TestClient

import main

URL = "https://github.com/psf/requests"


class FakeIndexer:
    deleted = False

    def delete(self):
        self.deleted = True


@pytest.fixture
def client(monkeypatch):
    fake = FakeIndexer()
    monkeypatch.setattr(main, "_indexed_repos", {URL: {
        "indexer": fake,
        "stats": {"files": 64, "chunks": 292, "languages": ["md", "py"]},
        "summary": {"name": "requests"},
        "timings_ms": {"embed": 594802, "total": 597774},
    }})
    return TestClient(main.app), fake


def test_already_indexed_repo_is_returned_without_rebuilding(client, monkeypatch):
    c, fake = client

    def no_rebuild(*args, **kwargs):
        raise AssertionError("RepoIndexer must not be built for a cached repo")
    monkeypatch.setattr(main, "RepoIndexer", no_rebuild)

    r = c.post("/index", json={"repo_url": f"  {URL} "})
    assert r.status_code == 200
    body = r.json()
    assert body["cached"] is True
    assert body["chunks_created"] == 292
    assert body["summary"] == {"name": "requests"}
    assert body["timings_ms"]["total"] == 597774  # timings of the original build
    assert not fake.deleted


def test_force_rebuilds(client, monkeypatch):
    c, fake = client

    class Boom:
        def __init__(self, url):
            raise RuntimeError("rebuild started")
    monkeypatch.setattr(main, "RepoIndexer", Boom)

    # Reaching the rebuild path means the old index is deleted and a new
    # indexer is constructed (which raises here, so nothing is cloned).
    with pytest.raises(RuntimeError, match="rebuild started"):
        c.post("/index", json={"repo_url": URL, "force": True})
    assert fake.deleted
