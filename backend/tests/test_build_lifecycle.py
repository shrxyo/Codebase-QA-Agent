"""Build state machine (db.py) and the worker pipeline (tasks.run_build).

These are the guarantees that make it safe to run many API replicas and
workers: one build per repo at a time, stale/duplicate jobs can't publish,
and a rebuild swaps the serving index atomically.
"""
import subprocess

import pytest

import db
import tasks

URL = "https://github.com/psf/requests"
STATS = {"files": 1, "chunks": 1, "languages": ["py"]}
BM25 = {"version": 1, "ids": ["a.py::1-60"], "tokens": [["a"]]}


@pytest.fixture(autouse=True)
def fresh_db(tmp_path):
    db.init(f"sqlite:///{tmp_path / 'test.db'}")


def finish(build_id, collection):
    return db.finish_build(URL, build_id, collection, STATS, {}, {}, BM25)


def test_superseded_build_cannot_publish():
    first, _ = db.request_build(URL)
    assert db.mark_indexing(URL, first.build_id)
    db.fail_build(URL, first.build_id, "boom")
    second, started = db.request_build(URL)
    assert started and second.build_id != first.build_id

    # The old job finishing late must not overwrite the new build.
    assert db.mark_indexing(URL, first.build_id) is False
    assert finish(first.build_id, "old_col") is False
    assert db.get_repo(URL).collection is None


def test_duplicate_run_of_one_job_publishes_once():
    row, _ = db.request_build(URL)
    assert db.mark_indexing(URL, row.build_id)
    assert db.mark_indexing(URL, row.build_id)  # redelivered copy also runs
    assert finish(row.build_id, "col_a") is None  # first to finish wins
    assert finish(row.build_id, "col_b") is False  # loser deletes its copy
    assert db.get_repo(URL).collection == "col_a"


def test_late_failure_of_duplicate_attempt_keeps_repo_ready():
    row, _ = db.request_build(URL)
    db.mark_indexing(URL, row.build_id)
    finish(row.build_id, "col_a")
    db.fail_build(URL, row.build_id, "duplicate attempt lost its collection")
    got = db.get_repo(URL)
    assert got.status == "ready" and got.error is None and got.collection == "col_a"


def test_rebuild_returns_previous_collection_for_cleanup():
    row, _ = db.request_build(URL)
    db.mark_indexing(URL, row.build_id)
    finish(row.build_id, "col_v1")

    row2, started = db.request_build(URL, force=True)
    assert started
    assert db.get_repo(URL).collection == "col_v1"  # still serving
    db.mark_indexing(URL, row2.build_id)
    assert finish(row2.build_id, "col_v2") == "col_v1"
    assert db.get_repo(URL).collection == "col_v2"
    assert db.get_bm25(URL, "col_v1") is None  # tokens follow the collection
    assert db.get_bm25(URL, "col_v2") == BM25


def test_fail_in_progress_only_touches_unfinished_builds():
    row, _ = db.request_build(URL)
    db.mark_indexing(URL, row.build_id)
    finish(row.build_id, "col")
    db.request_build("https://github.com/pallets/flask")
    assert db.fail_in_progress("restart") == 1
    assert db.get_repo(URL).status == "ready"


# --- worker pipeline -------------------------------------------------------

class FakeIndexer:
    instances = []
    clone_error = None

    def __init__(self, repo_url, collection_name=None, persist_bm25_file=True):
        self.repo_url, self.collection_name = repo_url, collection_name
        self.repo_dir, self.chunks = "/tmp/x", []
        self.deleted = self.cloned = False
        FakeIndexer.instances.append(self)

    def clone(self):
        self.cloned = True
        if FakeIndexer.clone_error:
            raise FakeIndexer.clone_error

    def chunk(self):
        return dict(STATS)

    def build_index(self):
        pass

    def save_metadata(self, stats, summary):
        pass

    def bm25_payload(self):
        return BM25

    def remove_clone(self):
        pass

    def delete(self):
        self.deleted = True


@pytest.fixture
def fake_pipeline(monkeypatch):
    FakeIndexer.instances, FakeIndexer.clone_error = [], None
    deleted = []
    monkeypatch.setattr(tasks, "RepoIndexer", FakeIndexer)
    monkeypatch.setattr(tasks.summarizer, "generate_repo_summary", lambda *a: {"name": "x"})
    monkeypatch.setattr(tasks.indexer_module, "delete_collection", deleted.append)
    # Pretend Chroma holds every collection a FakeIndexer created, plus one
    # orphan left by a killed attempt.
    monkeypatch.setattr(tasks.indexer_module, "build_collections_for_repo",
                        lambda url: ["orphan"] + [i.collection_name for i in FakeIndexer.instances])
    return deleted


def test_run_build_publishes_and_deletes_previous(fake_pipeline):
    row, _ = db.request_build(URL)
    tasks.run_build(URL, row.build_id)
    first = db.get_repo(URL)
    assert first.status == "ready" and first.collection == FakeIndexer.instances[0].collection_name
    assert set(first.timings_ms) >= {"clone", "chunk", "embed", "summary", "total"}

    assert fake_pipeline == ["orphan"]  # killed attempt's leftovers collected
    fake_pipeline.clear()

    row2, _ = db.request_build(URL, force=True)
    tasks.run_build(URL, row2.build_id)
    # Old serving collection collected; the new one kept.
    assert sorted(fake_pipeline) == sorted(["orphan", first.collection])
    assert db.get_repo(URL).collection not in fake_pipeline


def test_run_build_skips_superseded_job(fake_pipeline):
    row, _ = db.request_build(URL)
    db.fail_build(URL, row.build_id, "x")
    db.request_build(URL)  # newer build
    tasks.run_build(URL, row.build_id)
    assert FakeIndexer.instances == []  # never cloned


def test_run_build_failure_is_recorded_for_the_client(fake_pipeline):
    FakeIndexer.clone_error = subprocess.CalledProcessError(128, "git")
    row, _ = db.request_build(URL)
    tasks.run_build(URL, row.build_id)
    got = db.get_repo(URL)
    assert got.status == "failed" and "Failed to clone repository" in got.error
    assert FakeIndexer.instances[0].deleted
