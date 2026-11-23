"""Shared repo registry: which repos exist, their build status, and the data
every API replica needs to serve /chat for them.

This replaces the per-process `_indexed_repos` dict, which made horizontal
scaling impossible (a repo indexed on replica A was a 404 on replica B).

- DATABASE_URL set (docker compose): Postgres, shared by every API replica
  and Celery worker.
- DATABASE_URL unset (single instance, e.g. Render free tier): a SQLite file
  next to the ChromaDB data, so the same code path runs everywhere.

Sync SQLAlchemy on purpose: the Celery worker is sync, /index is a sync
endpoint (runs in FastAPI's thread pool), and /chat's generator already runs
in a worker thread. Each call here is one primary-key lookup or one small
UPDATE (~1 ms), so an async driver would add a second code path for nothing.

Build lifecycle (one row per repo):
    status   queued -> indexing -> ready | failed
    build_id the latest requested build. A worker only writes its result if
             the row still has its build_id, so a stale or duplicate job
             can never overwrite a newer one.
    collection  the ChromaDB collection that serves chat. Switched in the
             same transaction that marks a build ready (blue/green), so a
             forced rebuild never takes chat down and readers never see a
             half-built index.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import JSON, DateTime, String, Text, create_engine, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, deferred, mapped_column, sessionmaker

# A build still "queued"/"indexing" after this long is treated as dead (its
# worker crashed and the job was lost), so a new /index can take it over.
# Indexing psf/requests takes ~10 min on Render's free CPU, ~1 min locally.
STALE_BUILD_SECONDS = 30 * 60

IN_PROGRESS = ("queued", "indexing")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Repo(Base):
    __tablename__ = "repos"

    repo_url: Mapped[str] = mapped_column(String(500), primary_key=True)
    status: Mapped[str] = mapped_column(String(16))
    build_id: Mapped[str] = mapped_column(String(32))
    collection: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    stats: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    timings_ms: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # {"ids": [...], "tokens": [[...], ...]} for the serving collection, about
    # 1 MB for a 300-chunk repo. Deferred so status checks never load it.
    bm25: Mapped[dict | None] = deferred(mapped_column(JSON, nullable=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


_engine = None
_Session: sessionmaker | None = None


def database_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if url:
        # Render/Heroku-style URLs say postgres://; SQLAlchemy needs a driver.
        if url.startswith("postgres://"):
            url = "postgresql+psycopg://" + url[len("postgres://"):]
        elif url.startswith("postgresql://"):
            url = "postgresql+psycopg://" + url[len("postgresql://"):]
        return url
    data_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
    os.makedirs(data_dir, exist_ok=True)
    return f"sqlite:///{os.path.join(data_dir, 'app.db')}"


def init(url: str | None = None) -> None:
    """Create the engine and the table (idempotent; safe on every replica)."""
    global _engine, _Session
    url = url or database_url()
    kwargs: dict = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        # Per replica: a handful of chats + status polls at once is plenty.
        kwargs.update(pool_size=5, max_overflow=5)
    _engine = create_engine(url, **kwargs)
    _Session = sessionmaker(_engine, expire_on_commit=False)
    Base.metadata.create_all(_engine)


def session() -> Session:
    if _Session is None:
        init()
    return _Session()


def ping() -> None:
    with session() as s:
        s.execute(select(1))


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def get_repo(repo_url: str) -> Repo | None:
    with session() as s:
        return s.get(Repo, repo_url)


def list_ready() -> list[Repo]:
    with session() as s:
        return list(s.scalars(select(Repo).where(Repo.collection.is_not(None)).order_by(Repo.repo_url)))


def get_bm25(repo_url: str, collection: str) -> dict | None:
    """BM25 tokens for a repo, only if they still belong to `collection`."""
    with session() as s:
        row = s.execute(
            select(Repo.bm25).where(Repo.repo_url == repo_url, Repo.collection == collection)
        ).first()
        return row[0] if row else None


# ---------------------------------------------------------------------------
# Build lifecycle (compare-and-set writes)
# ---------------------------------------------------------------------------

def request_build(repo_url: str, force: bool = False) -> tuple[Repo, bool]:
    """Start a build for repo_url unless one isn't needed.

    Returns (row, started). started is False when the repo is already ready
    (and not forced) or a build is already in progress: concurrent /index
    calls for the same repo, from any replica, share one build.
    """
    new_id = uuid.uuid4().hex
    with session() as s:
        try:
            s.add(Repo(repo_url=repo_url, status="queued", build_id=new_id, updated_at=_utcnow()))
            s.commit()
            return s.get(Repo, repo_url), True
        except IntegrityError:  # row exists (maybe created by another replica just now)
            s.rollback()

        row = s.get(Repo, repo_url)
        stale = _utcnow() - timedelta(seconds=STALE_BUILD_SECONDS)
        in_progress = row.status in IN_PROGRESS and _aware(row.updated_at) > stale
        # A repo with a serving collection counts as indexed even if its
        # latest rebuild failed; only `force` replaces it.
        if in_progress or (row.collection is not None and not force):
            return row, False

        # Compare-and-set on the build_id we just read: if another replica
        # started a build in between, rowcount is 0 and we join theirs.
        res = s.execute(
            update(Repo)
            .where(Repo.repo_url == repo_url, Repo.build_id == row.build_id)
            .values(status="queued", build_id=new_id, error=None, updated_at=_utcnow())
        )
        s.commit()
        s.expire_all()
        return s.get(Repo, repo_url), res.rowcount == 1


def mark_indexing(repo_url: str, build_id: str) -> bool:
    """Claim the build. False means it was superseded and the job should stop."""
    with session() as s:
        res = s.execute(
            update(Repo)
            .where(Repo.repo_url == repo_url, Repo.build_id == build_id, Repo.status.in_(IN_PROGRESS))
            .values(status="indexing", updated_at=_utcnow())
        )
        s.commit()
        return res.rowcount == 1


def finish_build(repo_url: str, build_id: str, collection: str, stats: dict,
                 summary: dict, timings_ms: dict, bm25: dict) -> str | None | bool:
    """Atomically make `collection` the serving index.

    Returns the previous collection name (or None) for the caller to delete,
    or False if this build was superseded or already finished by a duplicate
    run of the same job (the caller then deletes its own collection).
    """
    with session() as s:
        row = s.execute(
            select(Repo).where(Repo.repo_url == repo_url).with_for_update()
        ).scalar_one_or_none()
        if row is None or row.build_id != build_id or row.status != "indexing":
            return False
        previous = row.collection
        row.collection = collection
        row.status = "ready"
        row.error = None
        row.stats, row.summary, row.timings_ms, row.bm25 = stats, summary, timings_ms, bm25
        row.updated_at = _utcnow()
        s.commit()
        return previous


def fail_build(repo_url: str, build_id: str, error: str) -> None:
    """Record a failure, unless the build already finished: a duplicate
    attempt failing after another attempt published must not flip a ready
    repo to "failed"."""
    with session() as s:
        s.execute(
            update(Repo)
            .where(Repo.repo_url == repo_url, Repo.build_id == build_id, Repo.status.in_(IN_PROGRESS))
            .values(status="failed", error=error[:2000], updated_at=_utcnow())
        )
        s.commit()


def fail_in_progress(reason: str) -> int:
    """Mark every queued/indexing build failed (single-instance mode only:
    after a restart, its in-process builds are gone)."""
    with session() as s:
        res = s.execute(
            update(Repo).where(Repo.status.in_(IN_PROGRESS))
            .values(status="failed", error=reason, updated_at=_utcnow())
        )
        s.commit()
        return res.rowcount


def _aware(dt: datetime) -> datetime:
    # SQLite returns naive datetimes even for timezone=True columns.
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
