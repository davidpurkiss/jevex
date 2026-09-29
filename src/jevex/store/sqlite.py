"""The default store: SQLite in WAL mode, safe for several processes on one host.

WAL lets readers run alongside a writer. Every write takes the database's write lock up
front (``BEGIN IMMEDIATE``), and ``busy_timeout`` makes a process wait for that lock
rather than fail. So read-modify-write operations (stats increments, ``try_spend``) are
atomic across processes, not only across tasks.

``sqlite3`` is blocking, so each call runs in a worker thread. One connection per store
is serialised by a lock; open one store per process (don't share it across a fork).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic_core import to_jsonable_python

from jevex.store.base import (
    GeneratorRecord,
    GeneratorStats,
    KeyMapping,
    SpendEntry,
    StoreError,
    VerifiedExample,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE generators (
    id TEXT PRIMARY KEY,
    field TEXT NOT NULL,
    spec TEXT NOT NULL,
    scope TEXT NOT NULL,
    enabled INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX generators_field ON generators (field, created_at);

CREATE TABLE key_mappings (
    fingerprint TEXT NOT NULL,
    path TEXT NOT NULL,
    field TEXT NOT NULL,
    normalisers TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (fingerprint, path)
);

CREATE TABLE examples (
    id TEXT PRIMARY KEY,
    field TEXT NOT NULL,
    statement TEXT NOT NULL,
    value TEXT NOT NULL,
    evidence_start INTEGER,
    evidence_end INTEGER,
    context TEXT NOT NULL,
    source TEXT NOT NULL,
    probability REAL,
    created_at REAL NOT NULL
);
CREATE INDEX examples_field ON examples (field, created_at);

CREATE TABLE generator_stats (
    generator_id TEXT PRIMARY KEY,
    documents INTEGER NOT NULL DEFAULT 0,
    hits INTEGER NOT NULL DEFAULT 0,
    wins INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE spend (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    amount_micro_usd INTEGER NOT NULL,
    kind TEXT NOT NULL,
    run_id TEXT,
    note TEXT,
    at REAL NOT NULL
);
CREATE INDEX spend_at ON spend (at);
"""


def _ts(value: datetime) -> float:
    if value.tzinfo is None:
        raise ValueError(f"naive datetime {value!r}: pass a timezone-aware datetime")
    return value.timestamp()


def _dt(value: float) -> datetime:
    return datetime.fromtimestamp(value, UTC)


def _micro(usd: float) -> int:
    # Integer millionths of a dollar, so ledger sums are exact and a cap is hit exactly.
    return round(usd * 1_000_000)


def _json(value: Any) -> str:
    return json.dumps(to_jsonable_python(value), ensure_ascii=False, sort_keys=True)


class SQLiteStore:
    """:class:`~jevex.store.Store` on a SQLite file (or ``":memory:"`` for tests).

    The database and its tables are created on first open. A database written by a newer
    jevex (a higher schema version) is refused rather than misread.
    """

    def __init__(self, path: Path | str, *, busy_timeout_s: float = 30.0) -> None:
        self.path = path if path == ":memory:" else Path(path)
        self._lock = threading.Lock()
        try:
            self._conn = sqlite3.connect(
                self.path,
                timeout=busy_timeout_s,
                isolation_level=None,  # autocommit; transactions are explicit
                check_same_thread=False,  # used from worker threads, behind self._lock
            )
            self._conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_s * 1000)}")
            if self.path != ":memory:":
                mode = self._conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
                if str(mode).lower() != "wal":
                    raise StoreError(f"{self.path}: couldn't enable WAL mode (got {mode})")
                self._conn.execute("PRAGMA synchronous = NORMAL")
            self._migrate()
        except sqlite3.Error as exc:
            raise StoreError(f"can't open store {self.path}: {exc}") from exc

    def _migrate(self) -> None:
        with self._write() as cur:
            version = int(cur.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise StoreError(
                    f"{self.path} has store schema v{version}; this jevex reads up to "
                    f"v{SCHEMA_VERSION}. Upgrade jevex."
                )
            if version == 0:
                for statement in _SCHEMA.split(";"):
                    if statement.strip():
                        cur.execute(statement)
                cur.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    # -- plumbing ---------------------------------------------------------------------

    @contextmanager
    def _write(self) -> Generator[sqlite3.Cursor]:
        """A write transaction holding the database's write lock from the start."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn.cursor()
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

    async def _call[T](self, fn: Callable[[], T]) -> T:
        def locked() -> T:
            with self._lock:
                try:
                    return fn()
                except sqlite3.Error as exc:
                    raise StoreError(f"store {self.path}: {exc}") from exc

        return await asyncio.to_thread(locked)

    def _rows(self, sql: str, params: tuple[Any, ...] = ()) -> Iterator[sqlite3.Row]:
        cur = self._conn.execute(sql, params)
        cur.row_factory = sqlite3.Row
        return iter(cur.fetchall())

    # -- generators -------------------------------------------------------------------

    async def put_generator(self, generator: GeneratorRecord) -> None:
        def run() -> None:
            with self._write() as cur:
                cur.execute(
                    "INSERT OR REPLACE INTO generators VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        generator.id,
                        generator.field,
                        _json(generator.spec),
                        _json(generator.scope),
                        int(generator.enabled),
                        _ts(generator.created_at),
                    ),
                )

        await self._call(run)

    @staticmethod
    def _generator(row: sqlite3.Row) -> GeneratorRecord:
        return GeneratorRecord(
            id=row["id"],
            field=row["field"],
            spec=json.loads(row["spec"]),
            scope=json.loads(row["scope"]),
            enabled=bool(row["enabled"]),
            created_at=_dt(row["created_at"]),
        )

    async def get_generator(self, generator_id: str) -> GeneratorRecord | None:
        def run() -> GeneratorRecord | None:
            rows = list(self._rows("SELECT * FROM generators WHERE id = ?", (generator_id,)))
            return self._generator(rows[0]) if rows else None

        return await self._call(run)

    async def generators(
        self, field: str | None = None, *, include_disabled: bool = False
    ) -> list[GeneratorRecord]:
        where: list[str] = []
        params: list[Any] = []
        if field is not None:
            where.append("field = ?")
            params.append(field)
        if not include_disabled:
            where.append("enabled = 1")
        sql = "SELECT * FROM generators"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at, id"

        return await self._call(
            lambda: [self._generator(r) for r in self._rows(sql, tuple(params))]
        )

    async def set_generator_enabled(self, generator_id: str, enabled: bool) -> None:
        def run() -> None:
            with self._write() as cur:
                cur.execute(
                    "UPDATE generators SET enabled = ? WHERE id = ?", (int(enabled), generator_id)
                )
                if cur.rowcount == 0:
                    raise KeyError(generator_id)

        await self._call(run)

    # -- key mappings -----------------------------------------------------------------

    async def put_key_mapping(self, mapping: KeyMapping) -> None:
        def run() -> None:
            with self._write() as cur:
                cur.execute(
                    "INSERT OR REPLACE INTO key_mappings VALUES (?, ?, ?, ?, ?)",
                    (
                        mapping.fingerprint,
                        mapping.path,
                        mapping.field,
                        _json(mapping.normalisers),
                        _ts(mapping.created_at),
                    ),
                )

        await self._call(run)

    async def key_mappings(self, fingerprint: str) -> list[KeyMapping]:
        def run() -> list[KeyMapping]:
            return [
                KeyMapping(
                    fingerprint=r["fingerprint"],
                    path=r["path"],
                    field=r["field"],
                    normalisers=json.loads(r["normalisers"]),
                    created_at=_dt(r["created_at"]),
                )
                for r in self._rows(
                    "SELECT * FROM key_mappings WHERE fingerprint = ? ORDER BY path",
                    (fingerprint,),
                )
            ]

        return await self._call(run)

    # -- verified examples ------------------------------------------------------------

    async def add_example(self, example: VerifiedExample) -> None:
        start, end = example.evidence or (None, None)

        def run() -> None:
            with self._write() as cur:
                cur.execute(
                    "INSERT OR REPLACE INTO examples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        example.id,
                        example.field,
                        example.statement,
                        _json(example.value),
                        start,
                        end,
                        _json(example.context),
                        example.source,
                        example.probability,
                        _ts(example.created_at),
                    ),
                )

        await self._call(run)

    async def examples(self, field: str, *, limit: int | None = None) -> list[VerifiedExample]:
        sql = "SELECT * FROM examples WHERE field = ? ORDER BY created_at DESC, id"
        params: tuple[Any, ...] = (field,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (field, limit)

        def run() -> list[VerifiedExample]:
            return [
                VerifiedExample(
                    id=r["id"],
                    field=r["field"],
                    statement=r["statement"],
                    value=json.loads(r["value"]),
                    evidence=(
                        None
                        if r["evidence_start"] is None
                        else (r["evidence_start"], r["evidence_end"])
                    ),
                    context=json.loads(r["context"]),
                    source=r["source"],
                    probability=r["probability"],
                    created_at=_dt(r["created_at"]),
                )
                for r in self._rows(sql, params)
            ]

        return await self._call(run)

    # -- generator stats --------------------------------------------------------------

    async def record_generator_stats(
        self, generator_id: str, *, documents: int = 0, hits: int = 0, wins: int = 0
    ) -> None:
        def run() -> None:
            with self._write() as cur:
                cur.execute(
                    "INSERT INTO generator_stats VALUES (?, ?, ?, ?) "
                    "ON CONFLICT (generator_id) DO UPDATE SET "
                    "documents = documents + excluded.documents, "
                    "hits = hits + excluded.hits, wins = wins + excluded.wins",
                    (generator_id, documents, hits, wins),
                )

        await self._call(run)

    async def generator_stats(self, generator_id: str) -> GeneratorStats:
        def run() -> GeneratorStats:
            rows = list(
                self._rows("SELECT * FROM generator_stats WHERE generator_id = ?", (generator_id,))
            )
            if not rows:
                return GeneratorStats(generator_id=generator_id)
            r = rows[0]
            return GeneratorStats(
                generator_id=generator_id,
                documents=r["documents"],
                hits=r["hits"],
                wins=r["wins"],
            )

        return await self._call(run)

    # -- spend ledger -----------------------------------------------------------------

    @staticmethod
    def _insert_spend(cur: sqlite3.Cursor, entry: SpendEntry) -> None:
        cur.execute(
            "INSERT INTO spend (amount_micro_usd, kind, run_id, note, at) VALUES (?, ?, ?, ?, ?)",
            (_micro(entry.amount_usd), entry.kind, entry.run_id, entry.note, _ts(entry.at)),
        )

    @staticmethod
    def _spend_query(
        since: datetime | None, kind: str | None, run_id: str | None
    ) -> tuple[str, tuple[Any, ...]]:
        where: list[str] = []
        params: list[Any] = []
        if since is not None:
            where.append("at >= ?")
            params.append(_ts(since))
        if kind is not None:
            where.append("kind = ?")
            params.append(kind)
        if run_id is not None:
            where.append("run_id = ?")
            params.append(run_id)
        sql = "SELECT COALESCE(SUM(amount_micro_usd), 0) FROM spend"
        if where:
            sql += " WHERE " + " AND ".join(where)
        return sql, tuple(params)

    async def record_spend(self, entry: SpendEntry) -> None:
        def run() -> None:
            with self._write() as cur:
                self._insert_spend(cur, entry)

        await self._call(run)

    async def spend(
        self,
        *,
        since: datetime | None = None,
        kind: Literal["jev", "llm"] | None = None,
        run_id: str | None = None,
    ) -> float:
        sql, params = self._spend_query(since, kind, run_id)
        micro = await self._call(lambda: int(self._conn.execute(sql, params).fetchone()[0]))
        return micro / 1_000_000

    async def try_spend(
        self, entry: SpendEntry, *, cap_usd: float, since: datetime | None = None
    ) -> bool:
        sql, params = self._spend_query(since, None, None)

        def run() -> bool:
            with self._write() as cur:
                spent = int(cur.execute(sql, params).fetchone()[0])
                if spent + _micro(entry.amount_usd) > _micro(cap_usd):
                    return False
                self._insert_spend(cur, entry)
                return True

        return await self._call(run)

    async def aclose(self) -> None:
        await self._call(self._conn.close)
