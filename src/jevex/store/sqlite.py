"""The default store: SQLite in WAL mode, safe for several processes on one host.

WAL lets readers run alongside a writer. Every write takes the database's write lock up
front (``BEGIN IMMEDIATE``), and ``busy_timeout`` makes a process wait for that lock
rather than fail. So read-modify-write operations (stats increments, ``try_spend``) are
atomic across processes, not only across tasks. A failed write always rolls back, so it
never leaves the lock held.

``sqlite3`` is blocking, so each store runs its calls on its own single worker thread,
which also serialises them; it never takes threads from asyncio's shared executor. Open
one store per process (don't share it across a fork).

Durability: ``synchronous=NORMAL`` is WAL's usual setting. A committed write survives a
crash of the process but can be lost on power loss or an OS crash. For the ledger that
means under-counting the last few charges, never corrupting the database.

The spend ledger keeps one row per charge and nothing prunes it yet. With a run budget,
a busy host adds a row per document (its Jev spend) and per priced LLM call, plus one
``llm_call`` row per LLM call when ``llm_rpm`` is set.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

    from jevex.store.base import SpendKind

SCHEMA_VERSION = 2

# Version 1's tables; a new database runs this, then every migration.
_SCHEMA = """
CREATE TABLE generators (
    id TEXT PRIMARY KEY,
    field TEXT NOT NULL,
    spec TEXT NOT NULL,
    scope TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX generators_field ON generators (field, created_at);

CREATE TABLE generator_disables (
    generator_id TEXT PRIMARY KEY
);

CREATE TABLE key_mappings (
    fingerprint TEXT NOT NULL,
    schema_name TEXT NOT NULL,
    path TEXT NOT NULL,
    field TEXT,
    normalisers TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (fingerprint, schema_name, path)
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
    amount_nano_usd INTEGER NOT NULL,
    kind TEXT NOT NULL,
    run_id TEXT,
    note TEXT,
    at REAL NOT NULL
);
CREATE INDEX spend_at ON spend (at);
"""

# Schema version → the statements that bring the version before it up to it.
_MIGRATIONS = {
    2: """
ALTER TABLE key_mappings ADD COLUMN unsure INTEGER NOT NULL DEFAULT 0;

CREATE TABLE key_path_unsure (
    fingerprint TEXT NOT NULL,
    schema_name TEXT NOT NULL,
    path TEXT NOT NULL,
    count INTEGER NOT NULL,
    PRIMARY KEY (fingerprint, schema_name, path)
);
""",
}

_NANO = 1_000_000_000
_INT64_MAX = 2**63 - 1


def _ts(value: datetime) -> float:
    if value.tzinfo is None:
        raise ValueError(f"naive datetime {value!r}: pass a timezone-aware datetime")
    return value.timestamp()


def _dt(value: float) -> datetime:
    return datetime.fromtimestamp(value, UTC)


def _nano(usd: float) -> int:
    # Integer billionths of a dollar: Jev charges a few nano-dollars per token, and
    # integer sums are exact, so a cap is hit exactly.
    return round(usd * _NANO)


def _nano_cap(usd: float) -> int:
    if not math.isfinite(usd) or usd < 0:
        raise ValueError(f"cap_usd must be finite and non-negative, not {usd!r}")
    return min(round(usd * _NANO), _INT64_MAX)


def _json(value: Any) -> str:
    return json.dumps(to_jsonable_python(value), ensure_ascii=False, sort_keys=True)


def _busy(exc: sqlite3.Error) -> bool:
    message = str(exc).lower()
    return "locked" in message or "busy" in message


class SQLiteStore:
    """:class:`~jevex.store.Store` on a SQLite file (or ``":memory:"`` for tests).

    The database and its tables are created on first open; several processes may open a
    new database at once. A database written by a newer jevex (a higher schema version)
    is refused rather than misread.
    """

    def __init__(self, path: Path | str, *, busy_timeout_s: float = 30.0) -> None:
        self.path: Path | str = ":memory:" if str(path) == ":memory:" else Path(path)
        self._busy_timeout_s = busy_timeout_s
        self._conn: sqlite3.Connection | None = None
        try:
            self._conn = sqlite3.connect(
                self.path,
                timeout=busy_timeout_s,
                isolation_level=None,  # autocommit; transactions are explicit
                check_same_thread=False,  # opened here, then used only by the store's thread
            )
            self._conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_s * 1000)}")
            if self.path != ":memory:":
                self._enable_wal(self._conn)
                self._conn.execute("PRAGMA synchronous = NORMAL")
            self._migrate()
        except BaseException as exc:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            if isinstance(exc, sqlite3.Error):
                raise StoreError(f"can't open store {self.path}: {exc}") from exc
            raise
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jevex-store")

    def _enable_wal(self, conn: sqlite3.Connection) -> None:
        # Switching the journal mode needs an exclusive lock, and SQLite reports busy
        # at once instead of honouring busy_timeout. Several processes opening a new
        # database together hit that, so retry until the timeout.
        deadline = time.monotonic() + self._busy_timeout_s
        while True:
            try:
                mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
                break
            except sqlite3.OperationalError as exc:
                if not _busy(exc) or time.monotonic() >= deadline:
                    raise
                time.sleep(random.uniform(0.01, 0.05))
        if str(mode).lower() != "wal":
            raise StoreError(f"{self.path}: couldn't enable WAL mode (got {mode})")

    def _migrate(self) -> None:
        with self._write() as cur:
            version = int(cur.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise StoreError(
                    f"{self.path} has store schema v{version}; this jevex reads up to "
                    f"v{SCHEMA_VERSION}. Upgrade jevex."
                )
            if version == SCHEMA_VERSION:
                return
            scripts = [_SCHEMA] if version == 0 else []
            scripts += [_MIGRATIONS[v] for v in range(max(version, 1) + 1, SCHEMA_VERSION + 1)]
            for script in scripts:
                for statement in script.split(";"):
                    if statement.strip():
                        cur.execute(statement)
            cur.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    # -- plumbing ---------------------------------------------------------------------

    @property
    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            raise StoreError(f"store {self.path} is closed")
        return self._conn

    @contextmanager
    def _write(self) -> Generator[sqlite3.Cursor]:
        """A write transaction holding the database's write lock from the start.

        Any failure, including a failed ``COMMIT``, rolls back unless SQLite already
        did, so the lock is never left held and the original error is what's raised.
        """
        conn = self._db
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn.cursor()
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise

    async def _call[T](self, fn: Callable[[], T]) -> T:
        def run() -> T:
            try:
                return fn()
            except sqlite3.Error as exc:
                raise StoreError(f"store {self.path}: {exc}") from exc

        if self._conn is None:
            raise StoreError(f"store {self.path} is closed")
        return await asyncio.get_running_loop().run_in_executor(self._executor, run)

    def _rows(self, sql: str, params: tuple[Any, ...] = ()) -> Iterator[sqlite3.Row]:
        cur = self._db.execute(sql, params)
        cur.row_factory = sqlite3.Row
        return iter(cur.fetchall())

    # -- generators -------------------------------------------------------------------

    _SELECT_GENERATORS = (
        "SELECT g.*, d.generator_id IS NOT NULL AS disabled FROM generators g "
        "LEFT JOIN generator_disables d ON d.generator_id = g.id"
    )

    async def put_generator(self, generator: GeneratorRecord) -> None:
        def run() -> None:
            with self._write() as cur:
                cur.execute(
                    "INSERT OR REPLACE INTO generators VALUES (?, ?, ?, ?, ?)",
                    (
                        generator.id,
                        generator.field,
                        _json(generator.spec),
                        _json(generator.scope),
                        _ts(generator.created_at),
                    ),
                )
                self._set_enabled(cur, generator.id, generator.enabled)

        await self._call(run)

    @staticmethod
    def _generator(row: sqlite3.Row) -> GeneratorRecord:
        return GeneratorRecord(
            id=row["id"],
            field=row["field"],
            spec=json.loads(row["spec"]),
            scope=json.loads(row["scope"]),
            enabled=not row["disabled"],
            created_at=_dt(row["created_at"]),
        )

    async def get_generator(self, generator_id: str) -> GeneratorRecord | None:
        def run() -> GeneratorRecord | None:
            sql = f"{self._SELECT_GENERATORS} WHERE g.id = ?"
            rows = list(self._rows(sql, (generator_id,)))
            return self._generator(rows[0]) if rows else None

        return await self._call(run)

    async def generators(
        self, field: str | None = None, *, include_disabled: bool = False
    ) -> list[GeneratorRecord]:
        where: list[str] = []
        params: list[Any] = []
        if field is not None:
            where.append("g.field = ?")
            params.append(field)
        if not include_disabled:
            where.append("d.generator_id IS NULL")
        sql = self._SELECT_GENERATORS
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY g.created_at, g.id"

        return await self._call(
            lambda: [self._generator(r) for r in self._rows(sql, tuple(params))]
        )

    @staticmethod
    def _set_enabled(cur: sqlite3.Cursor, generator_id: str, enabled: bool) -> None:
        if enabled:
            cur.execute("DELETE FROM generator_disables WHERE generator_id = ?", (generator_id,))
        else:
            cur.execute("INSERT OR IGNORE INTO generator_disables VALUES (?)", (generator_id,))

    async def set_generator_enabled(self, generator_id: str, enabled: bool) -> None:
        def run() -> None:
            with self._write() as cur:
                self._set_enabled(cur, generator_id, enabled)

        await self._call(run)

    async def disabled_generator_ids(self) -> set[str]:
        return await self._call(
            lambda: {r["generator_id"] for r in self._rows("SELECT * FROM generator_disables")}
        )

    # -- key mappings -----------------------------------------------------------------

    async def put_key_mapping(self, mapping: KeyMapping) -> None:
        key = (mapping.fingerprint, mapping.schema_name, mapping.path)

        def run() -> None:
            with self._write() as cur:
                cur.execute(
                    "INSERT OR REPLACE INTO key_mappings (fingerprint, schema_name, path, "
                    "field, normalisers, unsure, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        *key,
                        mapping.field,
                        _json(mapping.normalisers),
                        int(mapping.unsure),
                        _ts(mapping.created_at),
                    ),
                )
                cur.execute(
                    "DELETE FROM key_path_unsure "
                    "WHERE fingerprint = ? AND schema_name = ? AND path = ?",
                    key,
                )

        await self._call(run)

    async def key_mappings(
        self, fingerprint: str | None = None, *, schema: str | None = None
    ) -> list[KeyMapping]:
        where: list[str] = []
        params: tuple[Any, ...] = ()
        if fingerprint is not None:
            where.append("fingerprint = ?")
            params += (fingerprint,)
        if schema is not None:
            where.append("schema_name = ?")
            params += (schema,)
        sql = "SELECT * FROM key_mappings"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY fingerprint, path, schema_name"

        def run() -> list[KeyMapping]:
            return [
                KeyMapping(
                    fingerprint=r["fingerprint"],
                    schema=r["schema_name"],
                    path=r["path"],
                    field=r["field"],
                    normalisers=json.loads(r["normalisers"]),
                    unsure=bool(r["unsure"]),
                    created_at=_dt(r["created_at"]),
                )
                for r in self._rows(sql, params)
            ]

        return await self._call(run)

    async def count_unsure_key_paths(
        self, fingerprint: str, schema: str, paths: list[str]
    ) -> dict[str, int]:
        def run() -> dict[str, int]:
            counts: dict[str, int] = {}
            with self._write() as cur:
                for path in dict.fromkeys(paths):
                    cur.execute(
                        "INSERT INTO key_path_unsure VALUES (?, ?, ?, 1) "
                        "ON CONFLICT (fingerprint, schema_name, path) "
                        "DO UPDATE SET count = count + 1",
                        (fingerprint, schema, path),
                    )
                    row = cur.execute(
                        "SELECT count FROM key_path_unsure "
                        "WHERE fingerprint = ? AND schema_name = ? AND path = ?",
                        (fingerprint, schema, path),
                    ).fetchone()
                    counts[path] = int(row[0])
            return counts

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

    async def examples(
        self, field: str | None = None, *, limit: int | None = None
    ) -> list[VerifiedExample]:
        sql = "SELECT * FROM examples"
        params: tuple[Any, ...] = ()
        if field is not None:
            sql += " WHERE field = ?"
            params += (field,)
        sql += " ORDER BY created_at DESC, id"
        if limit is not None:
            sql += " LIMIT ?"
            params += (limit,)

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
            "INSERT INTO spend (amount_nano_usd, kind, run_id, note, at) VALUES (?, ?, ?, ?, ?)",
            (_nano(entry.amount_usd), entry.kind, entry.run_id, entry.note, _ts(entry.at)),
        )

    @staticmethod
    def _spend_filter(
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
        return (" WHERE " + " AND ".join(where) if where else ""), tuple(params)

    async def record_spend(self, entry: SpendEntry) -> None:
        def run() -> None:
            with self._write() as cur:
                self._insert_spend(cur, entry)

        await self._call(run)

    async def spend(
        self,
        *,
        since: datetime | None = None,
        kind: SpendKind | None = None,
        run_id: str | None = None,
    ) -> float:
        where, params = self._spend_filter(since, kind, run_id)
        sql = f"SELECT COALESCE(SUM(amount_nano_usd), 0) FROM spend{where}"
        nano = await self._call(lambda: int(self._db.execute(sql, params).fetchone()[0]))
        return nano / _NANO

    async def try_spend(
        self,
        entry: SpendEntry,
        *,
        cap_usd: float | None = None,
        max_count: int | None = None,
        since: datetime | None = None,
        kind: SpendKind | None = None,
    ) -> bool:
        cap = None if cap_usd is None else _nano_cap(cap_usd)
        if max_count is not None and max_count < 0:
            raise ValueError(f"max_count must be non-negative, not {max_count}")
        if kind is not None and entry.kind != kind:
            raise ValueError(f"a {entry.kind} entry can't be checked against {kind} limits")
        where, params = self._spend_filter(since, kind, None)
        sql = f"SELECT COALESCE(SUM(amount_nano_usd), 0), COUNT(*) FROM spend{where}"

        def run() -> bool:
            with self._write() as cur:
                spent, count = cur.execute(sql, params).fetchone()
                if cap is not None and int(spent) + _nano(entry.amount_usd) > cap:
                    return False
                if max_count is not None and int(count) + 1 > max_count:
                    return False
                self._insert_spend(cur, entry)
                return True

        return await self._call(run)

    async def aclose(self) -> None:
        conn = self._conn
        if conn is None:
            return
        try:
            await self._call(conn.close)
        finally:
            self._conn = None
            self._executor.shutdown(wait=True)
