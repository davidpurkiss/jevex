"""The Postgres store (``postgres`` extra): one store shared by many hosts and workers.

Every table lives in its own Postgres schema (``jevex`` unless ``db_schema`` says
otherwise), so a store can share a database with an application's own tables, and
several independent stores can share one database.

Concurrency: each operation is one transaction on a pooled connection, at Postgres's
default isolation (read committed). Counters are single upserts (``count = count + 1``),
so concurrent increments add up. ``try_spend`` locks the spend table against other
writers for its check and insert, so workers sharing a cap can't overshoot it together;
readers aren't blocked. Rows written together are locked in one order, so two workers
can't deadlock on them.

Cancellation: an operation already sent runs to the end even if the task awaiting it is
cancelled (as :class:`~jevex.store.Store` describes), and :meth:`PostgresStore.aclose`
waits for it. So a charge recorded by a document being cancelled still counts.

The store's connection pool belongs to the event loop that first used it: use a store
from one event loop, and open one per process (don't share it across a fork).

JSON values (specs, scopes, example values and context, normaliser chains) are stored
as ``json``, which keeps the text as written, so they come back as SQLite returns them.

Limits: Postgres text can't hold NUL characters, so a record with one in a text column
(a statement, an id, a field name...) is refused with a :class:`StoreError`, as is a
JSON value holding a NaN or infinite float.
"""

from __future__ import annotations

import asyncio
import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, LiteralString

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Json
from psycopg_pool import AsyncConnectionPool
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
    from collections.abc import Awaitable, Callable

    from psycopg import AsyncConnection
    from psycopg.rows import TupleRow

    from jevex.store.base import SpendKind

SCHEMA_VERSION = 1

# Version 1's tables. ``{s}`` is the store's Postgres schema.
_SCHEMA: LiteralString = """
CREATE TABLE {s}.generators (
    id TEXT PRIMARY KEY,
    field TEXT NOT NULL,
    spec JSON NOT NULL,
    scope JSON NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX generators_field ON {s}.generators (field, created_at);

CREATE TABLE {s}.generator_disables (
    generator_id TEXT PRIMARY KEY
);

CREATE TABLE {s}.key_mappings (
    fingerprint TEXT NOT NULL,
    schema_name TEXT NOT NULL,
    path TEXT NOT NULL,
    field TEXT,
    normalisers JSON NOT NULL,
    unsure BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (fingerprint, schema_name, path)
);

CREATE TABLE {s}.key_path_unsure (
    fingerprint TEXT NOT NULL,
    schema_name TEXT NOT NULL,
    path TEXT NOT NULL,
    count INTEGER NOT NULL,
    PRIMARY KEY (fingerprint, schema_name, path)
);

CREATE TABLE {s}.examples (
    id TEXT PRIMARY KEY,
    field TEXT NOT NULL,
    statement TEXT NOT NULL,
    value JSON NOT NULL,
    evidence_start INTEGER,
    evidence_end INTEGER,
    context JSON NOT NULL,
    source TEXT NOT NULL,
    probability DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX examples_field ON {s}.examples (field, created_at);

CREATE TABLE {s}.generator_stats (
    generator_id TEXT PRIMARY KEY,
    documents BIGINT NOT NULL DEFAULT 0,
    hits BIGINT NOT NULL DEFAULT 0,
    wins BIGINT NOT NULL DEFAULT 0
);

CREATE TABLE {s}.spend (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    amount_nano_usd BIGINT NOT NULL,
    kind TEXT NOT NULL,
    run_id TEXT,
    note TEXT,
    at TIMESTAMPTZ NOT NULL
);
CREATE INDEX spend_at ON {s}.spend (at);
"""

# Schema version → the statements that bring the version before it up to it.
_MIGRATIONS: dict[int, LiteralString] = {}

# Serialises schema creation and migration across every process opening a store on the
# same database (an advisory lock: it holds no rows and ends with the transaction).
_MIGRATION_LOCK = 0x6A65_7665_7853_7431  # "jevexSt1"

_NANO = 1_000_000_000
_INT64_MAX = 2**63 - 1

# Text comparisons in ORDER BY use byte order, as SQLite does, not the database's locale.
_C: LiteralString = 'COLLATE "C"'


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"naive datetime {value!r}: pass a timezone-aware datetime")
    return value


def _utc(value: datetime) -> datetime:
    # timestamptz comes back in the session's time zone.
    return value.astimezone(UTC)


def _nano(usd: float) -> int:
    return round(usd * _NANO)


def _nano_cap(usd: float) -> int:
    if not math.isfinite(usd) or usd < 0:
        raise ValueError(f"cap_usd must be finite and non-negative, not {usd!r}")
    return min(round(usd * _NANO), _INT64_MAX)


def _json(value: Any) -> Json:
    return Json(to_jsonable_python(value))


def _settle(task: asyncio.Task[Any]) -> None:
    # A shielded operation whose caller was cancelled has no one left to raise to; read
    # its outcome so asyncio doesn't log "exception was never retrieved".
    if not task.cancelled():
        task.exception()


class PostgresStore:
    """:class:`~jevex.store.Store` on a Postgres database, for several hosts and workers.

    ``conninfo`` is a libpq connection string or URL (``postgresql://user@host/db``).
    Opening connects once to create or migrate the tables in ``db_schema`` (several
    processes may do that at once), so a database that can't be reached, or that a newer
    jevex has written (a higher schema version), raises :class:`StoreError` here. Each
    process then holds a pool of ``min_size`` to ``max_size`` connections, opened on
    first use; an operation waits up to ``timeout_s`` for a free one.
    """

    def __init__(
        self,
        conninfo: str,
        *,
        db_schema: str = "jevex",
        min_size: int = 1,
        max_size: int = 10,
        timeout_s: float = 30.0,
    ) -> None:
        self.conninfo = conninfo
        self.db_schema = db_schema
        self._min_size = min_size
        self._max_size = max_size
        self._timeout_s = timeout_s
        self._schema_id = sql.Identifier(db_schema)
        self._pool: AsyncConnectionPool[AsyncConnection[TupleRow]] | None = None
        self._pool_lock: asyncio.Lock | None = None
        self._pending: set[asyncio.Task[Any]] = set()
        self._closed = False
        try:
            self._migrate()
        except psycopg.Error as exc:
            raise StoreError(f"can't open store {self}: {exc}") from exc

    def __str__(self) -> str:
        # Never the conninfo: it may hold a password.
        return f"postgres schema {self.db_schema!r}"

    def _q(self, query: LiteralString) -> sql.Composed:
        return sql.SQL(query).format(s=self._schema_id)

    def _migrate(self) -> None:
        connect_timeout = max(1, math.ceil(self._timeout_s))
        with psycopg.connect(self.conninfo, connect_timeout=connect_timeout) as conn:
            # The connection's block is one transaction, committed on a clean exit.
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATION_LOCK,))
            row = conn.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = %s AND table_name = 'schema_version')",
                (self.db_schema,),
            ).fetchone()
            version = 0
            if row is not None and row[0]:
                current = conn.execute(self._q("SELECT version FROM {s}.schema_version"))
                version = int((current.fetchone() or (0,))[0])
            if version > SCHEMA_VERSION:
                raise StoreError(
                    f"{self} has store schema v{version}; this jevex reads up to "
                    f"v{SCHEMA_VERSION}. Upgrade jevex."
                )
            if version == SCHEMA_VERSION:
                return
            if version == 0:
                conn.execute(self._q("CREATE SCHEMA IF NOT EXISTS {s}"))
                conn.execute(self._q("CREATE TABLE {s}.schema_version (version INTEGER NOT NULL)"))
                conn.execute(self._q("INSERT INTO {s}.schema_version VALUES (0)"))
                conn.execute(self._q(_SCHEMA))
            for v in range(max(version, 1) + 1, SCHEMA_VERSION + 1):
                conn.execute(self._q(_MIGRATIONS[v]))
            conn.execute(self._q("UPDATE {s}.schema_version SET version = %s"), (SCHEMA_VERSION,))

    # -- plumbing ---------------------------------------------------------------------

    async def _open_pool(self) -> AsyncConnectionPool[AsyncConnection[TupleRow]]:
        if self._closed:
            raise StoreError(f"store {self} is closed")
        if self._pool is not None:
            return self._pool
        if self._pool_lock is None:
            self._pool_lock = asyncio.Lock()
        async with self._pool_lock:
            if self._closed:
                raise StoreError(f"store {self} is closed")
            if self._pool is None:
                pool: AsyncConnectionPool[AsyncConnection[TupleRow]] = AsyncConnectionPool(
                    self.conninfo,
                    min_size=self._min_size,
                    max_size=self._max_size,
                    timeout=self._timeout_s,
                    # Replaces connections dropped while idle (a server restart, a proxy's
                    # idle timeout) instead of failing the next operation on them.
                    check=AsyncConnectionPool.check_connection,
                    name=f"jevex-store-{self.db_schema}",
                    open=False,
                )
                await pool.open()
                if self._closed:  # aclose() ran while the pool was opening
                    await pool.close()
                    raise StoreError(f"store {self} is closed")
                self._pool = pool
        return self._pool

    async def _run[T](self, op: Callable[[AsyncConnection[Any]], Awaitable[T]]) -> T:
        """Run ``op`` as one transaction, committed if it returns and rolled back if it
        raises. It runs to the end even if the caller is cancelled."""
        pool = await self._open_pool()

        async def run() -> T:
            try:
                async with pool.connection() as conn:
                    return await op(conn)
            except psycopg.Error as exc:
                raise StoreError(f"store {self}: {exc}") from exc

        task = asyncio.ensure_future(run())
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        task.add_done_callback(_settle)
        return await asyncio.shield(task)

    async def _rows(
        self, query: sql.Composed, params: tuple[Any, ...] = ()
    ) -> list[dict[str, Any]]:
        async def op(conn: AsyncConnection[Any]) -> list[dict[str, Any]]:
            cur = conn.cursor(row_factory=dict_row)
            await cur.execute(query, params)
            return await cur.fetchall()

        return await self._run(op)

    # -- generators -------------------------------------------------------------------

    _SELECT_GENERATORS: LiteralString = (
        "SELECT g.*, d.generator_id IS NOT NULL AS disabled FROM {s}.generators g "
        "LEFT JOIN {s}.generator_disables d ON d.generator_id = g.id"
    )

    async def put_generator(self, generator: GeneratorRecord) -> None:
        params = (
            generator.id,
            generator.field,
            _json(generator.spec),
            _json(generator.scope),
            _aware(generator.created_at),
        )

        async def op(conn: AsyncConnection[Any]) -> None:
            await conn.execute(
                self._q(
                    "INSERT INTO {s}.generators VALUES (%s, %s, %s, %s, %s) "
                    "ON CONFLICT (id) DO UPDATE SET field = excluded.field, "
                    "spec = excluded.spec, scope = excluded.scope, "
                    "created_at = excluded.created_at"
                ),
                params,
            )
            await self._set_enabled(conn, generator.id, generator.enabled)

        await self._run(op)

    @staticmethod
    def _generator(row: dict[str, Any]) -> GeneratorRecord:
        return GeneratorRecord(
            id=row["id"],
            field=row["field"],
            spec=row["spec"],
            scope=row["scope"],
            enabled=not row["disabled"],
            created_at=_utc(row["created_at"]),
        )

    async def get_generator(self, generator_id: str) -> GeneratorRecord | None:
        rows = await self._rows(
            self._q(self._SELECT_GENERATORS + " WHERE g.id = %s"), (generator_id,)
        )
        return self._generator(rows[0]) if rows else None

    async def generators(
        self, field: str | None = None, *, include_disabled: bool = False
    ) -> list[GeneratorRecord]:
        where: list[LiteralString] = []
        params: list[Any] = []
        if field is not None:
            where.append("g.field = %s")
            params.append(field)
        if not include_disabled:
            where.append("d.generator_id IS NULL")
        query = self._SELECT_GENERATORS
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY g.created_at, g.id " + _C
        return [self._generator(r) for r in await self._rows(self._q(query), tuple(params))]

    async def _set_enabled(
        self, conn: AsyncConnection[Any], generator_id: str, enabled: bool
    ) -> None:
        if enabled:
            query = "DELETE FROM {s}.generator_disables WHERE generator_id = %s"
        else:
            query = "INSERT INTO {s}.generator_disables VALUES (%s) ON CONFLICT DO NOTHING"
        await conn.execute(self._q(query), (generator_id,))

    async def set_generator_enabled(self, generator_id: str, enabled: bool) -> None:
        await self._run(lambda conn: self._set_enabled(conn, generator_id, enabled))

    async def disabled_generator_ids(self) -> set[str]:
        rows = await self._rows(self._q("SELECT generator_id FROM {s}.generator_disables"))
        return {r["generator_id"] for r in rows}

    # -- key mappings -----------------------------------------------------------------

    async def put_key_mapping(self, mapping: KeyMapping) -> None:
        key = (mapping.fingerprint, mapping.schema_name, mapping.path)
        params = (
            *key,
            mapping.field,
            _json(mapping.normalisers),
            mapping.unsure,
            _aware(mapping.created_at),
        )

        async def op(conn: AsyncConnection[Any]) -> None:
            await conn.execute(
                self._q(
                    "INSERT INTO {s}.key_mappings (fingerprint, schema_name, path, field, "
                    "normalisers, unsure, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (fingerprint, schema_name, path) DO UPDATE SET "
                    "field = excluded.field, normalisers = excluded.normalisers, "
                    "unsure = excluded.unsure, created_at = excluded.created_at"
                ),
                params,
            )
            await conn.execute(
                self._q(
                    "DELETE FROM {s}.key_path_unsure "
                    "WHERE fingerprint = %s AND schema_name = %s AND path = %s"
                ),
                key,
            )

        await self._run(op)

    async def key_mappings(
        self, fingerprint: str, *, schema: str | None = None
    ) -> list[KeyMapping]:
        query: LiteralString = "SELECT * FROM {s}.key_mappings WHERE fingerprint = %s"
        params: tuple[Any, ...] = (fingerprint,)
        if schema is not None:
            query += " AND schema_name = %s"
            params = (fingerprint, schema)
        query += f" ORDER BY path {_C}, schema_name {_C}"
        return [
            KeyMapping(
                fingerprint=r["fingerprint"],
                schema=r["schema_name"],
                path=r["path"],
                field=r["field"],
                normalisers=r["normalisers"],
                unsure=r["unsure"],
                created_at=_utc(r["created_at"]),
            )
            for r in await self._rows(self._q(query), params)
        ]

    async def count_unsure_key_paths(
        self, fingerprint: str, schema: str, paths: list[str]
    ) -> dict[str, int]:
        query = self._q(
            "INSERT INTO {s}.key_path_unsure AS u VALUES (%s, %s, %s, 1) "
            "ON CONFLICT (fingerprint, schema_name, path) "
            "DO UPDATE SET count = u.count + 1 RETURNING count"
        )

        async def op(conn: AsyncConnection[Any]) -> dict[str, int]:
            counts: dict[str, int] = {}
            # Sorted, so workers counting overlapping paths lock rows in the same order.
            for path in sorted(set(paths)):
                row = await (await conn.execute(query, (fingerprint, schema, path))).fetchone()
                assert row is not None  # RETURNING always gives the row
                counts[path] = int(row[0])
            return {path: counts[path] for path in dict.fromkeys(paths)}

        return await self._run(op)

    # -- verified examples ------------------------------------------------------------

    async def add_example(self, example: VerifiedExample) -> None:
        start, end = example.evidence or (None, None)
        params = (
            example.id,
            example.field,
            example.statement,
            _json(example.value),
            start,
            end,
            _json(example.context),
            example.source,
            example.probability,
            _aware(example.created_at),
        )

        async def op(conn: AsyncConnection[Any]) -> None:
            await conn.execute(
                self._q(
                    "INSERT INTO {s}.examples "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (id) DO UPDATE SET field = excluded.field, "
                    "statement = excluded.statement, value = excluded.value, "
                    "evidence_start = excluded.evidence_start, "
                    "evidence_end = excluded.evidence_end, context = excluded.context, "
                    "source = excluded.source, probability = excluded.probability, "
                    "created_at = excluded.created_at"
                ),
                params,
            )

        await self._run(op)

    async def examples(self, field: str, *, limit: int | None = None) -> list[VerifiedExample]:
        # LIMIT NULL is no limit.
        query = "SELECT * FROM {s}.examples WHERE field = %s ORDER BY created_at DESC, id "
        return [
            VerifiedExample(
                id=r["id"],
                field=r["field"],
                statement=r["statement"],
                value=r["value"],
                evidence=(
                    None
                    if r["evidence_start"] is None
                    else (r["evidence_start"], r["evidence_end"])
                ),
                context=r["context"],
                source=r["source"],
                probability=r["probability"],
                created_at=_utc(r["created_at"]),
            )
            for r in await self._rows(self._q(query + _C + " LIMIT %s"), (field, limit))
        ]

    # -- generator stats --------------------------------------------------------------

    async def record_generator_stats(
        self, generator_id: str, *, documents: int = 0, hits: int = 0, wins: int = 0
    ) -> None:
        async def op(conn: AsyncConnection[Any]) -> None:
            await conn.execute(
                self._q(
                    "INSERT INTO {s}.generator_stats AS t VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (generator_id) DO UPDATE SET "
                    "documents = t.documents + excluded.documents, "
                    "hits = t.hits + excluded.hits, wins = t.wins + excluded.wins"
                ),
                (generator_id, documents, hits, wins),
            )

        await self._run(op)

    async def generator_stats(self, generator_id: str) -> GeneratorStats:
        rows = await self._rows(
            self._q("SELECT * FROM {s}.generator_stats WHERE generator_id = %s"),
            (generator_id,),
        )
        if not rows:
            return GeneratorStats(generator_id=generator_id)
        r = rows[0]
        return GeneratorStats(
            generator_id=generator_id, documents=r["documents"], hits=r["hits"], wins=r["wins"]
        )

    # -- spend ledger -----------------------------------------------------------------

    def _insert_spend(self) -> sql.Composed:
        return self._q(
            "INSERT INTO {s}.spend (amount_nano_usd, kind, run_id, note, at) "
            "VALUES (%s, %s, %s, %s, %s)"
        )

    @staticmethod
    def _spend_params(entry: SpendEntry) -> tuple[Any, ...]:
        return (_nano(entry.amount_usd), entry.kind, entry.run_id, entry.note, _aware(entry.at))

    @staticmethod
    def _spend_filter(
        since: datetime | None, kind: str | None, run_id: str | None
    ) -> tuple[LiteralString, tuple[Any, ...]]:
        where: list[LiteralString] = []
        params: list[Any] = []
        if since is not None:
            where.append("at >= %s")
            params.append(_aware(since))
        if kind is not None:
            where.append("kind = %s")
            params.append(kind)
        if run_id is not None:
            where.append("run_id = %s")
            params.append(run_id)
        return (" WHERE " + " AND ".join(where) if where else ""), tuple(params)

    async def record_spend(self, entry: SpendEntry) -> None:
        params = self._spend_params(entry)

        async def op(conn: AsyncConnection[Any]) -> None:
            await conn.execute(self._insert_spend(), params)

        await self._run(op)

    async def spend(
        self,
        *,
        since: datetime | None = None,
        kind: SpendKind | None = None,
        run_id: str | None = None,
    ) -> float:
        where, params = self._spend_filter(since, kind, run_id)
        query = self._q("SELECT COALESCE(SUM(amount_nano_usd), 0) AS nano FROM {s}.spend" + where)
        rows = await self._rows(query, params)
        return int(rows[0]["nano"]) / _NANO

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
        total = self._q("SELECT COALESCE(SUM(amount_nano_usd), 0), COUNT(*) FROM {s}.spend" + where)
        values = self._spend_params(entry)

        async def op(conn: AsyncConnection[Any]) -> bool:
            # Self-conflicting and held to commit: one check-and-insert at a time, while
            # plain reads carry on.
            await conn.execute(self._q("LOCK TABLE {s}.spend IN SHARE ROW EXCLUSIVE MODE"))
            row = await (await conn.execute(total, params)).fetchone()
            assert row is not None  # an aggregate always gives a row
            spent, count = int(row[0]), int(row[1])
            if cap is not None and spent + _nano(entry.amount_usd) > cap:
                return False
            if max_count is not None and count + 1 > max_count:
                return False
            await conn.execute(self._insert_spend(), values)
            return True

        return await self._run(op)

    async def aclose(self) -> None:
        """Wait for operations still running, then close the pool. Closing twice is fine."""
        self._closed = True
        if self._pending:
            await asyncio.wait(set(self._pending))
        pool, self._pool = self._pool, None
        if pool is not None:
            await pool.close()
