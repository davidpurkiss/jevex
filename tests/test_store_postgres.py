"""Postgres-only store behaviour. The shared store suite in ``test_store.py`` runs on
Postgres too (its ``postgres`` param)."""

import asyncio
import multiprocessing
import sys
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from multiprocessing.synchronize import Barrier
from typing import LiteralString, cast

import pytest
from pydantic import BaseModel

pytest.importorskip("psycopg")

import psycopg

from jevex import Extractor
from jevex.store import (
    GeneratorRecord,
    KeyMapping,
    SpendEntry,
    StoreError,
    VerifiedExample,
    open_store,
)
from jevex.store.postgres import SCHEMA_VERSION, PostgresStore

T0 = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)


class Car(BaseModel):
    name: str


def gen(gid: str, **kw: object) -> GeneratorRecord:
    return GeneratorRecord.model_validate(
        {"id": gid, "field": "VehicleSpec.zero_to_62_s", "spec": {"regex": r"(\d+)"}, **kw}
    )


def charge(usd: float) -> SpendEntry:
    return SpendEntry(amount_usd=usd, kind="jev")


def query(url: str, sql: str) -> list[tuple[object, ...]]:
    with psycopg.connect(url, autocommit=True) as conn:
        cur = conn.execute(cast("LiteralString", sql))
        return cur.fetchall() if cur.description else []


@pytest.fixture
def default_schema(postgres_url: str) -> Iterator[None]:
    """For tests whose store uses the default ``jevex`` schema; drops it afterwards."""
    yield
    query(postgres_url, "DROP SCHEMA IF EXISTS jevex CASCADE")


async def test_open_store_opens_postgres_urls(postgres_url: str, default_schema: None) -> None:
    for url in (postgres_url, postgres_url.replace("postgresql://", "postgres://", 1)):
        store = open_store(url)
        assert isinstance(store, PostgresStore)
        assert store.db_schema == "jevex"
        await store.record_generator_stats("g1", documents=1)
        await store.aclose()
    assert query(postgres_url, "SELECT documents FROM jevex.generator_stats") == [(2,)]


async def test_an_extractor_opens_and_closes_a_postgres_url(
    postgres_url: str, default_schema: None
) -> None:
    extractor = Extractor(schemas=[Car], store=postgres_url)
    store = await extractor.store()
    assert isinstance(store, PostgresStore)
    ledger = await extractor.ledger()
    assert ledger.store is store
    await store.record_spend(charge(0.5))
    await extractor.aclose()
    with pytest.raises(StoreError, match="closed"):
        await store.spend()


def test_open_store_without_the_extra_says_what_to_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "jevex.store.postgres", None)
    with pytest.raises(StoreError, match=r"jevex\[postgres\]"):
        open_store("postgresql://localhost/jevex")


async def test_tables_live_in_the_store_schema(postgres_url: str, pg_schema: str) -> None:
    store = PostgresStore(postgres_url, db_schema=pg_schema)
    await store.aclose()
    tables = query(
        postgres_url,
        "SELECT table_name FROM information_schema.tables "
        f"WHERE table_schema = '{pg_schema}' ORDER BY table_name",
    )
    assert [t for (t,) in tables] == [
        "examples",
        "generator_disables",
        "generator_stats",
        "generators",
        "key_mappings",
        "key_path_unsure",
        "schema_version",
        "spend",
    ]
    assert query(postgres_url, f"SELECT version FROM {pg_schema}.schema_version") == [
        (SCHEMA_VERSION,)
    ]


async def test_stores_in_different_schemas_are_separate(postgres_url: str, pg_schema: str) -> None:
    other_schema = f"{pg_schema}_other"
    first = PostgresStore(postgres_url, db_schema=pg_schema)
    second = PostgresStore(postgres_url, db_schema=other_schema)
    try:
        await first.put_generator(gen("g1"))
        assert await second.generators() == []
        assert [g.id for g in await first.generators()] == ["g1"]
    finally:
        await first.aclose()
        await second.aclose()
        query(postgres_url, f"DROP SCHEMA IF EXISTS {other_schema} CASCADE")


async def test_data_survives_reopening(postgres_url: str, pg_schema: str) -> None:
    first = PostgresStore(postgres_url, db_schema=pg_schema)
    await first.put_generator(gen("g1", created_at=T0))
    await first.set_generator_enabled("pack:x", False)
    await first.aclose()
    second = PostgresStore(postgres_url, db_schema=pg_schema)
    assert await second.get_generator("g1") == gen("g1", created_at=T0)
    assert await second.disabled_generator_ids() == {"pack:x"}
    await second.aclose()


async def test_newer_schema_is_refused(postgres_url: str, pg_schema: str) -> None:
    await PostgresStore(postgres_url, db_schema=pg_schema).aclose()
    query(postgres_url, f"UPDATE {pg_schema}.schema_version SET version = 99")
    with pytest.raises(StoreError, match="v99"):
        PostgresStore(postgres_url, db_schema=pg_schema)


def test_an_unreachable_server_is_a_store_error_without_the_password() -> None:
    url = f"postgresql://jevex:s3cret@/jevex?host=/nonexistent-{uuid.uuid4().hex}"
    with pytest.raises(StoreError, match="can't open") as info:
        PostgresStore(url, timeout_s=1)
    assert "s3cret" not in str(info.value)


async def test_times_come_back_in_utc_whatever_the_session_zone(
    postgres_url: str, pg_schema: str
) -> None:
    sep = "&" if "?" in postgres_url else "?"
    url = f"{postgres_url}{sep}options=-c%20TimeZone%3DAsia%2FTokyo"
    store = PostgresStore(url, db_schema=pg_schema)
    await store.put_key_mapping(
        KeyMapping(fingerprint="fp", schema="S", path="$.a", field="a", created_at=T0)
    )
    [mapping] = await store.key_mappings("fp")
    assert mapping.created_at == T0
    assert mapping.created_at.tzinfo is UTC
    await store.aclose()


async def test_a_failed_write_rolls_back_and_the_store_keeps_working(
    postgres_url: str, pg_schema: str
) -> None:
    store = PostgresStore(postgres_url, db_schema=pg_schema)
    bad = VerifiedExample(id="e1", field="S.f", statement="9.1\x00s", value=9.1)
    with pytest.raises(StoreError, match="NUL"):
        await store.add_example(bad)
    assert await store.examples("S.f") == []
    with pytest.raises(StoreError):
        await store.put_generator(gen("g1", spec={"bad": float("nan")}))
    assert await store.get_generator("g1") is None
    await store.add_example(VerifiedExample(id="e2", field="S.f", statement="9.1 s", value=9.1))
    assert [e.id for e in await store.examples("S.f")] == ["e2"]
    await store.aclose()


async def test_a_cancelled_write_still_commits(postgres_url: str, pg_schema: str) -> None:
    store = PostgresStore(postgres_url, db_schema=pg_schema)
    await store.record_generator_stats("g1", documents=1)  # opens the pool
    task = asyncio.create_task(store.try_spend(charge(0.25), cap_usd=1.0))
    await asyncio.sleep(0)  # the write is sent
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await store.aclose()  # waits for it
    reopened = PostgresStore(postgres_url, db_schema=pg_schema)
    assert await reopened.spend() == 0.25
    await reopened.aclose()


async def test_calls_after_closing_raise(postgres_url: str, pg_schema: str) -> None:
    store = PostgresStore(postgres_url, db_schema=pg_schema)
    await store.aclose()  # never used: no pool to close
    with pytest.raises(StoreError, match="closed"):
        await store.generators()


# --- several processes ---------------------------------------------------------------


def _open_and_write(url: str, schema: str, barrier: Barrier) -> None:
    async def run() -> None:
        barrier.wait()
        store = PostgresStore(url, db_schema=schema)
        await store.record_generator_stats("opened", documents=1)
        await store.aclose()

    asyncio.run(run())


def test_several_processes_can_create_a_new_store_together(
    postgres_url: str, pg_schema: str
) -> None:
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(6)
    procs = [
        ctx.Process(target=_open_and_write, args=(postgres_url, pg_schema, barrier))
        for _ in range(6)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
    assert [p.exitcode for p in procs] == [0] * 6
    assert query(postgres_url, f"SELECT documents FROM {pg_schema}.generator_stats") == [(6,)]


def _worker(url: str, schema: str, rounds: int, barrier: Barrier) -> None:
    async def run() -> None:
        store = PostgresStore(url, db_schema=schema)
        barrier.wait()
        for _ in range(rounds):
            await asyncio.gather(
                store.record_generator_stats("shared", documents=1, hits=1),
                store.try_spend(charge(0.01), cap_usd=2.0),
                store.count_unsure_key_paths("fp", "S", ["$.b", "$.a"]),
                store.count_unsure_key_paths("fp", "S", ["$.a", "$.b"]),
            )
        await store.aclose()

    asyncio.run(run())


def test_several_processes_share_one_store(postgres_url: str, pg_schema: str) -> None:
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(6)
    procs = [
        ctx.Process(target=_worker, args=(postgres_url, pg_schema, 50, barrier)) for _ in range(6)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
    assert [p.exitcode for p in procs] == [0] * 6

    assert query(postgres_url, f"SELECT documents, hits FROM {pg_schema}.generator_stats") == [
        (300, 300)
    ]
    # 300 attempts at $0.01 against a $2 cap, crossed mid-run: exactly 200 recorded.
    assert query(postgres_url, f"SELECT COUNT(*), SUM(amount_nano_usd) FROM {pg_schema}.spend") == [
        (200, 2 * 1_000_000_000)
    ]
    # Overlapping paths counted in either order, without deadlocks or lost counts.
    assert query(
        postgres_url, f"SELECT path, count FROM {pg_schema}.key_path_unsure ORDER BY path"
    ) == [("$.a", 600), ("$.b", 600)]
