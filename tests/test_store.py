import asyncio
import multiprocessing
import sqlite3
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from jevex.store import (
    GeneratorRecord,
    KeyMapping,
    SpendEntry,
    SQLiteStore,
    Store,
    StoreError,
    VerifiedExample,
    open_store,
)

T0 = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SQLiteStore]:
    s = SQLiteStore(tmp_path / "jevex.db")
    yield s
    await s.aclose()


def gen(gid: str, field: str = "VehicleSpec.zero_to_62_s", **kw: object) -> GeneratorRecord:
    return GeneratorRecord.model_validate(
        {"id": gid, "field": field, "spec": {"match": {"regex": r"(\d+)"}}, **kw}
    )


# --- opening -------------------------------------------------------------------------


async def test_open_store_urls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    cases: list[tuple[str | Path, Path | str]] = [
        ("sqlite:///rel.db", Path("rel.db")),
        (f"sqlite:///{tmp_path / 'abs.db'}", tmp_path / "abs.db"),
        ("sqlite://:memory:", ":memory:"),
        (tmp_path / "p.db", tmp_path / "p.db"),
        ("bare.db", Path("bare.db")),
    ]
    for url, path in cases:
        store = open_store(url)
        assert isinstance(store, SQLiteStore)
        assert store.path == path
        await store.aclose()
    assert (tmp_path / "rel.db").exists()


def test_open_store_rejects_other_urls() -> None:
    with pytest.raises(StoreError, match="#36"):
        open_store("postgresql://localhost/jevex")
    with pytest.raises(StoreError, match="unsupported"):
        open_store("redis://localhost")


def test_sqlite_store_is_a_store(store: SQLiteStore) -> None:
    assert isinstance(store, Store)


def test_file_store_uses_wal(store: SQLiteStore, tmp_path: Path) -> None:
    conn = sqlite3.connect(tmp_path / "jevex.db")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    conn.close()


def test_newer_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "future.db"
    SQLiteStore(path)._conn.close()  # pyright: ignore[reportPrivateUsage]
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 99")
    conn.close()
    with pytest.raises(StoreError, match="v99"):
        SQLiteStore(path)


def test_unopenable_path_is_a_store_error(tmp_path: Path) -> None:
    with pytest.raises(StoreError, match="can't open"):
        SQLiteStore(tmp_path / "missing-dir" / "x.db")


async def test_data_survives_reopening(tmp_path: Path) -> None:
    path = tmp_path / "jevex.db"
    first = SQLiteStore(path)
    await first.put_generator(gen("g1", created_at=T0))
    await first.aclose()
    second = SQLiteStore(path)
    assert await second.get_generator("g1") == gen("g1", created_at=T0)
    await second.aclose()


# --- generators ----------------------------------------------------------------------


async def test_generator_round_trip_and_replace(store: SQLiteStore) -> None:
    g = gen("g1", scope={"locale": "en-GB"}, created_at=T0)
    await store.put_generator(g)
    assert await store.get_generator("g1") == g
    assert await store.get_generator("nope") is None

    replaced = g.model_copy(update={"spec": {"match": {"regex": "x"}}})
    await store.put_generator(replaced)
    assert await store.get_generator("g1") == replaced
    assert len(await store.generators()) == 1


async def test_generators_filter_by_field_and_enabled(store: SQLiteStore) -> None:
    await store.put_generator(gen("b", created_at=T0 + timedelta(seconds=1)))
    await store.put_generator(gen("a", created_at=T0))
    await store.put_generator(gen("c", field="Book.title", created_at=T0))
    await store.put_generator(gen("d", enabled=False, created_at=T0))

    assert [g.id for g in await store.generators("VehicleSpec.zero_to_62_s")] == ["a", "b"]
    assert [g.id for g in await store.generators()] == ["a", "c", "b"]
    everything = await store.generators(include_disabled=True)
    assert {g.id for g in everything} == {"a", "b", "c", "d"}


async def test_disable_and_enable_a_generator(store: SQLiteStore) -> None:
    await store.put_generator(gen("g1"))
    await store.set_generator_enabled("g1", False)
    assert await store.generators() == []
    assert (await store.get_generator("g1")).enabled is False  # type: ignore[union-attr]
    await store.set_generator_enabled("g1", True)
    assert [g.id for g in await store.generators()] == ["g1"]
    with pytest.raises(KeyError):
        await store.set_generator_enabled("missing", False)


async def test_naive_datetimes_are_rejected(store: SQLiteStore) -> None:
    with pytest.raises(ValueError, match="naive"):
        await store.put_generator(gen("g1", created_at=datetime(2026, 1, 1)))


# --- key mappings --------------------------------------------------------------------


async def test_key_mappings_per_fingerprint_replace_by_path(store: SQLiteStore) -> None:
    await store.put_key_mapping(
        KeyMapping(fingerprint="fp1", path="$.offers.price", field="Listing.price")
    )
    await store.put_key_mapping(
        KeyMapping(
            fingerprint="fp1",
            path="$.offers.price",
            field="Listing.price_gbp",
            normalisers=["parse_money", {"unit": {"from": "GBP", "to": "GBP"}}],
        )
    )
    await store.put_key_mapping(KeyMapping(fingerprint="fp1", path="$.name", field="Listing.title"))
    await store.put_key_mapping(KeyMapping(fingerprint="fp2", path="$.name", field="Book.title"))

    mappings = await store.key_mappings("fp1")
    assert [(m.path, m.field) for m in mappings] == [
        ("$.name", "Listing.title"),
        ("$.offers.price", "Listing.price_gbp"),
    ]
    assert mappings[1].normalisers == ["parse_money", {"unit": {"from": "GBP", "to": "GBP"}}]
    assert await store.key_mappings("unknown") == []


# --- examples ------------------------------------------------------------------------


async def test_examples_newest_first_with_limit(store: SQLiteStore) -> None:
    for i in range(3):
        await store.add_example(
            VerifiedExample(
                id=f"ex{i}",
                field="VehicleSpec.zero_to_62_s",
                statement=f"0-62 mph in {6 + i}.1 s",
                value=6.1 + i,
                evidence=(12, 15),
                context={"heading_trail": ["Performance"]},
                probability=0.97,
                created_at=T0 + timedelta(minutes=i),
            )
        )
    await store.add_example(
        VerifiedExample(id="other", field="Book.title", statement="Dune", value="Dune")
    )

    examples = await store.examples("VehicleSpec.zero_to_62_s")
    assert [e.id for e in examples] == ["ex2", "ex1", "ex0"]
    assert examples[0].evidence == (12, 15)
    assert examples[0].context == {"heading_trail": ["Performance"]}
    assert examples[0].created_at == T0 + timedelta(minutes=2)
    assert [e.id for e in await store.examples("VehicleSpec.zero_to_62_s", limit=2)] == [
        "ex2",
        "ex1",
    ]


async def test_example_values_come_back_in_json_form(store: SQLiteStore) -> None:
    await store.add_example(
        VerifiedExample(
            id="price",
            field="Listing.price",
            statement="£19,995",
            value=Decimal("19995.00"),
            source="human",
        )
    )
    (example,) = await store.examples("Listing.price")
    assert example.value == "19995.00"
    assert example.evidence is None
    assert example.source == "human"


# --- stats ---------------------------------------------------------------------------


async def test_generator_stats_accumulate(store: SQLiteStore) -> None:
    empty = await store.generator_stats("g1")
    assert (empty.documents, empty.hits, empty.wins) == (0, 0, 0)
    assert empty.hit_rate is None
    assert empty.win_rate is None

    await store.record_generator_stats("g1", documents=4, hits=2, wins=1)
    await store.record_generator_stats("g1", documents=6, hits=3)
    stats = await store.generator_stats("g1")
    assert (stats.documents, stats.hits, stats.wins) == (10, 5, 1)
    assert stats.hit_rate == 0.5
    assert stats.win_rate == 0.2


async def test_concurrent_tasks_do_not_lose_increments(store: SQLiteStore) -> None:
    await asyncio.gather(*(store.record_generator_stats("g1", documents=1) for _ in range(200)))
    assert (await store.generator_stats("g1")).documents == 200


# --- spend ledger --------------------------------------------------------------------


async def test_spend_filters(store: SQLiteStore) -> None:
    await store.record_spend(SpendEntry(amount_usd=0.25, kind="jev", run_id="r1", at=T0))
    await store.record_spend(
        SpendEntry(amount_usd=1.0, kind="llm", run_id="r1", at=T0 + timedelta(hours=1))
    )
    await store.record_spend(
        SpendEntry(amount_usd=0.5, kind="llm", run_id="r2", at=T0 + timedelta(hours=2))
    )

    assert await store.spend() == pytest.approx(1.75)
    assert await store.spend(kind="llm") == pytest.approx(1.5)
    assert await store.spend(run_id="r1") == pytest.approx(1.25)
    assert await store.spend(since=T0 + timedelta(hours=1)) == pytest.approx(1.5)
    assert await store.spend(since=T0 + timedelta(hours=1), kind="llm", run_id="r2") == (
        pytest.approx(0.5)
    )
    assert await store.spend(since=T0 + timedelta(days=1)) == 0.0


async def test_try_spend_respects_the_cap(store: SQLiteStore) -> None:
    assert await store.try_spend(SpendEntry(amount_usd=0.6, kind="llm", at=T0), cap_usd=1.0)
    assert not await store.try_spend(SpendEntry(amount_usd=0.5, kind="llm", at=T0), cap_usd=1.0)
    assert await store.try_spend(SpendEntry(amount_usd=0.4, kind="jev", at=T0), cap_usd=1.0)
    assert await store.spend() == pytest.approx(1.0)
    # Spend before ``since`` doesn't count against the cap (a new period).
    later = T0 + timedelta(days=1)
    assert await store.try_spend(
        SpendEntry(amount_usd=0.9, kind="llm", at=later), cap_usd=1.0, since=later
    )


async def test_concurrent_try_spend_never_overshoots(store: SQLiteStore) -> None:
    results = await asyncio.gather(
        *(store.try_spend(SpendEntry(amount_usd=0.1, kind="llm"), cap_usd=1.0) for _ in range(30))
    )
    assert sum(results) == 10
    assert await store.spend() == pytest.approx(1.0)


# --- several processes ---------------------------------------------------------------


def _worker(path: str, rounds: int) -> None:
    async def run() -> None:
        store = SQLiteStore(Path(path), busy_timeout_s=60)
        for _ in range(rounds):
            await store.record_generator_stats("shared", documents=1, hits=1)
            await store.try_spend(SpendEntry(amount_usd=0.01, kind="jev"), cap_usd=0.5)
        await store.aclose()

    asyncio.run(run())


def test_several_processes_share_one_database(tmp_path: Path) -> None:
    path = tmp_path / "shared.db"
    SQLiteStore(path)._conn.close()  # pyright: ignore[reportPrivateUsage]
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_worker, args=(str(path), 40)) for _ in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
    assert [p.exitcode for p in procs] == [0, 0, 0, 0]

    conn = sqlite3.connect(path)
    documents, hits = conn.execute(
        "SELECT documents, hits FROM generator_stats WHERE generator_id = 'shared'"
    ).fetchone()
    count, total = conn.execute("SELECT COUNT(*), SUM(amount_micro_usd) FROM spend").fetchone()
    conn.close()
    assert (documents, hits) == (160, 160)
    # 160 attempts at $0.01 against a $0.50 cap: exactly 50 recorded, never more.
    assert count == 50
    assert total == 500_000
