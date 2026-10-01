import asyncio
import multiprocessing
import sqlite3
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from multiprocessing.synchronize import Barrier
from pathlib import Path

import pytest
from pydantic import ValidationError

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
from jevex.store.sqlite import _SCHEMA, SCHEMA_VERSION  # pyright: ignore[reportPrivateUsage]

T0 = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)


# Every backend runs the shared suite below.
@pytest.fixture(params=["sqlite-file", "sqlite-memory", "postgres"])
async def store(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[Store]:
    if request.param == "postgres":
        pytest.importorskip("psycopg")
        from jevex.store.postgres import PostgresStore

        url: str = request.getfixturevalue("postgres_url")
        s: Store = PostgresStore(url, db_schema=request.getfixturevalue("pg_schema"))
    else:
        s = SQLiteStore(tmp_path / "jevex.db" if request.param == "sqlite-file" else ":memory:")
    yield s
    await s.aclose()


def gen(gid: str, field: str = "VehicleSpec.zero_to_62_s", **kw: object) -> GeneratorRecord:
    return GeneratorRecord.model_validate(
        {"id": gid, "field": field, "spec": {"match": {"regex": r"(\d+)"}}, **kw}
    )


def charge(usd: float, kind: str = "llm", **kw: object) -> SpendEntry:
    return SpendEntry.model_validate({"amount_usd": usd, "kind": kind, **kw})


# === Shared store behaviour ===========================================================


def test_is_a_store(store: Store) -> None:
    assert isinstance(store, Store)


# --- generators ----------------------------------------------------------------------


async def test_generator_round_trip_and_replace(store: Store) -> None:
    g = gen("g1", scope={"locale": "en-GB"}, created_at=T0)
    await store.put_generator(g)
    assert await store.get_generator("g1") == g
    assert await store.get_generator("nope") is None

    replaced = g.model_copy(update={"spec": {"match": {"regex": "x"}}})
    await store.put_generator(replaced)
    assert await store.get_generator("g1") == replaced
    assert len(await store.generators()) == 1


async def test_generators_filter_by_field_and_enabled(store: Store) -> None:
    await store.put_generator(gen("b", created_at=T0 + timedelta(seconds=1)))
    await store.put_generator(gen("a", created_at=T0))
    await store.put_generator(gen("c", field="Book.title", created_at=T0))
    await store.put_generator(gen("d", enabled=False, created_at=T0))

    assert [g.id for g in await store.generators("VehicleSpec.zero_to_62_s")] == ["a", "b"]
    assert [g.id for g in await store.generators()] == ["a", "c", "b"]
    everything = await store.generators(include_disabled=True)
    assert {g.id for g in everything} == {"a", "b", "c", "d"}
    assert await store.disabled_generator_ids() == {"d"}


async def test_generators_created_together_order_by_id_bytes(store: Store) -> None:
    for gid in ("a", "B", "_x"):
        await store.put_generator(gen(gid, created_at=T0))
    assert [g.id for g in await store.generators()] == ["B", "_x", "a"]


async def test_disable_and_enable_a_generator(store: Store) -> None:
    await store.put_generator(gen("g1"))
    await store.set_generator_enabled("g1", False)
    assert await store.generators() == []
    g = await store.get_generator("g1")
    assert g is not None
    assert g.enabled is False
    await store.set_generator_enabled("g1", True)
    assert [g.id for g in await store.generators()] == ["g1"]
    assert await store.disabled_generator_ids() == set()


async def test_can_disable_a_generator_stored_elsewhere(store: Store) -> None:
    # A pack generator (another layer) is disabled without copying its spec here.
    await store.set_generator_enabled("pack:gen-7", False)
    await store.set_generator_enabled("pack:gen-7", False)  # idempotent
    assert await store.disabled_generator_ids() == {"pack:gen-7"}
    assert await store.get_generator("pack:gen-7") is None
    # If its spec is later stored here disabled, it stays disabled.
    await store.set_generator_enabled("pack:gen-8", False)
    await store.put_generator(gen("pack:gen-8", enabled=False))
    assert await store.generators() == []


async def test_putting_an_enabled_record_clears_a_disable(store: Store) -> None:
    await store.set_generator_enabled("g1", False)
    await store.put_generator(gen("g1"))
    assert [g.id for g in await store.generators()] == ["g1"]


async def test_naive_datetimes_are_rejected(store: Store) -> None:
    with pytest.raises(ValueError, match="naive"):
        await store.put_generator(gen("g1", created_at=datetime(2026, 1, 1)))


# --- key mappings --------------------------------------------------------------------


async def test_key_mappings_replace_by_fingerprint_schema_and_path(store: Store) -> None:
    await store.put_key_mapping(
        KeyMapping(fingerprint="fp1", schema="Listing", path="$.offers.price", field="price")
    )
    await store.put_key_mapping(
        KeyMapping(
            fingerprint="fp1",
            schema="Listing",
            path="$.offers.price",
            field="price_gbp",
            normalisers=["parse_money", {"unit": {"from": "GBP", "to": "GBP"}}],
        )
    )
    await store.put_key_mapping(
        KeyMapping(fingerprint="fp1", schema="Listing", path="$.name", field="title")
    )
    await store.put_key_mapping(
        KeyMapping(fingerprint="fp2", schema="Book", path="$.name", field="title")
    )

    mappings = await store.key_mappings("fp1")
    assert [(m.path, m.field) for m in mappings] == [
        ("$.name", "title"),
        ("$.offers.price", "price_gbp"),
    ]
    assert mappings[1].normalisers == ["parse_money", {"unit": {"from": "GBP", "to": "GBP"}}]
    assert mappings[1].schema_name == "Listing"
    assert await store.key_mappings("unknown") == []
    # Without a fingerprint: every mapping, by fingerprint then path.
    assert [(m.fingerprint, m.path) for m in await store.key_mappings()] == [
        ("fp1", "$.name"),
        ("fp1", "$.offers.price"),
        ("fp2", "$.name"),
    ]
    assert [m.fingerprint for m in await store.key_mappings(schema="Book")] == ["fp2"]


async def test_key_mappings_per_schema_and_none_answers(store: Store) -> None:
    # Two extractors with different schemas share one store and one fingerprint.
    await store.put_key_mapping(
        KeyMapping(fingerprint="fp", schema="Listing", path="$.name", field="title")
    )
    await store.put_key_mapping(
        KeyMapping(fingerprint="fp", schema="VehicleSpec", path="$.name", field="model")
    )
    # "none": Jev said the path is no field of this schema, so it's never asked again.
    await store.put_key_mapping(
        KeyMapping(fingerprint="fp", schema="Listing", path="$.sku", field=None)
    )

    listing = await store.key_mappings("fp", schema="Listing")
    assert [(m.path, m.field) for m in listing] == [("$.name", "title"), ("$.sku", None)]
    vehicle = await store.key_mappings("fp", schema="VehicleSpec")
    assert [(m.path, m.field) for m in vehicle] == [("$.name", "model")]
    assert len(await store.key_mappings("fp")) == 3


async def test_unsure_key_path_counts_accumulate_per_fingerprint_schema_and_path(
    store: Store,
) -> None:
    assert await store.count_unsure_key_paths("fp", "Listing", ["$.a", "$.b"]) == {
        "$.a": 1,
        "$.b": 1,
    }
    # A path named twice in one call counts once.
    assert await store.count_unsure_key_paths("fp", "Listing", ["$.a", "$.a"]) == {"$.a": 2}
    assert await store.count_unsure_key_paths("fp", "Book", ["$.a"]) == {"$.a": 1}
    assert await store.count_unsure_key_paths("fp2", "Listing", ["$.a"]) == {"$.a": 1}
    assert await store.count_unsure_key_paths("fp", "Listing", []) == {}


async def test_putting_a_key_mapping_clears_its_unsure_count(store: Store) -> None:
    await store.count_unsure_key_paths("fp", "Listing", ["$.a", "$.b"])
    await store.put_key_mapping(
        KeyMapping(fingerprint="fp", schema="Listing", path="$.a", field="title")
    )
    assert await store.count_unsure_key_paths("fp", "Listing", ["$.a", "$.b"]) == {
        "$.a": 1,
        "$.b": 2,
    }


async def test_concurrent_unsure_counts_are_not_lost(store: Store) -> None:
    await asyncio.gather(*(store.count_unsure_key_paths("fp", "S", ["$.a"]) for _ in range(20)))
    assert await store.count_unsure_key_paths("fp", "S", ["$.a"]) == {"$.a": 21}


async def test_an_unsure_none_round_trips_and_a_confident_mapping_replaces_it(
    store: Store,
) -> None:
    unsure = KeyMapping(fingerprint="fp", schema="Listing", path="$.a", field=None, unsure=True)
    await store.put_key_mapping(unsure)
    assert await store.key_mappings("fp") == [unsure]
    await store.put_key_mapping(
        KeyMapping(fingerprint="fp", schema="Listing", path="$.a", field="title")
    )
    [mapping] = await store.key_mappings("fp")
    assert (mapping.field, mapping.unsure) == ("title", False)


async def test_key_mappings_put_in_bulk(store: Store) -> None:
    await store.count_unsure_key_paths("fp", "Listing", ["$.a", "$.b"])
    await store.put_key_mapping(
        KeyMapping(fingerprint="fp", schema="Listing", path="$.a", field="old")
    )
    await store.put_key_mappings(
        [
            KeyMapping(fingerprint="fp", schema="Listing", path="$.a", field="title"),
            KeyMapping(fingerprint="fp", schema="Listing", path="$.c", field=None),
            # The same key again: the later mapping wins, as if put one by one.
            KeyMapping(
                fingerprint="fp", schema="Listing", path="$.c", field="price", normalisers=["x"]
            ),
            KeyMapping(fingerprint="fp", schema="Book", path="$.a", field="name"),
        ]
    )
    assert [(m.schema_name, m.path, m.field) for m in await store.key_mappings("fp")] == [
        ("Book", "$.a", "name"),
        ("Listing", "$.a", "title"),
        ("Listing", "$.c", "price"),
    ]
    assert (await store.key_mappings("fp", schema="Listing"))[1].normalisers == ["x"]
    # Each put path's unsure count is cleared; others keep theirs.
    assert await store.count_unsure_key_paths("fp", "Listing", ["$.a", "$.b"]) == {
        "$.a": 1,
        "$.b": 2,
    }
    # Any iterable will do, and nothing is a no-op.
    await store.put_key_mappings(
        KeyMapping(fingerprint="fp2", schema="Book", path=p, field=None) for p in ("$.x", "$.y")
    )
    await store.put_key_mappings([])
    assert [m.path for m in await store.key_mappings("fp2")] == ["$.x", "$.y"]


async def test_a_bulk_put_with_an_invalid_mapping_writes_none(store: Store) -> None:
    good = KeyMapping(fingerprint="fp", schema="Listing", path="$.a", field="title")
    naive = KeyMapping(
        fingerprint="fp", schema="Listing", path="$.b", field=None, created_at=datetime(2026, 1, 1)
    )
    with pytest.raises(ValueError, match="naive"):
        await store.put_key_mappings([good, naive])
    assert await store.key_mappings("fp") == []


async def test_concurrent_bulk_puts_of_overlapping_keys_all_land(store: Store) -> None:
    def batch(field: str) -> list[KeyMapping]:
        return [
            KeyMapping(fingerprint="fp", schema="S", path=f"$.{i}", field=field) for i in range(50)
        ]

    # Opposite orders: a backend locking rows as given would deadlock.
    await asyncio.gather(
        store.put_key_mappings(batch("a")), store.put_key_mappings(reversed(batch("b")))
    )
    mappings = await store.key_mappings("fp")
    assert len(mappings) == 50
    assert len({m.field for m in mappings}) == 1  # one batch wins every row: no interleaving


def test_an_unsure_key_mapping_maps_to_no_field() -> None:
    with pytest.raises(ValidationError, match="unsure key mapping maps to no field"):
        KeyMapping(fingerprint="fp", schema="Listing", path="$.a", field="title", unsure=True)


# --- examples ------------------------------------------------------------------------


async def test_json_values_come_back_as_written(store: Store) -> None:
    value = {"big": 1e20, "huge_int": 2**70, "nul": "a\x00b", "nested": [1.0, None, "é"]}
    await store.add_example(VerifiedExample(id="e1", field="S.f", statement="s", value=value))
    [example] = await store.examples("S.f")
    assert example.value == value
    assert isinstance(example.value["big"], float)
    assert isinstance(example.value["nested"][0], float)


async def test_examples_newest_first_with_limit(store: Store) -> None:
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
    # Without a field: every field's, still newest first.
    assert [e.id for e in await store.examples()][:2] == ["other", "ex2"]
    assert len(await store.examples()) == 4
    assert len(await store.examples(limit=3)) == 3


async def test_example_values_come_back_in_json_form(store: Store) -> None:
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
    assert example.document_source is None


async def test_an_examples_document_source_is_kept_and_replaced(store: Store) -> None:
    example = VerifiedExample(
        id="ex", field="S.f", statement="s", value=1, document_source="cars.example.com"
    )
    await store.add_example(example)
    assert await store.examples("S.f") == [example]
    # A human confirming the same answer from another site replaces it, source and all.
    human = example.model_copy(update={"source": "human", "document_source": None})
    await store.add_example(human)
    assert await store.examples("S.f") == [human]


# --- stats ---------------------------------------------------------------------------


async def test_generator_stats_accumulate(store: Store) -> None:
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


async def test_concurrent_tasks_do_not_lose_increments(store: Store) -> None:
    await asyncio.gather(*(store.record_generator_stats("g1", documents=1) for _ in range(200)))
    assert (await store.generator_stats("g1")).documents == 200


# --- spend ledger --------------------------------------------------------------------


async def test_spend_filters(store: Store) -> None:
    await store.record_spend(charge(0.25, "jev", run_id="r1", at=T0))
    await store.record_spend(charge(1.0, run_id="r1", at=T0 + timedelta(hours=1)))
    await store.record_spend(charge(0.5, run_id="r2", at=T0 + timedelta(hours=2)))

    assert await store.spend() == pytest.approx(1.75)
    assert await store.spend(kind="llm") == pytest.approx(1.5)
    assert await store.spend(run_id="r1") == pytest.approx(1.25)
    assert await store.spend(since=T0 + timedelta(hours=1)) == pytest.approx(1.5)
    assert await store.spend(since=T0 + timedelta(hours=1), kind="llm", run_id="r2") == (
        pytest.approx(0.5)
    )
    assert await store.spend(since=T0 + timedelta(days=1)) == 0.0


async def test_tiny_jev_charges_add_up_exactly(store: Store) -> None:
    # Jev costs $0.042 per million tokens: ten tokens is 420 nano-dollars.
    for _ in range(100):
        await store.record_spend(charge(10 * 0.042 / 1_000_000, "jev"))
    assert await store.spend() == pytest.approx(0.000042, rel=1e-9)


async def test_try_spend_respects_the_cap(store: Store) -> None:
    assert await store.try_spend(charge(0.6, at=T0), cap_usd=1.0)
    assert not await store.try_spend(charge(0.5, at=T0), cap_usd=1.0)
    assert await store.try_spend(charge(0.4, "jev", at=T0), cap_usd=1.0)
    assert await store.spend() == pytest.approx(1.0)
    # Spend before ``since`` doesn't count against the cap (a new period).
    later = T0 + timedelta(days=1)
    assert await store.try_spend(charge(0.9, at=later), cap_usd=1.0, since=later)
    # No limits: the entry is simply recorded.
    assert await store.try_spend(charge(5.0, at=later))


async def test_try_spend_caps_per_kind(store: Store) -> None:
    await store.record_spend(charge(0.9, "llm", at=T0))
    # Jev has its own cap: LLM spend doesn't count against it.
    assert await store.try_spend(charge(0.5, "jev", at=T0), cap_usd=0.6, kind="jev")
    assert not await store.try_spend(charge(0.2, "jev", at=T0), cap_usd=0.6, kind="jev")
    assert not await store.try_spend(charge(0.2, "llm", at=T0), cap_usd=1.0, kind="llm")
    with pytest.raises(ValueError, match="llm entry"):
        await store.try_spend(charge(0.1, "llm"), cap_usd=1.0, kind="jev")


async def test_try_spend_max_count_is_a_rate_limit(store: Store) -> None:
    now = T0
    for _ in range(3):
        assert await store.try_spend(charge(0.0, at=now), max_count=3, since=now, kind="llm")
    assert not await store.try_spend(charge(0.0, at=now), max_count=3, since=now, kind="llm")
    # The next minute is a new window.
    later = now + timedelta(minutes=1)
    assert await store.try_spend(charge(0.0, at=later), max_count=3, since=later, kind="llm")


async def test_llm_call_entries_count_calls_but_add_no_spend(store: Store) -> None:
    await store.record_spend(charge(1.25, kind="llm", at=T0))
    for _ in range(2):
        assert await store.try_spend(
            SpendEntry(amount_usd=0, kind="llm_call", at=T0), max_count=2, since=T0, kind="llm_call"
        )
    assert not await store.try_spend(
        SpendEntry(amount_usd=0, kind="llm_call", at=T0), max_count=2, since=T0, kind="llm_call"
    )
    assert await store.spend() == pytest.approx(1.25)
    assert await store.spend(kind="llm_call") == 0.0
    # llm_call rows don't count against an llm rate or spend limit.
    assert await store.try_spend(charge(0.0, kind="llm", at=T0), max_count=2, since=T0, kind="llm")


async def test_try_spend_rejects_bad_limits(store: Store) -> None:
    for cap in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="cap_usd"):
            await store.try_spend(charge(0.1), cap_usd=cap)
    with pytest.raises(ValueError, match="max_count"):
        await store.try_spend(charge(0.1), max_count=-1)
    assert await store.try_spend(charge(0.1), cap_usd=1e30)  # huge caps are fine


async def test_concurrent_try_spend_never_overshoots(store: Store) -> None:
    results = await asyncio.gather(*(store.try_spend(charge(0.1), cap_usd=1.0) for _ in range(30)))
    assert sum(results) == 10
    assert await store.spend() == pytest.approx(1.0)


@pytest.mark.parametrize("amount", [-0.01, float("nan"), float("inf"), 1e20])
def test_spend_entries_must_be_finite_charges(amount: float) -> None:
    with pytest.raises(ValidationError):
        charge(amount)


async def test_closed_store_raises(store: Store) -> None:
    await store.aclose()
    with pytest.raises(StoreError, match="closed"):
        await store.spend()
    await store.aclose()  # closing twice is fine


# === SQLite only ======================================================================


async def test_open_store_urls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    cases: list[tuple[str | Path, Path | str]] = [
        ("sqlite:///rel.db", Path("rel.db")),
        (f"sqlite:///{tmp_path / 'abs.db'}", tmp_path / "abs.db"),
        ("sqlite://:memory:", ":memory:"),
        ("sqlite:///:memory:", ":memory:"),
        (tmp_path / "p.db", tmp_path / "p.db"),
        (Path(":memory:"), ":memory:"),
        ("bare.db", Path("bare.db")),
    ]
    for url, path in cases:
        store = open_store(url)
        assert isinstance(store, SQLiteStore)
        assert store.path == path
        await store.aclose()
    assert (tmp_path / "rel.db").exists()
    assert not (tmp_path / ":memory:").exists()


def test_open_store_rejects_other_urls() -> None:
    with pytest.raises(StoreError, match="unsupported"):
        open_store("redis://localhost")


async def test_file_store_uses_wal(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "jevex.db")
    conn = sqlite3.connect(tmp_path / "jevex.db")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    conn.close()
    await store.aclose()


async def test_newer_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "future.db"
    await SQLiteStore(path).aclose()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 99")
    conn.close()
    with pytest.raises(StoreError, match="v99"):
        SQLiteStore(path)


async def test_a_version_1_database_is_migrated(tmp_path: Path) -> None:
    path = tmp_path / "v1.db"
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    conn.execute(
        "INSERT INTO key_mappings VALUES ('fp', 'Listing', '$.a', NULL, '[]', ?)",
        (T0.timestamp(),),
    )
    conn.execute(
        "INSERT INTO examples VALUES ('ex', 'S.f', 's', '1', NULL, NULL, '{}', 'llm', 0.9, ?)",
        (T0.timestamp(),),
    )
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()
    store = SQLiteStore(path)
    assert await store.key_mappings("fp") == [
        KeyMapping(fingerprint="fp", schema="Listing", path="$.a", field=None, created_at=T0)
    ]
    assert await store.count_unsure_key_paths("fp", "Listing", ["$.b"]) == {"$.b": 1}
    # v3: an example stored before document sources were recorded has none.
    assert await store.examples() == [
        VerifiedExample(
            id="ex", field="S.f", statement="s", value=1, probability=0.9, created_at=T0
        )
    ]
    await store.add_example(
        VerifiedExample(id="new", field="S.f", statement="t", value=2, document_source="a.com")
    )
    assert [e.document_source for e in await store.examples(limit=1)] == ["a.com"]
    await store.aclose()
    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 3
    conn.close()


def test_unopenable_path_is_a_store_error(tmp_path: Path) -> None:
    with pytest.raises(StoreError, match="can't open"):
        SQLiteStore(tmp_path / "missing-dir" / "x.db")


async def test_data_survives_reopening(tmp_path: Path) -> None:
    path = tmp_path / "jevex.db"
    first = SQLiteStore(path)
    await first.put_generator(gen("g1", created_at=T0))
    await first.set_generator_enabled("pack:x", False)
    await first.aclose()
    second = SQLiteStore(path)
    assert await second.get_generator("g1") == gen("g1", created_at=T0)
    assert await second.disabled_generator_ids() == {"pack:x"}
    await second.aclose()


async def test_a_failed_commit_rolls_back_and_releases_the_lock(tmp_path: Path) -> None:
    path = tmp_path / "jevex.db"
    store = SQLiteStore(path, busy_timeout_s=1)
    conn = store._conn  # pyright: ignore[reportPrivateUsage]
    assert conn is not None
    # A deferred foreign key is checked at COMMIT, so the trigger makes COMMIT fail.
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("CREATE TEMP TABLE parent (id INTEGER PRIMARY KEY)")
    conn.execute(
        "CREATE TEMP TABLE child (pid INTEGER REFERENCES parent (id) DEFERRABLE INITIALLY DEFERRED)"
    )
    conn.execute(
        "CREATE TEMP TRIGGER fail_commit AFTER INSERT ON main.generator_stats "
        "BEGIN INSERT INTO child VALUES (999); END"
    )
    with pytest.raises(StoreError, match="FOREIGN KEY"):
        await store.record_generator_stats("g1", documents=1)
    assert not conn.in_transaction

    # Another process could write straight away: the lock was released.
    other = sqlite3.connect(path, timeout=1)
    other.execute("BEGIN IMMEDIATE")
    other.execute("ROLLBACK")
    other.close()

    conn.execute("DROP TRIGGER fail_commit")
    await store.record_generator_stats("g1", documents=1)
    assert (await store.generator_stats("g1")).documents == 1
    await store.aclose()


async def test_store_calls_do_not_use_the_shared_executor(tmp_path: Path) -> None:
    path = tmp_path / "jevex.db"
    store = SQLiteStore(path, busy_timeout_s=5)
    blocker = sqlite3.connect(path)
    blocker.execute("BEGIN IMMEDIATE")  # every store write now waits for the lock
    pending = [
        asyncio.ensure_future(store.record_generator_stats("g", documents=1)) for _ in range(40)
    ]
    try:
        await asyncio.sleep(0.05)
        # asyncio's default executor (DNS lookups, other to_thread work) is still free.
        await asyncio.wait_for(asyncio.to_thread(lambda: None), timeout=1)
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
    await asyncio.gather(*pending)
    assert (await store.generator_stats("g")).documents == 40
    await store.aclose()


async def test_a_bulk_put_is_one_transaction(tmp_path: Path) -> None:
    path = tmp_path / "jevex.db"
    store = SQLiteStore(path)
    await store.count_unsure_key_paths("fp", "S", ["$.a"])
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TRIGGER no_bad BEFORE INSERT ON key_mappings WHEN NEW.path = '$.bad' "
        "BEGIN SELECT RAISE(ABORT, 'bad path'); END"
    )
    conn.commit()
    conn.close()
    batch = [KeyMapping(fingerprint="fp", schema="S", path=p, field=None) for p in ("$.a", "$.bad")]
    with pytest.raises(StoreError, match="bad path"):
        await store.put_key_mappings(batch)
    # The row written before the failure is rolled back, and its unsure count kept.
    assert await store.key_mappings("fp") == []
    assert await store.count_unsure_key_paths("fp", "S", ["$.a"]) == {"$.a": 2}
    await store.aclose()


# --- several processes ---------------------------------------------------------------


def _open_and_write(path: str, barrier: Barrier) -> None:
    async def run() -> None:
        barrier.wait()
        store = SQLiteStore(Path(path), busy_timeout_s=60)
        await store.record_generator_stats("opened", documents=1)
        await store.aclose()

    asyncio.run(run())


def test_several_processes_can_create_a_new_database_together(tmp_path: Path) -> None:
    ctx = multiprocessing.get_context("spawn")
    for trial in range(3):
        path = tmp_path / f"fresh-{trial}.db"
        barrier = ctx.Barrier(8)
        procs = [ctx.Process(target=_open_and_write, args=(str(path), barrier)) for _ in range(8)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=120)
        assert [p.exitcode for p in procs] == [0] * 8
        conn = sqlite3.connect(path)
        (documents,) = conn.execute(
            "SELECT documents FROM generator_stats WHERE generator_id = 'opened'"
        ).fetchone()
        conn.close()
        assert documents == 8


def _worker(path: str, rounds: int, barrier: Barrier) -> None:
    async def run() -> None:
        store = SQLiteStore(Path(path), busy_timeout_s=60)
        barrier.wait()
        for _ in range(rounds):
            await store.record_generator_stats("shared", documents=1, hits=1)
            await store.try_spend(charge(0.01, "jev"), cap_usd=2.0)
        await store.aclose()

    asyncio.run(run())


def test_several_processes_share_one_database(tmp_path: Path) -> None:
    path = tmp_path / "shared.db"
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(8)
    procs = [ctx.Process(target=_worker, args=(str(path), 60, barrier)) for _ in range(8)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
    assert [p.exitcode for p in procs] == [0] * 8

    conn = sqlite3.connect(path)
    documents, hits = conn.execute(
        "SELECT documents, hits FROM generator_stats WHERE generator_id = 'shared'"
    ).fetchone()
    count, total = conn.execute("SELECT COUNT(*), SUM(amount_nano_usd) FROM spend").fetchone()
    conn.close()
    assert (documents, hits) == (480, 480)
    # 480 attempts at $0.01 against a $2 cap, crossed mid-run: exactly 200 recorded.
    assert count == 200
    assert total == 2 * 1_000_000_000
