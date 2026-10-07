"""The shared spend ledger suite (:mod:`jevex.store.ledger`): every built-in ledger, the
stores and :class:`MemoryLedger`, runs it."""

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from jevex.store import MemoryLedger, SpendEntry, SpendLedger, SQLiteStore, Store

T0 = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)


@pytest.fixture(params=["sqlite-file", "sqlite-memory", "postgres", "memory"])
async def ledger(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[SpendLedger]:
    if request.param == "memory":
        yield MemoryLedger()
        return
    if request.param == "postgres":
        pytest.importorskip("psycopg")
        from jevex.store.postgres import PostgresStore

        url: str = request.getfixturevalue("postgres_url")
        s: Store = PostgresStore(url, db_schema=request.getfixturevalue("pg_schema"))
    else:
        s = SQLiteStore(tmp_path / "jevex.db" if request.param == "sqlite-file" else ":memory:")
    assert isinstance(s, SpendLedger)
    yield s
    await s.aclose()


def charge(usd: float, kind: str = "llm", **kw: object) -> SpendEntry:
    return SpendEntry.model_validate({"amount_usd": usd, "kind": kind, **kw})


def test_is_a_ledger(ledger: SpendLedger) -> None:
    assert isinstance(ledger, SpendLedger)


def test_a_ledger_is_not_a_store() -> None:
    assert not isinstance(MemoryLedger(), Store)


async def test_spend_filters(ledger: SpendLedger) -> None:
    await ledger.record_spend(charge(0.25, "jev", run_id="r1", at=T0))
    await ledger.record_spend(charge(1.0, run_id="r1", at=T0 + timedelta(hours=1)))
    await ledger.record_spend(charge(0.5, run_id="r2", at=T0 + timedelta(hours=2)))

    assert await ledger.spend() == pytest.approx(1.75)
    assert await ledger.spend(kind="llm") == pytest.approx(1.5)
    assert await ledger.spend(run_id="r1") == pytest.approx(1.25)
    assert await ledger.spend(since=T0 + timedelta(hours=1)) == pytest.approx(1.5)
    assert await ledger.spend(since=T0 + timedelta(hours=1), kind="llm", run_id="r2") == (
        pytest.approx(0.5)
    )
    assert await ledger.spend(since=T0 + timedelta(days=1)) == 0.0


async def test_spend_entries_oldest_first_with_filters(ledger: SpendLedger) -> None:
    await ledger.record_spend(charge(0.5, run_id="r2", at=T0 + timedelta(hours=2), note="n"))
    await ledger.record_spend(charge(0.25, "jev", run_id="r1", at=T0))
    await ledger.record_spend(charge(1.0, run_id="r1", at=T0 + timedelta(hours=1)))

    entries = await ledger.spend_entries()
    assert [(e.amount_usd, e.kind, e.at) for e in entries] == [
        (0.25, "jev", T0),
        (1.0, "llm", T0 + timedelta(hours=1)),
        (0.5, "llm", T0 + timedelta(hours=2)),
    ]
    assert (entries[2].note, entries[2].run_id) == ("n", "r2")
    assert [e.amount_usd for e in await ledger.spend_entries(kind="llm")] == [1.0, 0.5]
    assert [e.amount_usd for e in await ledger.spend_entries(since=T0 + timedelta(hours=1))] == [
        1.0,
        0.5,
    ]
    assert await ledger.spend_entries(since=T0 + timedelta(days=1)) == []


async def test_tiny_jev_charges_add_up_exactly(ledger: SpendLedger) -> None:
    # Jev costs $0.042 per million tokens: ten tokens is 420 nano-dollars.
    for _ in range(100):
        await ledger.record_spend(charge(10 * 0.042 / 1_000_000, "jev"))
    assert await ledger.spend() == pytest.approx(0.000042, rel=1e-9)


async def test_try_spend_respects_the_cap(ledger: SpendLedger) -> None:
    assert await ledger.try_spend(charge(0.6, at=T0), cap_usd=1.0)
    assert not await ledger.try_spend(charge(0.5, at=T0), cap_usd=1.0)
    assert await ledger.try_spend(charge(0.4, "jev", at=T0), cap_usd=1.0)
    assert await ledger.spend() == pytest.approx(1.0)
    # Spend before ``since`` doesn't count against the cap (a new period).
    later = T0 + timedelta(days=1)
    assert await ledger.try_spend(charge(0.9, at=later), cap_usd=1.0, since=later)
    # No limits: the entry is simply recorded.
    assert await ledger.try_spend(charge(5.0, at=later))


async def test_try_spend_caps_per_kind(ledger: SpendLedger) -> None:
    await ledger.record_spend(charge(0.9, "llm", at=T0))
    # Jev has its own cap: LLM spend doesn't count against it.
    assert await ledger.try_spend(charge(0.5, "jev", at=T0), cap_usd=0.6, kind="jev")
    assert not await ledger.try_spend(charge(0.2, "jev", at=T0), cap_usd=0.6, kind="jev")
    assert not await ledger.try_spend(charge(0.2, "llm", at=T0), cap_usd=1.0, kind="llm")
    with pytest.raises(ValueError, match="llm entry"):
        await ledger.try_spend(charge(0.1, "llm"), cap_usd=1.0, kind="jev")


async def test_try_spend_max_count_is_a_rate_limit(ledger: SpendLedger) -> None:
    now = T0
    for _ in range(3):
        assert await ledger.try_spend(charge(0.0, at=now), max_count=3, since=now, kind="llm")
    assert not await ledger.try_spend(charge(0.0, at=now), max_count=3, since=now, kind="llm")
    # The next minute is a new window.
    later = now + timedelta(minutes=1)
    assert await ledger.try_spend(charge(0.0, at=later), max_count=3, since=later, kind="llm")


async def test_llm_call_entries_count_calls_but_add_no_spend(ledger: SpendLedger) -> None:
    await ledger.record_spend(charge(1.25, kind="llm", at=T0))
    for _ in range(2):
        assert await ledger.try_spend(
            SpendEntry(amount_usd=0, kind="llm_call", at=T0), max_count=2, since=T0, kind="llm_call"
        )
    assert not await ledger.try_spend(
        SpendEntry(amount_usd=0, kind="llm_call", at=T0), max_count=2, since=T0, kind="llm_call"
    )
    assert await ledger.spend() == pytest.approx(1.25)
    assert await ledger.spend(kind="llm_call") == 0.0
    # llm_call rows don't count against an llm rate or spend limit.
    assert await ledger.try_spend(charge(0.0, kind="llm", at=T0), max_count=2, since=T0, kind="llm")


async def test_try_spend_rejects_bad_limits(ledger: SpendLedger) -> None:
    for cap in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="cap_usd"):
            await ledger.try_spend(charge(0.1), cap_usd=cap)
    with pytest.raises(ValueError, match="max_count"):
        await ledger.try_spend(charge(0.1), max_count=-1)
    assert await ledger.try_spend(charge(0.1), cap_usd=1e30)  # huge caps are fine


async def test_concurrent_try_spend_never_overshoots(ledger: SpendLedger) -> None:
    results = await asyncio.gather(*(ledger.try_spend(charge(0.1), cap_usd=1.0) for _ in range(30)))
    assert sum(results) == 10
    assert await ledger.spend() == pytest.approx(1.0)


@pytest.mark.parametrize("amount", [-0.01, float("nan"), float("inf"), 1e20])
def test_spend_entries_must_be_finite_charges(amount: float) -> None:
    with pytest.raises(ValidationError):
        charge(amount)


async def test_naive_times_are_rejected(ledger: SpendLedger) -> None:
    with pytest.raises(ValueError, match="naive"):
        await ledger.record_spend(charge(0.1, at=datetime(2026, 1, 1)))
    with pytest.raises(ValueError, match="naive"):
        await ledger.spend(since=datetime(2026, 1, 1))
    assert await ledger.spend() == 0.0
