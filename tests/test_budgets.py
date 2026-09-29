from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import BaseModel

from jevex import (
    Budgets,
    DocBudget,
    Document,
    DocumentBudget,
    Extractor,
    Field,
    Pipeline,
    RunBudget,
)
from jevex.budgets import period_start
from jevex.jev import JevClient, JevRequestCapError, Noul
from jevex.llm import reset_process_llm_cost
from jevex.pipeline import Context
from jevex.store import SpendEntry, SQLiteStore
from jevex.testing import FakeJev, FakeLLM


class Title(BaseModel):
    title: str


class Car(BaseModel):
    """A car."""

    model: str = Field(description="Model name")


@pytest.fixture(autouse=True)
def fresh_llm_spend(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("JEVEX_LLM_MAX_COST_USD", raising=False)
    reset_process_llm_cost()
    yield
    reset_process_llm_cost()


def llm(price: tuple[float, float] = (0.0, 0.0)) -> FakeLLM:
    return FakeLLM(lambda _p, _s: {"title": "Dune"}, price=price)


# --- per document --------------------------------------------------------------------


async def test_max_llm_calls_stops_llm_use_for_the_document() -> None:
    budget = DocumentBudget(Budgets(per_document=DocBudget(max_llm_calls=2)))
    fake = llm()
    results = [await budget.call_llm(fake, "Title: Dune", Title) for _ in range(4)]
    assert [r is not None for r in results] == [True, True, False, False]
    assert len(fake.calls) == 2
    assert budget.llm_stopped
    [event] = budget.events
    assert (event.scope, event.limit) == ("document", "max_llm_calls")


async def test_max_spend_counts_actual_cost() -> None:
    # FakeLLM estimates tokens from text; $1M per million tokens makes each call > $1.
    budget = DocumentBudget(Budgets(per_document=DocBudget(max_spend=1.0)))
    fake = llm(price=(1_000_000, 1_000_000))
    assert await budget.call_llm(fake, "Title: Dune", Title) is not None
    assert budget.llm_spend > 1.0
    assert await budget.call_llm(fake, "Title: Dune", Title) is None
    assert [e.limit for e in budget.events] == ["max_spend"]


async def test_timeout_is_a_deadline_for_starting_llm_calls() -> None:
    budget = DocumentBudget(Budgets(per_document=DocBudget(timeout_s=30)))
    assert await budget.allow_llm()
    budget.started -= 31
    assert not await budget.allow_llm()
    assert [e.limit for e in budget.events] == ["timeout_s"]


async def test_a_failed_call_still_counts() -> None:
    def boom(_p: str, _s: type[BaseModel]) -> object:
        raise RuntimeError("provider down")

    budget = DocumentBudget(Budgets(per_document=DocBudget(max_llm_calls=1)))
    with pytest.raises(RuntimeError):
        await budget.call_llm(FakeLLM(boom), "x", Title)
    assert budget.llm_calls == 1
    assert await budget.call_llm(llm(), "x", Title) is None


async def test_no_budgets_means_unlimited() -> None:
    budget = DocumentBudget(Budgets())
    fake = llm()
    for _ in range(20):
        assert await budget.call_llm(fake, "x", Title) is not None
    assert budget.events == []


# --- per run, through the store's ledger ---------------------------------------------


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SQLiteStore]:
    s = SQLiteStore(tmp_path / "ledger.db")
    yield s
    await s.aclose()


async def test_run_spend_is_shared_across_documents(store: SQLiteStore) -> None:
    budgets = Budgets(run=RunBudget(max_spend=1.0, period="day"))
    fake = llm(price=(1_000_000, 1_000_000))
    first = DocumentBudget(budgets, store, "run-a")
    assert await first.call_llm(fake, "x", Title) is not None
    second = DocumentBudget(budgets, store, "run-b")  # another worker, same ledger
    assert await second.call_llm(fake, "x", Title) is None
    assert [(e.scope, e.limit) for e in second.events] == [("run", "max_spend")]
    assert await store.spend(kind="llm") == pytest.approx(first.llm_spend)


async def test_spend_before_the_period_doesnt_count(store: SQLiteStore) -> None:
    yesterday = datetime.now(UTC) - timedelta(days=1, hours=1)
    await store.record_spend(SpendEntry(amount_usd=100.0, kind="llm", at=yesterday))
    budget = DocumentBudget(Budgets(run=RunBudget(max_spend=1.0, period="day")), store)
    assert await budget.allow_llm()


async def test_rate_limit_skips_calls_without_stopping_the_document(store: SQLiteStore) -> None:
    budget = DocumentBudget(Budgets(run=RunBudget(llm_rpm=2)), store)
    assert [await budget.allow_llm() for _ in range(3)] == [True, True, False]
    assert not budget.llm_stopped  # the next minute may have room
    assert [e.limit for e in budget.events] == ["llm_rpm"]
    # Calls a minute ago no longer count.
    old = datetime.now(UTC) - timedelta(minutes=2)
    other = SQLiteStore(":memory:")
    await other.record_spend(SpendEntry(amount_usd=0, kind="llm_call", at=old))
    await other.record_spend(SpendEntry(amount_usd=0, kind="llm_call", at=old))
    assert await DocumentBudget(Budgets(run=RunBudget(llm_rpm=2)), other).allow_llm()
    await other.aclose()


def test_period_starts() -> None:
    now = datetime(2026, 9, 30, 14, 25, 7, tzinfo=UTC)  # a Wednesday
    started = datetime(2026, 9, 29, 8, 0, tzinfo=UTC)
    assert period_start("hour", now, started) == datetime(2026, 9, 30, 14, tzinfo=UTC)
    assert period_start("day", now, started) == datetime(2026, 9, 30, tzinfo=UTC)
    assert period_start("week", now, started) == datetime(2026, 9, 28, tzinfo=UTC)
    assert period_start("month", now, started) == datetime(2026, 9, 1, tzinfo=UTC)
    assert period_start("run", now, started) == started


# --- the extractor -------------------------------------------------------------------


@dataclass
class AskTwice:
    """Asks Jev two separate questions, then records a value."""

    name: str = "select"

    async def run(self, ctx: Context) -> None:
        await ctx.jev.ask("state one", {"q": Noul(instructions="first?")})
        await ctx.jev.ask("state two", {"q": Noul(instructions="second?")})
        ctx.schemas["Car"].values["document"] = {"model": "Golf"}


@dataclass
class UsesLLM:
    name: str = "fallback"

    async def run(self, ctx: Context) -> None:
        assert ctx.budget is not None
        for _ in range(3):
            response = await ctx.budget.call_llm(llm(), "Title: Dune", Title)
            ctx.event(self.name, "llm", "called" if response else "skipped")


def doc() -> Document:
    return Document.from_bytes(b"<p>Golf</p>", url="https://cars.test/golf")


async def test_jev_request_cap_stops_the_document_and_keeps_its_results() -> None:
    ex = Extractor(
        [Car],
        jev=FakeJev().client(),
        pipeline=Pipeline([AskTwice()]),
        budgets=Budgets(per_document=DocBudget(max_jev_requests=1)),
    )
    result = await ex.extract(doc())
    assert result.meta.stopped
    assert [(e.scope, e.limit) for e in result.meta.budget_events] == [
        ("document", "max_jev_requests")
    ]
    assert result.meta.jev.requests == 1
    uncapped = Extractor([Car], jev=FakeJev().client(), pipeline=Pipeline([AskTwice()]))
    assert (await uncapped.extract(doc())).values == {"Car": {"document": {"model": "Golf"}}}


async def test_jev_request_cap_error_on_the_client() -> None:
    jev = JevClient(FakeJev(), max_requests=1)
    await jev.ask("s", {"q": Noul(instructions="a?")})
    with pytest.raises(JevRequestCapError):
        await jev.ask("s", {"q": Noul(instructions="b?")})


async def test_llm_budget_events_reach_document_meta() -> None:
    ex = Extractor(
        [Car],
        jev=FakeJev().client(),
        pipeline=Pipeline([UsesLLM()]),
        budgets=Budgets(per_document=DocBudget(max_llm_calls=1)),
    )
    result = await ex.extract(doc())
    assert [e.message for e in result.meta.events] == ["called", "skipped", "skipped"]
    assert result.to_dict()["meta"]["budget_events"] == [
        {"scope": "document", "limit": "max_llm_calls", "message": "1 LLM calls (the limit)"}
    ]
    assert not result.meta.stopped  # Jev and generators carry on


async def test_run_jev_spend_cap_skips_documents_once_reached(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'jevex.db'}"
    ex = Extractor(
        [Car],
        jev=FakeJev().client(),
        pipeline=Pipeline([AskTwice()]),
        budgets=Budgets(run=RunBudget(max_jev_spend=1e-9)),
        store=url,
    )
    first = await ex.extract(doc())
    assert not first.meta.stopped
    store = await ex.store()
    assert store is not None
    assert await store.spend(kind="jev") == pytest.approx(first.meta.jev.cost)
    second = await ex.extract(doc())
    assert second.meta.stopped
    assert second.meta.jev.requests == 0
    assert [e.limit for e in second.meta.budget_events] == ["max_jev_spend"]
    await ex.aclose()
    assert ex._store is None  # pyright: ignore[reportPrivateUsage]


async def test_a_run_budget_without_a_store_uses_an_in_memory_ledger() -> None:
    ex = Extractor([Car], jev=FakeJev().client(), budgets=Budgets(run=RunBudget(llm_rpm=5)))
    store = await ex.store()
    assert isinstance(store, SQLiteStore)
    assert store.path == ":memory:"
    await ex.aclose()


async def test_a_store_passed_in_is_not_closed(store: SQLiteStore) -> None:
    ex = Extractor([Car], jev=FakeJev().client(), store=store)
    assert await ex.store() is store
    await ex.aclose()
    assert await store.spend() == 0.0  # still open


async def test_no_budgets_and_no_store_opens_nothing() -> None:
    ex = Extractor([Car], jev=FakeJev().client(), pipeline=Pipeline([]))
    assert await ex.store() is None
    result = await ex.extract(doc())
    assert result.meta.budget_events == []
