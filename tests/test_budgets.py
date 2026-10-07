import asyncio
import logging
from collections.abc import AsyncIterator, Iterator, Sequence
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
    RunLedger,
)
from jevex.budgets import period_start
from jevex.jev import JevBackendError, Noul
from jevex.llm import LLMImage, LLMResponse, LLMUsage, reset_process_llm_cost
from jevex.pipeline import Context, SchemaRun, for_each_schema
from jevex.store import SpendEntry, SQLiteStore, StoreError
from jevex.testing import FakeJev, FakeLLM


class Title(BaseModel):
    title: str


class Car(BaseModel):
    """A car."""

    model: str = Field(description="Model name")


class Boat(BaseModel):
    """A boat."""

    name: str = Field(description="Boat name")


@pytest.fixture(autouse=True)
def fresh_llm_spend(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("JEVEX_LLM_MAX_COST_USD", raising=False)
    reset_process_llm_cost()
    yield
    reset_process_llm_cost()


def llm(price: tuple[float, float] = (0.0, 0.0)) -> FakeLLM:
    return FakeLLM(lambda _p, _s: {"title": "Dune"}, price=price)


class ScriptedLLM:
    """An LLM whose cost can be unknown (``None``) and whose calls can take time."""

    def __init__(self, *, cost: float | None = 0.0, delay: float = 0.0, retries: int = 0) -> None:
        self.cost = cost
        self.delay = delay
        self.retries = retries
        self.calls = 0

    async def structured[T: BaseModel](
        self, prompt: str, schema: type[T], *, images: Sequence[LLMImage] = ()
    ) -> LLMResponse[T]:
        self.calls += 1
        await asyncio.sleep(self.delay)
        output = schema.model_validate({"title": "Dune"})
        return LLMResponse(
            output=output, usage=LLMUsage(10, 5, self.cost), model="scripted", retries=self.retries
        )


def doc_budget(**kw: object) -> DocumentBudget:
    return DocumentBudget(Budgets(per_document=DocBudget.model_validate(kw)))


# --- per document --------------------------------------------------------------------


async def test_max_llm_calls_stops_llm_use_for_the_document() -> None:
    budget = doc_budget(max_llm_calls=2)
    fake = llm()
    results = [await budget.call_llm(fake, "Title: Dune", Title) for _ in range(4)]
    assert [r is not None for r in results] == [True, True, False, False]
    assert len(fake.calls) == 2
    assert budget.llm_stopped
    [event] = budget.events
    assert (event.scope, event.limit) == ("document", "max_llm_calls")


async def test_the_adapters_retries_are_added_up() -> None:
    budget = doc_budget()
    flaky = ScriptedLLM(retries=2)
    for _ in range(3):
        await budget.call_llm(flaky, "x", Title)
    assert (budget.llm_calls, budget.llm_retries) == (3, 6)


async def test_concurrent_calls_cant_pass_the_call_cap_together() -> None:
    budget = doc_budget(max_llm_calls=2)
    slow = ScriptedLLM(delay=0.01)
    results = await asyncio.gather(*(budget.call_llm(slow, "x", Title) for _ in range(6)))
    assert sum(r is not None for r in results) == 2
    assert slow.calls == 2
    assert budget.llm_calls == 2


async def test_calls_waiting_on_the_ledger_dont_falsely_stop_llm_use(
    store: SQLiteStore,
) -> None:
    now = datetime.now(UTC)
    for _ in range(2):  # the rate limit is already full
        await store.record_spend(SpendEntry(amount_usd=0, kind="llm_call", at=now))
    budget = DocumentBudget(
        Budgets(per_document=DocBudget(max_llm_calls=2)), ledger(store, llm_rpm=2)
    )
    fake = ScriptedLLM()
    results = await asyncio.gather(*(budget.call_llm(fake, "x", Title) for _ in range(3)))
    assert results == [None, None, None]
    assert fake.calls == 0
    assert not budget.llm_stopped
    assert [e.limit for e in budget.events] == ["llm_rpm"]


class SlowLedger(RunLedger):
    """A ledger whose check takes time (or fails), to interrupt calls mid-reservation."""

    def __init__(self, *, delay: float = 0.1, error: Exception | None = None) -> None:
        super().__init__(RunBudget(llm_rpm=100), SQLiteStore(":memory:"))
        self.delay = delay
        self.error = error

    async def refuse_llm(self) -> None:
        await asyncio.sleep(self.delay)
        if self.error is not None:
            error, self.error = self.error, None  # fail once
            raise error


async def test_a_call_cancelled_during_the_ledger_check_frees_its_slot_and_wakes_waiters() -> None:
    budget = DocumentBudget(Budgets(per_document=DocBudget(max_llm_calls=1)), SlowLedger())
    fake = ScriptedLLM()
    first = asyncio.create_task(asyncio.wait_for(budget.call_llm(fake, "x", Title), 0.02))
    await asyncio.sleep(0)  # the first call now holds the only slot, pending
    second = asyncio.create_task(budget.call_llm(fake, "x", Title))
    with pytest.raises(TimeoutError):
        await first
    assert await asyncio.wait_for(second, 1.0) is not None  # woken, took the freed slot
    assert fake.calls == 1
    assert budget.llm_calls == 1
    assert budget.events == []


async def test_a_ledger_error_gives_the_slot_back() -> None:
    budget = DocumentBudget(
        Budgets(per_document=DocBudget(max_llm_calls=1)),
        SlowLedger(delay=0.0, error=StoreError("disk full")),
    )
    fake = ScriptedLLM()
    with pytest.raises(StoreError):
        await budget.call_llm(fake, "x", Title)
    assert budget.llm_calls == 0
    assert await budget.call_llm(fake, "x", Title) is not None
    assert budget.events == []


async def test_a_call_cap_hit_under_a_live_ledger_is_still_reported(store: SQLiteStore) -> None:
    budget = DocumentBudget(
        Budgets(per_document=DocBudget(max_llm_calls=2)), ledger(store, llm_rpm=100)
    )
    fake = ScriptedLLM()
    results = await asyncio.gather(*(budget.call_llm(fake, "x", Title) for _ in range(3)))
    assert sum(r is not None for r in results) == 2
    assert fake.calls == 2
    assert budget.llm_stopped
    assert [e.limit for e in budget.events] == ["max_llm_calls"]


async def test_a_slot_given_back_by_the_ledger_goes_to_a_waiting_call(
    store: SQLiteStore,
) -> None:
    now = datetime.now(UTC)
    await store.record_spend(SpendEntry(amount_usd=0, kind="llm_call", at=now))
    # One rpm slot left, a call cap of 1: the first call takes the rpm slot, so no call is
    # refused by the call cap, and the rest are rate-limited, not stopped.
    budget = DocumentBudget(
        Budgets(per_document=DocBudget(max_llm_calls=1)), ledger(store, llm_rpm=2)
    )
    fake = ScriptedLLM()
    results = await asyncio.gather(*(budget.call_llm(fake, "x", Title) for _ in range(3)))
    assert sum(r is not None for r in results) == 1
    assert budget.llm_calls == 1


async def test_a_failed_call_under_a_spend_cap_isnt_reported_as_unpriced() -> None:
    def boom(_p: str, _s: type[BaseModel]) -> object:
        raise RuntimeError("provider down")

    budget = doc_budget(max_spend=1.0)
    with pytest.raises(RuntimeError):
        await budget.call_llm(FakeLLM(boom), "x", Title)
    assert budget.llm_calls == 1
    assert budget.unpriced_calls == 0
    assert budget.events == []


async def test_max_spend_counts_actual_cost() -> None:
    # FakeLLM estimates tokens from text; $1M per million tokens makes each call > $1.
    budget = doc_budget(max_spend=1.0)
    fake = llm(price=(1_000_000, 1_000_000))
    assert await budget.call_llm(fake, "Title: Dune", Title) is not None
    assert budget.llm_spend > 1.0
    assert await budget.call_llm(fake, "Title: Dune", Title) is None
    assert [e.limit for e in budget.events] == ["max_spend"]


async def test_unpriced_calls_under_a_spend_cap_are_reported() -> None:
    budget = doc_budget(max_spend=0.0001)
    unpriced = ScriptedLLM(cost=None)
    for _ in range(3):
        assert await budget.call_llm(unpriced, "x", Title) is not None
    assert budget.unpriced_calls == 3
    [event] = budget.events
    assert event.limit == "unpriced_llm"
    assert event.message.startswith("3 LLM calls with unknown cost")


async def test_unpriced_calls_without_a_spend_cap_are_not_reported() -> None:
    budget = doc_budget(max_llm_calls=5)
    await budget.call_llm(ScriptedLLM(cost=None), "x", Title)
    assert budget.events == []


async def test_timeout_is_a_deadline_for_starting_llm_calls() -> None:
    budget = doc_budget(timeout_s=30)
    assert await budget.call_llm(llm(), "x", Title) is not None
    budget.started -= 31
    assert await budget.call_llm(llm(), "x", Title) is None
    assert [e.limit for e in budget.events] == ["timeout_s"]


async def test_a_failed_call_still_counts() -> None:
    def boom(_p: str, _s: type[BaseModel]) -> object:
        raise RuntimeError("provider down")

    budget = doc_budget(max_llm_calls=1)
    with pytest.raises(RuntimeError):
        await budget.call_llm(FakeLLM(boom), "x", Title)
    assert budget.llm_calls == 1
    assert await budget.call_llm(llm(), "x", Title) is None


class TooManyRequests(Exception):
    status_code = 429


async def test_a_call_failing_on_a_rate_limit_is_counted() -> None:
    def limited(_p: str, _s: type[BaseModel]) -> object:
        raise TooManyRequests("slow down")

    def down(_p: str, _s: type[BaseModel]) -> object:
        raise RuntimeError("provider down")

    budget = doc_budget()
    for answer in (limited, down, limited):
        with pytest.raises(Exception, match=r"slow down|provider down"):
            await budget.call_llm(FakeLLM(answer), "x", Title)
    assert (budget.llm_calls, budget.llm_rate_limited) == (3, 2)


async def test_the_first_hit_of_each_limit_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="jevex")
    budget = doc_budget(max_llm_calls=1)
    for _ in range(3):
        await budget.call_llm(llm(), "x", Title)
    assert [r.getMessage() for r in caplog.records] == [
        "document budget max_llm_calls hit: 1 LLM calls (the limit)"
    ]


async def test_the_process_backstop_stops_llm_use_instead_of_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget = DocumentBudget()
    fake = llm(price=(1_000_000, 1_000_000))
    assert await budget.call_llm(fake, "x", Title) is not None
    monkeypatch.setenv("JEVEX_LLM_MAX_COST_USD", "0.5")  # already spent more than that
    assert await budget.call_llm(fake, "x", Title) is None
    assert budget.llm_calls == 1  # the refused call made no request
    assert [(e.scope, e.limit) for e in budget.events] == [("process", "JEVEX_LLM_MAX_COST_USD")]
    assert await budget.call_llm(fake, "x", Title) is None
    assert len(fake.calls) == 1  # the backstop refuses before a request


async def test_no_budgets_means_unlimited() -> None:
    budget = DocumentBudget()
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


def ledger(store: SQLiteStore, run_id: str = "run-a", **kw: object) -> RunLedger:
    return RunLedger(RunBudget.model_validate(kw), store, run_id=run_id)


async def test_run_spend_is_shared_across_documents(store: SQLiteStore) -> None:
    fake = llm(price=(1_000_000, 1_000_000))
    budgets = Budgets(run=RunBudget(max_spend=1.0))
    first = DocumentBudget(budgets, ledger(store, max_spend=1.0))
    assert await first.call_llm(fake, "x", Title) is not None
    second = DocumentBudget(budgets, ledger(store, "run-b", max_spend=1.0))  # another worker
    assert await second.call_llm(fake, "x", Title) is None
    assert [(e.scope, e.limit) for e in second.events] == [("run", "max_spend")]
    assert await store.spend(kind="llm") == pytest.approx(first.llm_spend)


async def test_spend_before_the_period_doesnt_count(store: SQLiteStore) -> None:
    yesterday = datetime.now(UTC) - timedelta(days=1, hours=1)
    await store.record_spend(SpendEntry(amount_usd=100.0, kind="llm", at=yesterday))
    assert await ledger(store, max_spend=1.0, period="day").refuse_llm() is None


async def test_a_run_period_counts_only_its_own_run_id(store: SQLiteStore) -> None:
    await store.record_spend(SpendEntry(amount_usd=2.0, kind="llm", run_id="earlier-run"))
    mine = ledger(store, "this-run", max_spend=1.0, period="run")
    assert await mine.refuse_llm() is None
    await store.record_spend(SpendEntry(amount_usd=1.5, kind="llm", run_id="this-run"))
    refusal = await mine.refuse_llm()
    assert refusal is not None
    assert refusal.limit == "max_spend"


async def test_rate_limit_skips_calls_without_stopping_the_document(store: SQLiteStore) -> None:
    budget = DocumentBudget(Budgets(), ledger(store, llm_rpm=2))
    fake = llm()
    results = [await budget.call_llm(fake, "x", Title) is not None for _ in range(4)]
    assert results == [True, True, False, False]
    assert not budget.llm_stopped  # the next minute may have room
    assert budget.rpm_skips == 2
    [event] = budget.events
    assert event.limit == "llm_rpm"
    assert event.message == "2 calls skipped: over 2 LLM calls a minute"
    assert budget.llm_calls == 2  # skipped calls gave their slots back


async def test_calls_a_minute_ago_dont_count_against_the_rate(store: SQLiteStore) -> None:
    old = datetime.now(UTC) - timedelta(minutes=2)
    for _ in range(2):
        await store.record_spend(SpendEntry(amount_usd=0, kind="llm_call", at=old))
    assert await ledger(store, llm_rpm=2).refuse_llm() is None


async def test_a_run_ledger_works_without_a_document(store: SQLiteStore) -> None:
    run = ledger(store, max_spend=1.0)
    fake = llm(price=(1_000_000, 1_000_000))
    assert await run.call_llm(fake, "x", Title) is not None
    assert await run.call_llm(fake, "x", Title) is None
    assert len(fake.calls) == 1


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
class FoundThenAsksTwice:
    """Records a value, then asks Jev two separate questions."""

    name: str = "select"

    async def run(self, ctx: Context) -> None:
        ctx.schemas["Car"].values["document"] = {"model": "Golf"}
        await ctx.jev.ask("state one", {"q": Noul(instructions="first?")})
        await ctx.jev.ask("state two", {"q": Noul(instructions="second?")})
        ctx.schemas["Car"].values["document"] = {"model": "Polo"}


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
        pipeline=Pipeline([FoundThenAsksTwice()]),
        budgets=Budgets(per_document=DocBudget(max_jev_requests=1)),
    )
    result = await ex.extract(doc())
    assert result.meta.stopped
    assert [(e.scope, e.limit) for e in result.meta.budget_events] == [
        ("document", "max_jev_requests")
    ]
    assert result.meta.jev.requests == 1
    assert result.values == {"Car": {"document": {"model": "Golf"}}}  # found before the cap


@dataclass
class TwoBranches:
    """One branch per schema: Car asks twice (hitting a cap of 1), Boat asks slowly."""

    name: str = "select"
    late: list[str] | None = None

    async def run(self, ctx: Context) -> None:
        async def branch(run: SchemaRun) -> None:
            name = run.name
            if name == "Car":
                await ctx.jev.ask("a", {"q": Noul(instructions="a?")})
                await ctx.jev.ask("b", {"q": Noul(instructions="b?")})
            else:
                await asyncio.sleep(0.05)
                if self.late is not None:
                    self.late.append(name)
                ctx.schemas["Boat"].values["document"] = {"name": "late write"}

        await for_each_schema(ctx, branch)


async def test_a_capped_branch_cancels_its_siblings_before_the_result() -> None:
    late: list[str] = []
    ex = Extractor(
        [Car, Boat],
        jev=FakeJev().client(),
        pipeline=Pipeline([TwoBranches(late=late)]),
        budgets=Budgets(per_document=DocBudget(max_jev_requests=1)),
    )
    result = await ex.extract(doc())
    await asyncio.sleep(0.1)
    assert result.meta.stopped
    assert late == []  # the Boat branch was cancelled, not left running
    assert "Boat" not in result.values


@dataclass
class AsksThenFails:
    name: str = "select"

    async def run(self, ctx: Context) -> None:
        await ctx.jev.ask("s", {"q": Noul(instructions="a?")})
        raise JevBackendError("backend down")


async def test_jev_spend_is_recorded_even_when_a_stage_fails(store: SQLiteStore) -> None:
    ex = Extractor(
        [Car],
        jev=FakeJev().client(),
        pipeline=Pipeline([AsksThenFails()]),
        budgets=Budgets(run=RunBudget(max_jev_spend=10.0)),
        store=store,
    )
    result = await ex.extract(doc())
    assert result.status == "failed"
    assert isinstance(result.cause, JevBackendError)
    assert await store.spend(kind="jev") > 0


async def test_llm_use_and_budget_events_reach_document_meta() -> None:
    ex = Extractor(
        [Car],
        jev=FakeJev().client(),
        pipeline=Pipeline([UsesLLM()]),
        budgets=Budgets(per_document=DocBudget(max_llm_calls=1)),
    )
    result = await ex.extract(doc())
    assert [e.message for e in result.meta.events] == ["called", "skipped", "skipped"]
    assert result.meta.llm.calls == 1
    meta = result.to_dict()["meta"]
    assert meta["budget_events"] == [
        {"scope": "document", "limit": "max_llm_calls", "message": "1 LLM calls (the limit)"}
    ]
    assert meta["llm"] == {
        "calls": 1,
        "cost": 0.0,
        "unpriced_calls": 0,
        "retries": 0,
        "rate_limited": 0,
    }
    assert not result.meta.stopped  # Jev and generators carry on


@dataclass
class AsksOnce:
    name: str = "select"

    async def run(self, ctx: Context) -> None:
        await ctx.jev.ask("s", {"q": Noul(instructions="a?")})


async def test_run_jev_spend_cap_skips_documents_once_reached(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'jevex.db'}"
    ex = Extractor(
        [Car],
        jev=FakeJev().client(),
        pipeline=Pipeline([AsksOnce()]),
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
    assert (await ex.ledger()).store is store
    await ex.aclose()


async def test_a_store_passed_in_is_not_closed(store: SQLiteStore) -> None:
    ex = Extractor([Car], jev=FakeJev().client(), store=store)
    assert await ex.store() is store
    await ex.aclose()
    assert await store.spend() == 0.0  # still open


def test_the_store_reopens_on_a_new_loop_after_aclose(tmp_path: Path) -> None:
    ex = Extractor([Car], jev=FakeJev().client(), store=f"sqlite:///{tmp_path / 'x.db'}")

    async def use() -> None:
        await asyncio.gather(*(ex.store() for _ in range(3)))
        await ex.aclose()

    asyncio.run(use())
    asyncio.run(use())  # the store lock isn't bound to the first loop


async def test_an_owned_store_is_closed_even_if_the_jev_backend_fails_to_close(
    tmp_path: Path,
) -> None:
    class BadBackend(FakeJev):
        async def aclose(self) -> None:
            raise RuntimeError("close failed")

    ex = Extractor([Car], jev=BadBackend().client(), store=f"sqlite:///{tmp_path / 'x.db'}")
    store = await ex.store()
    assert isinstance(store, SQLiteStore)
    with pytest.raises(RuntimeError, match="close failed"):
        await ex.aclose()
    assert ex._store is None  # pyright: ignore[reportPrivateUsage]


async def test_no_budgets_and_no_store_opens_nothing() -> None:
    ex = Extractor([Car], jev=FakeJev().client(), pipeline=Pipeline([]))
    assert await ex.store() is None
    result = await ex.extract(doc())
    assert result.meta.budget_events == []


# --- images --------------------------------------------------------------------------------

PICTURE = LLMImage(b"\x89PNG", "image/png")


class TextOnlyLLM:
    """An LLM written before images: no ``images`` argument."""

    def __init__(self) -> None:
        self.calls = 0

    async def structured(self, prompt: str, schema: type[Title]) -> LLMResponse[Title]:
        self.calls += 1
        return LLMResponse(Title(title="Dune"), LLMUsage(10, 5, 0.001), "old")


async def test_images_go_to_the_llm_and_the_call_is_metered_like_any_other(
    store: SQLiteStore,
) -> None:
    fake = llm(price=(1_000_000, 1_000_000))
    budgets = Budgets(per_document=DocBudget(max_llm_calls=1), run=RunBudget(max_spend=100.0))
    budget = DocumentBudget(budgets, ledger(store, max_spend=100.0))
    assert await budget.call_llm(fake, "Title?", Title, images=[PICTURE]) is not None
    assert await budget.call_llm(fake, "Title?", Title, images=[PICTURE]) is None
    [call] = fake.calls
    assert call.images == (PICTURE,)
    assert budget.llm_spend > 0
    assert await store.spend(kind="llm") == pytest.approx(budget.llm_spend)
    assert [e.limit for e in budget.events] == ["max_llm_calls"]

    response = await ledger(store, max_spend=100.0).call_llm(fake, "x", Title, images=[PICTURE])
    assert response is not None
    assert fake.calls[-1].images == (PICTURE,)


async def test_a_text_only_llm_still_serves_calls_without_images() -> None:
    old = TextOnlyLLM()
    response = await DocumentBudget().call_llm(old, "Title?", Title)  # pyright: ignore[reportArgumentType]
    assert response is not None
    assert await RunLedger().call_llm(old, "Title?", Title) is not None  # pyright: ignore[reportArgumentType]
    assert old.calls == 2
