from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from jevex import Budgets, DocBudget, Document, Extractor, Field, Pipeline, RunBudget
from jevex.errors import ExtractionError, PartError
from jevex.extractor import document_stat
from jevex.images import UnreadableImageError
from jevex.jev import (
    JevBackendError,
    JevBudgetExceededError,
    JevClient,
    JevResponse,
    JevTransientError,
    JSONContent,
    Noul,
    Question,
    RetryPolicy,
)
from jevex.layout_pdf import PdfLayoutError
from jevex.pipeline import Context
from jevex.results import FieldMeta
from jevex.store import (
    MAX_STAT_VALUE_CHARS,
    DocumentEvent,
    DocumentStat,
    SpendEntry,
    SQLiteStore,
    StoreError,
    ValueStat,
)
from jevex.testing import FakeJev, FakeLLM


class Car(BaseModel):
    """A car."""

    model: str = Field(description="Model name")
    power_ps: int | None = Field(None, description="Power in PS")


@pytest.fixture
async def store() -> AsyncIterator[SQLiteStore]:
    s = SQLiteStore(":memory:")
    yield s
    await s.aclose()


def doc() -> Document:
    return Document.from_bytes(b"<p>Golf</p>", url="https://cars.test/golf")


@dataclass
class Finds:
    """Asks Jev once, calls the LLM once (through the budget), and finds two values."""

    name: str = "select"

    async def run(self, ctx: Context) -> None:
        await ctx.jev.ask("s", {"q": Noul(instructions="a?")})
        assert ctx.budget is not None
        await ctx.budget.call_llm(FakeLLM(lambda _p, _s: {"model": "x"}), "p", Car)
        await ctx.budget.call_llm(FakeLLM(lambda _p, _s: {"model": "x"}), "p", Car)
        run = ctx.schemas["Car"]
        run.set_field("document", "model", FieldMeta(value="Golf " * 50, method="jev"))
        run.set_field("document", "power_ps", FieldMeta(value=150, method="llm", confidence=0.7))


@dataclass
class Fails:
    name: str = "select"

    async def run(self, ctx: Context) -> None:
        await ctx.jev.ask("s", {"q": Noul(instructions="a?")})
        raise JevBackendError("backend down")


@dataclass
class OverCap:
    name: str = "select"

    async def run(self, ctx: Context) -> None:
        await ctx.jev.ask("s", {"q": Noul(instructions="a?")})
        raise JevBudgetExceededError("capped")


@dataclass
class Stops:
    name: str = "select"

    async def run(self, ctx: Context) -> None:
        ctx.stop("select", "nothing to read")


async def test_each_document_is_recorded_in_the_store(store: SQLiteStore) -> None:
    ex = Extractor(
        [Car],
        jev=FakeJev().client(),
        pipeline=Pipeline([Finds()]),
        budgets=Budgets(per_document=DocBudget(max_llm_calls=1)),
        store=store,
        run_id="run-1",
    )
    result = await ex.extract(doc())
    [stat] = await store.documents()
    assert stat.run_id == "run-1"
    assert stat.url == "https://cars.test/golf"
    assert stat.schemas == ["Car"]
    assert stat.records == 1
    assert (stat.jev_requests, stat.jev_questions) == (1, 1)
    assert stat.jev_cost == result.meta.jev.cost
    assert stat.llm_calls == 1
    assert stat.seconds > 0
    assert stat.values == [
        ValueStat(field="Car.model", method="jev", value=("Golf " * 50)[:MAX_STAT_VALUE_CHARS]),
        ValueStat(field="Car.power_ps", method="llm", confidence=0.7, value="150"),
    ]
    assert stat.events == [
        DocumentEvent(kind="budget", message="document max_llm_calls: 1 LLM calls (the limit)")
    ]
    assert stat == document_stat(
        result, doc_id=stat.id, run_id="run-1", seconds=stat.seconds
    ).model_copy(update={"at": stat.at})


async def test_a_stopped_document_records_why(store: SQLiteStore) -> None:
    ex = Extractor([Car], jev=FakeJev().client(), pipeline=Pipeline([Stops()]), store=store)
    await ex.extract(doc())
    [stat] = await store.documents()
    assert stat.events == [
        DocumentEvent(kind="stopped", message="select: nothing to read", stage="select")
    ]
    assert stat.records == 0


async def test_a_failed_document_is_recorded_with_its_error(store: SQLiteStore) -> None:
    ex = Extractor([Car], jev=FakeJev().client(), pipeline=Pipeline([Fails()]), store=store)
    result = await ex.extract(doc())
    assert result.status == "failed"
    [stat] = await store.documents()
    error = PartError(
        stage="select", kind="jev", type="JevBackendError", message="backend down", fatal=True
    )
    assert (stat.status, stat.errors) == ("failed", [error])
    assert stat.events == [
        DocumentEvent(
            kind="error", message="select jev: JevBackendError: backend down", stage="select"
        )
    ]
    assert stat.jev_requests == 1
    assert stat.values == []


async def test_a_spend_cap_still_raises_and_is_recorded(store: SQLiteStore) -> None:
    ex = Extractor([Car], jev=FakeJev().client(), pipeline=Pipeline([OverCap()]), store=store)
    with pytest.raises(JevBudgetExceededError):
        await ex.extract(doc())
    [stat] = await store.documents()
    assert stat.status == "failed"
    assert stat.events == [
        DocumentEvent(kind="error", message="JevBudgetExceededError: capped", stage="select")
    ]
    assert stat.jev_requests == 1


async def test_record_stats_off_records_nothing(store: SQLiteStore) -> None:
    ex = Extractor(
        [Car], jev=FakeJev().client(), pipeline=Pipeline([Finds()]), store=store, record_stats=False
    )
    await ex.extract(doc())
    assert await store.documents() == []


async def test_without_a_store_nothing_is_opened_to_record_stats() -> None:
    ex = Extractor([Car], jev=FakeJev().client(), pipeline=Pipeline([Stops()]))
    await ex.extract(doc())
    assert await ex.store() is None


async def test_an_in_memory_store_the_extractor_opened_records_no_stats() -> None:
    ex = Extractor(
        [Car],
        jev=FakeJev().client(),
        pipeline=Pipeline([Stops()]),
        budgets=Budgets(run=RunBudget(max_jev_spend=1.0)),
    )
    await ex.extract(doc())
    store = await ex.store()
    assert store is not None
    assert await store.documents() == []
    await ex.aclose()


async def test_plain_dict_has_each_records_values_without_meta() -> None:
    ex = Extractor([Car], jev=FakeJev().client(), pipeline=Pipeline([Finds()]))
    result = await ex.extract(doc())
    assert result.to_plain_dict() == {
        "status": "ok",
        "errors": [],
        "records": [
            {
                "schema": "Car",
                "entity": "document",
                "record": {"model": "Golf " * 50, "power_ps": 150},
            }
        ],
    }
    assert (
        result.to_dict()["records"][0]["record"] == result.to_plain_dict()["records"][0]["record"]
    )


# --- locale --------------------------------------------------------------------------


@dataclass
class SeesLocale:
    """Records the locale each document runs under."""

    seen: list[str | None]
    name: str = "select"

    async def run(self, ctx: Context) -> None:
        self.seen.append(ctx.locale)


async def test_the_extractors_locale_is_the_default_for_documents_that_dont_say() -> None:
    stage = SeesLocale([])
    async with Extractor(
        [Car], jev=FakeJev().client(), pipeline=Pipeline([stage]), locale="en_gb"
    ) as extractor:
        assert extractor.locale == "en-GB"
        await extractor.extract(Document.from_bytes(b"%PDF-1.7"))
        await extractor.extract(Document.from_bytes(b'<html lang="de-de"><p>Golf</p></html>'))
        await extractor.extract(Document.from_bytes(b"%PDF-1.7", locale="fr_FR"))
    assert stage.seen == ["en-GB", "de-DE", "fr-FR"]


async def test_without_a_locale_documents_that_dont_say_have_none() -> None:
    stage = SeesLocale([])
    async with Extractor([Car], jev=FakeJev().client(), pipeline=Pipeline([stage])) as extractor:
        assert extractor.locale is None
        await extractor.extract(Document.from_bytes(b"%PDF-1.7"))
    assert stage.seen == [None]


@pytest.mark.parametrize("locale", ["English", "", "en GB", "en-GB\n"])
def test_the_extractors_locale_must_be_a_language_tag(locale: str) -> None:
    with pytest.raises(ValueError, match="locale must be a BCP 47 language tag"):
        Extractor([Car], jev=FakeJev().client(), locale=locale)


# --- status and errors (#230) ----------------------------------------------------------


@dataclass
class Raises:
    """Finds a value, then raises ``error``."""

    error: Exception
    name: str = "select"

    async def run(self, ctx: Context) -> None:
        ctx.schemas["Car"].set_field("document", "model", FieldMeta(value="Golf", method="jev"))
        raise self.error


@dataclass
class SkipsAPart:
    name: str = "candidates"

    async def run(self, ctx: Context) -> None:
        ctx.schemas["Car"].set_field("document", "model", FieldMeta(value="Golf", method="jev"))
        ctx.part_failed(self.name, "generator", "gen-1", ValueError("bad regex"))


async def test_a_bug_in_a_stage_fails_the_document_instead_of_raising() -> None:
    boom = RuntimeError("boom")
    ex = Extractor([Car], jev=FakeJev().client(), pipeline=Pipeline([Raises(boom)]))
    result = await ex.extract(doc())
    error = PartError(stage="select", kind="stage", type="RuntimeError", message="boom", fatal=True)
    assert (result.status, result.errors) == ("failed", [error])
    assert result.cause is boom
    assert result.one(Car).record.model == "Golf"  # what was found before it failed
    data = result.to_dict()
    assert (data["status"], data["errors"]) == ("failed", [error.model_dump(mode="json")])
    assert (data["meta"]["status"], data["meta"]["errors"]) == (data["status"], data["errors"])
    with pytest.raises(ExtractionError, match="select stage: RuntimeError: boom") as raised:
        result.raise_for_errors(partial=False)
    assert raised.value.__cause__ is boom
    assert raised.value.errors == [error]


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (JevBackendError("down"), "jev"),
        (StoreError("locked"), "store"),
        (PdfLayoutError("broken"), "document"),
        (UnreadableImageError("not an image"), "document"),
        (KeyError("x"), "stage"),
    ],
)
async def test_core_failures_say_what_failed(error: Exception, kind: str) -> None:
    ex = Extractor([Car], jev=FakeJev().client(), pipeline=Pipeline([Raises(error)]))
    result = await ex.extract(doc())
    [found] = result.errors
    assert (found.stage, found.kind, found.type, found.fatal) == (
        "select",
        kind,
        type(error).__name__,
        True,
    )


async def test_a_skipped_part_makes_the_result_partial() -> None:
    ex = Extractor([Car], jev=FakeJev().client(), pipeline=Pipeline([SkipsAPart()]))
    result = await ex.extract(doc())
    assert result.status == "partial"
    assert result.errors == [
        PartError(
            stage="candidates",
            kind="generator",
            part="gen-1",
            type="ValueError",
            message="bad regex",
        )
    ]
    assert result.cause is None
    result.raise_for_errors(partial=False)  # only a failed result raises then
    with pytest.raises(ExtractionError, match=r"^extraction partial for https://cars\.test/golf"):
        result.raise_for_errors()


async def test_an_ok_result_raises_nothing() -> None:
    ex = Extractor([Car], jev=FakeJev().client(), pipeline=Pipeline([Finds()]))
    result = await ex.extract(doc())
    assert (result.status, result.errors) == ("ok", [])
    result.raise_for_errors()


async def test_a_ledger_that_cant_record_makes_the_result_partial() -> None:
    class NoLedger(SQLiteStore):
        async def record_spend(self, entry: SpendEntry) -> None:
            raise StoreError("ledger is read-only")

    store = NoLedger(":memory:")
    ex = Extractor(
        [Car],
        jev=FakeJev().client(),
        pipeline=Pipeline([Finds()]),
        budgets=Budgets(run=RunBudget(max_jev_spend=1.0)),
        store=store,
    )
    result = await ex.extract(doc())
    await store.aclose()
    assert result.status == "partial"
    assert [(e.stage, e.kind, e.part, e.type) for e in result.errors] == [
        ("extract", "store", "ledger", "StoreError")
    ]


async def test_stats_that_cant_be_recorded_make_the_result_partial() -> None:
    class NoStats(SQLiteStore):
        async def record_document(self, stat: DocumentStat) -> None:
            raise StoreError("disk full")

    store = NoStats(":memory:")
    ex = Extractor([Car], jev=FakeJev().client(), pipeline=Pipeline([Finds()]), store=store)
    result = await ex.extract(doc())
    await store.aclose()
    assert [(e.kind, e.part, e.message) for e in result.errors] == [("store", "stats", "disk full")]
    assert result.one(Car).record.model == "Golf " * 50


class FlakyOnce:
    """A Jev backend whose first request fails transiently."""

    def __init__(self) -> None:
        self.inner = FakeJev()
        self.failed = False

    async def system_one(
        self, state: JSONContent, questions: Mapping[str, Question]
    ) -> JevResponse:
        if not self.failed:
            self.failed = True
            raise JevTransientError("503")
        return await self.inner.system_one(state, questions)


async def test_jev_retries_are_counted_in_the_result() -> None:
    ex = Extractor(
        [Car],
        jev=JevClient(FlakyOnce()),
        pipeline=Pipeline([Finds()]),
        jev_retry=RetryPolicy(backoff_initial=0),
    )
    result = await ex.extract(doc())
    assert result.status == "ok"
    assert (result.meta.jev.retries, result.meta.jev.requests) == (1, 2)
    assert result.to_dict()["meta"]["jev"]["retries"] == 1


async def test_jev_retry_policy_applies_to_each_document() -> None:
    ex = Extractor(
        [Car],
        jev=JevClient(FlakyOnce()),
        pipeline=Pipeline([Finds()]),
        jev_retry=RetryPolicy(max_retries=0),
    )
    result = await ex.extract(doc())
    assert result.status == "failed"
    assert [(e.kind, e.type) for e in result.errors] == [("jev", "JevTransientError")]
    assert result.meta.jev.retries == 0


async def test_a_failing_refresh_of_learned_generators_fails_the_document(
    monkeypatch: pytest.MonkeyPatch, store: SQLiteStore
) -> None:
    ex = Extractor([Car], jev=FakeJev().client(), pipeline=Pipeline([Finds()]), store=store)
    learned = await ex.learned_generators()
    assert learned is not None

    async def refresh() -> object:
        raise StoreError("generator gen-x doesn't validate")

    monkeypatch.setattr(learned, "refresh", refresh)
    result = await ex.extract(doc())
    assert [(e.stage, e.kind, e.fatal) for e in result.errors] == [("extract", "store", True)]
    assert result.meta.jev.requests == 0  # nothing ran


async def test_the_learner_retries_jev_as_documents_do() -> None:
    policy = RetryPolicy(max_retries=5)
    ex = Extractor(
        [Car], jev=FakeJev().client(), generator_llm=FakeLLM(lambda _p, _s: {}), jev_retry=policy
    )
    learner = await ex.learner()
    assert learner is not None
    assert learner.jev.retry == policy
    await ex.aclose()
