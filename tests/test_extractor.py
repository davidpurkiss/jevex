from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from jevex import Budgets, DocBudget, Document, Extractor, Field, Pipeline, RunBudget
from jevex.extractor import document_stat
from jevex.jev import JevBackendError, Noul
from jevex.pipeline import Context
from jevex.results import FieldMeta
from jevex.store import MAX_STAT_VALUE_CHARS, DocumentEvent, SQLiteStore, ValueStat
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
    assert stat.events == [DocumentEvent(kind="stopped", message="select: nothing to read")]
    assert stat.records == 0


async def test_a_failed_document_is_recorded_with_its_error(store: SQLiteStore) -> None:
    ex = Extractor([Car], jev=FakeJev().client(), pipeline=Pipeline([Fails()]), store=store)
    with pytest.raises(JevBackendError):
        await ex.extract(doc())
    [stat] = await store.documents()
    assert stat.events == [DocumentEvent(kind="error", message="JevBackendError: backend down")]
    assert stat.jev_requests == 1
    assert stat.values == []


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
        "records": [
            {
                "schema": "Car",
                "entity": "document",
                "record": {"model": "Golf " * 50, "power_ps": 150},
            }
        ]
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
