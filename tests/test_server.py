import base64
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from jevex import Budgets, Context, ExtractionResult, Field, RunBudget, UnreadablePdfError
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
from jevex.llm import LLMBudgetExceededError
from jevex.results import FieldMeta
from jevex.server import (
    DocumentIn,
    ExtractRequest,
    Metrics,
    Service,
    UnknownSchemaError,
    create_app,
)
from jevex.store import (
    LedgerError,
    MemoryLedger,
    SpendEntry,
    SpendLedger,
    SQLiteStore,
    Store,
    open_store,
)
from jevex.testing import FakeJev, FakeLLM

HTML = b"<html><body><h1>Dune</h1><p>Title: Dune</p></body></html>"


class Book(BaseModel):
    """A book for sale."""

    title: str = Field(description="Book title")


class Author(BaseModel):
    """A book's author."""

    name: str = Field(description="Author name")


@dataclass
class FindValues:
    """Stand-in for the real stages: asks Jev one question, then records a value for each
    schema, so the service has records, Jev usage and a resolution mix to report."""

    name: str = "select"
    seen: list[tuple[str, list[str]]] = field(default_factory=list[tuple[str, list[str]]])
    fail: Exception | None = None
    skip: Exception | None = None
    stop: bool = False

    async def run(self, ctx: Context) -> None:
        self.seen.append((ctx.document.content_type, [r.spec.name for r in ctx.active]))
        if self.fail is not None:
            raise self.fail
        await ctx.jev.ask("s", {"q": Noul(instructions="Is this a book page?")})
        for run in ctx.active:
            name = run.spec.fields[0].name
            run.set_field("document", name, FieldMeta(value="Dune", confidence=0.9, method="jev"))
        if self.skip is not None:
            ctx.part_failed("candidates", "generator", "gen-1", self.skip)
        if self.stop and ctx.budget is not None:
            ctx.budget.record_hit("run", "max_spend", "the day's LLM budget is spent")
            ctx.stop(self.name, "budget")


@pytest.fixture
def stage(monkeypatch: pytest.MonkeyPatch) -> FindValues:
    import jevex.extractor as extractor

    found = FindValues()
    monkeypatch.setattr(extractor, "DEFAULT_STAGES", (found,))
    return found


@pytest.fixture
def fake_jev() -> FakeJev:
    return FakeJev().noul("Is this a book page?", p=0.9)


def body(schema: str | list[str] = "Book", **document: Any) -> dict[str, Any]:
    return {
        "document": {"content": base64.b64encode(HTML).decode(), **document},
        "schema": schema,
    }


@pytest.fixture
def client(stage: FindValues, fake_jev: FakeJev) -> Iterator[TestClient]:
    service = Service([Book, Author], jev=fake_jev.client())
    with TestClient(create_app(service)) as c:
        yield c


def test_health_names_the_schemas(client: TestClient) -> None:
    from jevex import __version__

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": __version__,
        "schemas": ["Book", "Author"],
    }


def test_extract_returns_records_like_jevex_extract(client: TestClient) -> None:
    response = client.post("/extract", json=body(url="https://shop.example.com/dune"))
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "errors": [],
        "records": [{"schema": "Book", "entity": "document", "record": {"title": "Dune"}}],
    }


def test_extract_with_meta_returns_the_whole_result(client: TestClient) -> None:
    response = client.post("/extract", json={**body(), "meta": True})
    data = response.json()
    assert data["records"][0]["meta"]["title"]["method"] == "jev"
    assert data["meta"]["url"] is None
    assert data["meta"]["jev"]["requests"] == 1


def test_extract_asks_only_about_the_schemas_named(client: TestClient, stage: FindValues) -> None:
    client.post("/extract", json=body("Author"))
    client.post("/extract", json=body(["Author", "Book", "Author"]))
    response = client.post("/extract", json=body(["Book", "Author"]))
    assert [r["schema"] for r in response.json()["records"]] == ["Book", "Author"]
    # Registration order, whatever order the request names them in.
    assert [names for _, names in stage.seen] == [
        ["Author"],
        ["Book", "Author"],
        ["Book", "Author"],
    ]


async def test_the_services_locale_is_its_extractors(fake_jev: FakeJev) -> None:
    service = Service([Book, Author], jev=fake_jev.client(), locale="pt_br")
    assert service.locale == "pt-BR"
    await service.start()
    assert service.extractor(["Book"]).locale == "pt-BR"
    assert service.extractor(["Book", "Author"]).locale == "pt-BR"
    await service.aclose()
    assert Service([Book]).locale is None


def test_the_services_locale_is_checked_before_any_request() -> None:
    with pytest.raises(ValueError, match="locale must be a BCP 47 language tag"):
        Service([Book], locale="Portuguese")


async def test_extractors_are_made_once_per_set_of_schemas(
    stage: FindValues, fake_jev: FakeJev
) -> None:
    service = Service([Book, Author], jev=fake_jev.client())
    await service.start()
    both = service.extractor(["Author", "Book"])
    book = service.extractor(["Book"])
    assert service.extractor(["Book", "Author"]) is both
    assert book is not both
    # One store, ledger and run for all of them: the run budget and learned generators.
    assert both.run_id == book.run_id == service.run_id
    assert await both.store() is await book.store() is service.store is not None
    assert (await both.ledger()).ledger is (await book.ledger()).ledger is service.store
    assert service.ledger is service.store
    await service.aclose()


class NotALedger:
    """A store that isn't a spend ledger: an SQLite store without the ledger's methods."""

    LEDGER = frozenset({"record_spend", "spend", "try_spend", "spend_entries"})

    def __init__(self) -> None:
        self.inner = open_store(":memory:")

    def __getattr__(self, name: str) -> Any:
        if name in self.LEDGER:
            raise AttributeError(name)
        return getattr(self.inner, name)


async def extract_both(service: Service) -> tuple[ExtractionResult, ExtractionResult]:
    """A Book document, then an Author one: two schema sets, two extractors."""
    await service.start()
    book = await service.extract(DocumentIn(content=HTML).to_document(), ["Book"])
    author = await service.extract(DocumentIn(content=HTML).to_document(), ["Author"])
    return book, author


ONE_DOCUMENT = Budgets(run=RunBudget(max_jev_spend=1e-9))  # the first document spends it


async def test_a_ledger_given_keeps_the_run_budget_for_every_schema_set(
    stage: FindValues, fake_jev: FakeJev
) -> None:
    mine = MemoryLedger()
    store = SQLiteStore(":memory:")
    service = Service(
        [Book, Author], jev=fake_jev.client(), store=store, ledger=mine, budgets=ONE_DOCUMENT
    )
    book, author = await extract_both(service)
    assert service.ledger is mine
    for names in (["Book"], ["Author"]):
        assert (await service.extractor(names).ledger()).ledger is mine
    assert not book.meta.stopped
    assert author.meta.stopped
    assert [e.limit for e in author.meta.budget_events] == ["max_jev_spend"]
    assert author.meta.jev.requests == 0
    assert await mine.spend(kind="jev") == pytest.approx(book.meta.jev.cost)
    assert await store.spend() == 0.0
    await service.aclose()
    await store.aclose()


async def test_a_store_that_isnt_a_ledger_gets_one_in_memory_ledger_for_the_service(
    stage: FindValues, fake_jev: FakeJev
) -> None:
    store = NotALedger()
    assert not isinstance(store, SpendLedger)
    service = Service(
        [Book, Author], jev=fake_jev.client(), store=cast("Store", store), budgets=ONE_DOCUMENT
    )
    book, author = await extract_both(service)
    shared = service.ledger
    assert isinstance(shared, MemoryLedger)
    for names in (["Book"], ["Author"]):
        assert (await service.extractor(names).ledger()).ledger is shared
    assert not book.meta.stopped
    assert [e.limit for e in author.meta.budget_events] == ["max_jev_spend"]
    assert await shared.spend(kind="jev") == pytest.approx(book.meta.jev.cost)
    await service.aclose()
    await store.inner.aclose()


async def test_the_ledger_is_known_only_once_the_store_is_open(fake_jev: FakeJev) -> None:
    service = Service([Book], jev=fake_jev.client(), store=":memory:")
    assert service.ledger is None
    await service.start()
    assert service.ledger is service.store
    await service.aclose()
    mine = MemoryLedger()
    assert Service([Book], ledger=mine).ledger is mine


async def test_a_failing_ledger_is_reported_on_the_result(
    stage: FindValues, fake_jev: FakeJev
) -> None:
    class DownLedger(MemoryLedger):
        async def spend(self, **_: Any) -> float:
            raise ConnectionError("ledger is down")

    service = Service([Book], jev=fake_jev.client(), ledger=DownLedger(), budgets=ONE_DOCUMENT)
    await service.start()
    result = await service.extract(DocumentIn(content=HTML).to_document(), ["Book"])
    assert result.status == "partial"
    assert [(e.kind, e.part, e.type) for e in result.errors] == [
        ("ledger", "DownLedger", LedgerError.__name__)
    ]
    assert service.metrics.documents["partial"] == 1
    await service.aclose()


async def test_stats_read_spend_from_the_ledger_given(stage: FindValues, fake_jev: FakeJev) -> None:
    mine = MemoryLedger()
    store = open_store(":memory:")
    service = Service(
        [Book],
        jev=fake_jev.client(),
        store=store,
        ledger=mine,
        budgets=Budgets(run=RunBudget(max_jev_spend=1.0)),
        stats=True,
    )
    await service.start()
    result = await service.extract(DocumentIn(content=HTML).to_document(), ["Book"])
    # What the learner spent after the document: only the ledger has it.
    await mine.record_spend(SpendEntry(amount_usd=0.5, kind="llm", run_id=service.run_id))
    stats = await service.read_stats()
    assert stats.spend[-1].total == pytest.approx(result.meta.jev.cost + 0.5)
    await service.aclose()
    await store.aclose()


def test_content_type_is_sniffed_or_normalised(client: TestClient, stage: FindValues) -> None:
    client.post("/extract", json=body())
    client.post("/extract", json=body(content_type="text/html; charset=utf-8"))
    assert [t for t, _ in stage.seen] == ["text/html", "text/html"]


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        (body("Nope"), "unknown schema 'Nope'; this service has Book, Author"),
        (body(["Book", "Nope", "Gone"]), "unknown schema 'Nope', 'Gone'; this service has"),
    ],
)
def test_unknown_schemas_are_422(client: TestClient, payload: dict[str, Any], detail: str) -> None:
    response = client.post("/extract", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"].startswith(detail)


@pytest.mark.parametrize(
    "payload",
    [
        body([]),
        body(""),
        {"document": {"content": "not base64!"}, "schema": "Book"},
        {"document": {"content": "PGh0bWw+"}},
        {"schema": "Book"},
        body(locale="German"),
    ],
)
def test_bad_requests_are_422(client: TestClient, payload: dict[str, Any]) -> None:
    assert client.post("/extract", json=payload).status_code == 422


@pytest.mark.parametrize(
    ("error", "status", "detail"),
    [
        (JevBackendError("upstream down"), 502, "Jev: upstream down"),
        (JevBudgetExceededError("Jev spend cap reached"), 503, "Jev spend cap reached"),
        (LLMBudgetExceededError("LLM spend cap reached"), 503, "LLM spend cap reached"),
        (
            UnreadablePdfError("pdfium couldn't open the PDF"),
            422,
            "can't read the document: pdfium couldn't open the PDF",
        ),
    ],
)
def test_extraction_errors_map_to_statuses(
    client: TestClient, stage: FindValues, error: Exception, status: int, detail: str
) -> None:
    stage.fail = error
    response = client.post("/extract", json=body())
    assert (response.status_code, response.json()) == (status, {"detail": detail})
    assert 'jevex_documents_total{outcome="error"} 1\n' in client.get("/metrics").text


def test_a_partial_result_is_a_200_saying_what_was_skipped(
    client: TestClient, stage: FindValues
) -> None:
    stage.skip = RuntimeError("bad regex")
    response = client.post("/extract", json=body())
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "partial"
    assert data["errors"] == [
        {
            "stage": "candidates",
            "kind": "generator",
            "part": "gen-1",
            "type": "RuntimeError",
            "message": "bad regex",
            "count": 1,
            "fatal": False,
        }
    ]
    assert data["records"][0]["record"] == {"title": "Dune"}
    metrics = client.get("/metrics").text
    assert 'jevex_documents_total{outcome="partial"} 1\n' in metrics
    assert 'jevex_documents_total{outcome="ok"} 0\n' in metrics
    assert 'jevex_errors_total{stage="candidates",kind="generator"} 1\n' in metrics


def test_a_failed_result_is_counted_with_its_spend(client: TestClient, stage: FindValues) -> None:
    stage.fail = KeyError("bug")
    assert client.post("/extract", json=body()).status_code == 500
    metrics = client.get("/metrics").text
    assert 'jevex_documents_total{outcome="error"} 1\n' in metrics
    assert 'jevex_errors_total{stage="select",kind="stage"} 1\n' in metrics
    assert "jevex_extract_seconds_count 1\n" in metrics


class FlakyOnce:
    """A Jev backend whose first request fails transiently."""

    def __init__(self, inner: FakeJev) -> None:
        self.inner = inner
        self.failed = False

    async def system_one(
        self, state: JSONContent, questions: Mapping[str, Question]
    ) -> JevResponse:
        if not self.failed:
            self.failed = True
            raise JevTransientError("503")
        return await self.inner.system_one(state, questions)


@pytest.mark.usefixtures("stage")
def test_retries_are_counted(fake_jev: FakeJev) -> None:
    jev = JevClient(FlakyOnce(fake_jev), retry=RetryPolicy(backoff_initial=0))
    with TestClient(create_app(Service([Book], jev=jev))) as client:
        assert client.post("/extract", json=body()).json()["status"] == "ok"
        metrics = client.get("/metrics").text
    assert "jevex_jev_retries_total 1\n" in metrics
    assert "jevex_llm_retries_total 0\n" in metrics


def test_unexpected_errors_are_500_and_counted(stage: FindValues, fake_jev: FakeJev) -> None:
    stage.fail = RuntimeError("boom")
    service = Service([Book], jev=fake_jev.client())
    with TestClient(create_app(service), raise_server_exceptions=False) as client:
        assert client.post("/extract", json=body()).status_code == 500
        metrics = client.get("/metrics").text
    assert 'jevex_documents_total{outcome="error"} 1\n' in metrics
    assert "jevex_documents_in_progress 0\n" in metrics


def test_metrics_count_documents_values_and_spend(client: TestClient) -> None:
    client.post("/extract", json=body(["Book", "Author"]))
    client.post("/extract", json=body("Book"))
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"
    text = response.text
    for line in (
        "# TYPE jevex_documents_total counter",
        'jevex_documents_total{outcome="ok"} 2',
        'jevex_documents_total{outcome="stopped"} 0',
        'jevex_records_total{schema="Author"} 1',
        'jevex_records_total{schema="Book"} 2',
        'jevex_values_total{schema="Author",method="jev"} 1',
        'jevex_values_total{schema="Book",method="jev"} 2',
        "jevex_jev_requests_total 2",
        "jevex_jev_questions_total 2",
        "jevex_llm_calls_total 0",
        "jevex_llm_cost_usd_total 0.0",
        "# TYPE jevex_extract_seconds summary",
        "jevex_extract_seconds_count 2",
    ):
        assert line + "\n" in text
    assert text.endswith("\n")


def test_metrics_count_stopped_documents_and_budget_hits(
    client: TestClient, stage: FindValues
) -> None:
    stage.stop = True
    assert client.post("/extract", json=body()).status_code == 200
    text = client.get("/metrics").text
    assert 'jevex_documents_total{outcome="ok"} 0\n' in text
    assert 'jevex_documents_total{outcome="stopped"} 1\n' in text
    assert 'jevex_budget_events_total{scope="run",limit="max_spend"} 1\n' in text


def test_metrics_escape_label_values() -> None:
    metrics = Metrics()
    metrics.records['We"ird\\Na\nme'] += 1
    assert 'jevex_records_total{schema="We\\"ird\\\\Na\\nme"} 1\n' in metrics.render()


def test_stats_are_off_by_default(client: TestClient) -> None:
    assert client.get("/stats/").status_code == 404
    assert client.get("/stats/api/summary").status_code == 404


def test_stats_need_a_store() -> None:
    with pytest.raises(ValueError, match="the stats UI reads the store"):
        Service([Book], stats=True)


def test_stats_serve_the_store(stage: FindValues, fake_jev: FakeJev, tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'jevex.db'}"
    service = Service([Book], jev=fake_jev.client(), store=url, stats=True, stats_budget_usd=5)
    with TestClient(create_app(service)) as client:
        client.post("/extract", json=body())
        summary = client.get("/stats/api/summary")
        assert summary.headers["cache-control"] == "no-store"
        assert summary.json()["documents"] == 1
        assert summary.json()["budget_usd"] == 5
        assert summary.json()["source"] == url
        page = client.get("/stats/")
        assert page.status_code == 200
        assert page.headers["content-type"].startswith("text/html")
        assert "jevex · stats" in page.text
        assert client.get("/stats", follow_redirects=False).headers["location"] == "/stats/"
        chart = client.get("/stats/api/chart/learning.svg?x=docs&animate=1")
        assert chart.headers["content-type"] == "image/svg+xml; charset=utf-8"
        assert chart.text.startswith("<svg")
        assert client.get("/stats/api/nope").status_code == 404
        assert client.get("/stats/api/chart/nope.svg").status_code == 404
        assert client.get("/stats/api/chart/learning").status_code == 404
        assert client.get("/stats/api/chart/learning.svg?x=pages").status_code == 400


def test_a_store_url_is_opened_and_closed(
    stage: FindValues, fake_jev: FakeJev, tmp_path: Path
) -> None:
    service = Service([Book], jev=fake_jev.client(), store=tmp_path / "jevex.db")
    with TestClient(create_app(service)) as client:
        client.post("/extract", json=body())
        assert service.store is not None
    assert service.store is None


async def test_a_given_store_records_stats_and_stays_open(
    stage: FindValues, fake_jev: FakeJev
) -> None:
    store = open_store(":memory:")
    service = Service([Book], jev=fake_jev.client(), store=store)
    await service.start()
    await service.extract(DocumentIn(content=HTML).to_document(), ["Book"])
    await service.aclose()
    assert len(await store.documents()) == 1  # still open: the caller closes it
    await store.aclose()


async def test_without_a_store_one_in_memory_store_is_shared_and_no_stats_kept(
    stage: FindValues, fake_jev: FakeJev
) -> None:
    service = Service([Book, Author], jev=fake_jev.client())
    await service.start()
    store = service.store
    assert store is not None
    await service.extract(DocumentIn(content=HTML).to_document(), ["Book"])
    await service.extract(DocumentIn(content=HTML).to_document(), ["Author"])
    assert await store.documents() == []
    await service.aclose()
    assert service.store is None


async def test_extract_before_start_raises(fake_jev: FakeJev) -> None:
    service = Service([Book], jev=fake_jev.client())
    with pytest.raises(RuntimeError, match="isn't started"):
        await service.extract(DocumentIn(content=HTML).to_document(), ["Book"])
    with pytest.raises(UnknownSchemaError):
        service.extractor(["Nope"])


class ClosingLLM(FakeLLM):
    def __init__(self) -> None:
        super().__init__([])
        self.closed = 0

    async def aclose(self) -> None:
        self.closed += 1


@pytest.mark.parametrize("close_llms", [True, False])
async def test_llms_are_closed_only_when_asked(fake_jev: FakeJev, close_llms: bool) -> None:
    llm = ClosingLLM()
    service = Service(
        [Book],
        jev=fake_jev.client(),
        extraction_llm=llm,
        generator_llm=llm,
        close_llms=close_llms,
    )
    await service.start()
    await service.aclose()
    assert llm.closed == (1 if close_llms else 0)


def test_schemas_must_be_given_and_unique() -> None:
    with pytest.raises(ValueError, match="at least one schema"):
        Service([])
    with pytest.raises(ValueError, match="'Book' is registered twice"):
        Service([Book, Book])


def test_request_model_takes_one_name_or_several() -> None:
    one = ExtractRequest.model_validate({"document": {"content": "PGh0bWw+"}, "schema": "Book"})
    several = ExtractRequest.model_validate(
        {"document": {"content": "PGh0bWw+"}, "schema": ["Author", "Book", "Author"]}
    )
    assert one.schema_names() == ["Book"]
    assert several.schema_names() == ["Author", "Book"]
    assert one.document.to_document().content == b"<html>"


def test_a_documents_language_reaches_the_extractor() -> None:
    request = ExtractRequest.model_validate(
        body(content_language="de-DE, en", locale="de-AT", content_type="text/html")
    )
    document = request.document.to_document()
    assert (document.content_language, document.locale) == ("de-DE, en", "de-AT")
    assert DocumentIn(content=HTML).to_document().locale is None


async def test_stats_dont_read_spend_from_the_in_memory_ledger(
    stage: FindValues, fake_jev: FakeJev
) -> None:
    store = NotALedger()
    service = Service(
        [Book],
        jev=fake_jev.client(),
        store=cast("Store", store),
        budgets=Budgets(run=RunBudget(max_jev_spend=1.0)),
        stats=True,
    )
    await service.start()
    result = await service.extract(DocumentIn(content=HTML).to_document(), ["Book"])
    shared = service.ledger
    assert isinstance(shared, MemoryLedger)
    # Only this process's spend is in it; the store's documents may be every process's.
    await shared.record_spend(SpendEntry(amount_usd=0.5, kind="llm", run_id=service.run_id))
    stats = await service.read_stats()
    assert stats.spend[-1].total == pytest.approx(result.meta.jev.cost)
    await service.aclose()
    await store.inner.aclose()
