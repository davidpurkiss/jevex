import asyncio
import base64
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from jevex import Context, Document, Extractor, Field, Pipeline, UnreadablePdfError
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
from jevex.store import SQLiteStore, StoreError, VerifiedExample, open_store
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
    llm: FakeLLM | None = None
    learn: VerifiedExample | None = None

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
        if self.llm is not None and ctx.budget is not None:
            try:
                await ctx.budget.call_llm(self.llm, "p", Book)
            except Exception as exc:
                ctx.part_failed("fallback", "llm_extractor", "FakeLLM", exc)
        if self.learn is not None and ctx.learner is not None:
            await ctx.learner.submit(self.learn)
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
        "checks": {"store": "ok"},
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


async def test_extractors_are_made_once_per_set_of_schemas(
    stage: FindValues, fake_jev: FakeJev
) -> None:
    service = Service([Book, Author], jev=fake_jev.client())
    await service.start()
    both = service.extractor(["Author", "Book"])
    book = service.extractor(["Book"])
    assert service.extractor(["Book", "Author"]) is both
    assert book is not both
    # One store and run for all of them: the run budget's ledger and learned generators.
    assert both.run_id == book.run_id == service.run_id
    assert await both.store() is await book.store() is service.store is not None
    await service.aclose()


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
    assert 'jevex_errors_total{stage="candidates",kind="generator",part="gen-1"} 1\n' in metrics


def test_a_failed_result_is_counted_with_its_spend(client: TestClient, stage: FindValues) -> None:
    stage.fail = KeyError("bug")
    assert client.post("/extract", json=body()).status_code == 500
    metrics = client.get("/metrics").text
    assert 'jevex_documents_total{outcome="error"} 1\n' in metrics
    assert 'jevex_errors_total{stage="select",kind="stage",part=""} 1\n' in metrics
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


# --- monitoring ----------------------------------------------------------------------


class RateLimited(Exception):
    """An SDK's 429, as Anthropic's and OpenAI's errors carry it."""

    status_code = 429


def raises(exc: Exception) -> FakeLLM:
    def answer(_prompt: str, _schema: type[BaseModel]) -> object:
        raise exc

    return FakeLLM(answer)


class FlakyOnce429(FlakyOnce):
    async def system_one(
        self, state: JSONContent, questions: Mapping[str, Question]
    ) -> JevResponse:
        if not self.failed:
            self.failed = True
            raise JevTransientError("429 slow down", status=429)
        return await self.inner.system_one(state, questions)


def test_metrics_attribute_errors_and_time_stages_and_count_rate_limits(
    stage: FindValues, fake_jev: FakeJev
) -> None:
    stage.skip = RuntimeError("bad regex")
    stage.llm = raises(RateLimited("too many requests"))
    jev = JevClient(FlakyOnce429(fake_jev), retry=RetryPolicy(backoff_initial=0))
    with TestClient(create_app(Service([Book], jev=jev))) as client:
        assert client.post("/extract", json=body()).json()["status"] == "partial"
        text = client.get("/metrics").text
    for line in (
        'jevex_errors_total{stage="candidates",kind="generator",part="gen-1"} 1',
        'jevex_errors_total{stage="fallback",kind="llm_extractor",part="FakeLLM"} 1',
        "jevex_jev_rate_limited_total 1",
        "jevex_jev_retries_total 1",
        "jevex_llm_rate_limited_total 1",
        "# TYPE jevex_stage_seconds summary",
        'jevex_stage_seconds_count{stage="select"} 1',
        "jevex_store_errors_total 0",
    ):
        assert line + "\n" in text
    assert 'jevex_stage_seconds_sum{stage="select"} ' in text


@dataclass
class FailsTwice:
    """One part raising two exception types in a document."""

    name: str = "candidates"

    async def run(self, ctx: Context) -> None:
        ctx.part_failed(self.name, "generator", "gen-1", KeyError("k"))
        ctx.part_failed(self.name, "generator", "gen-1", IndexError("i"))


async def test_errors_count_documents_not_exception_types() -> None:
    ex = Extractor([Book], jev=FakeJev().client(), pipeline=Pipeline([FailsTwice()]))
    result = await ex.extract(Document.from_bytes(HTML))
    assert len(result.errors) == 2
    metrics = Metrics()
    metrics.add(result, 0.1)
    assert metrics.errors == {("candidates", "generator", "gen-1"): 1}


async def test_a_hung_store_leaves_metrics_without_headroom(
    stage: FindValues, fake_jev: FakeJev, monkeypatch: pytest.MonkeyPatch
) -> None:
    import jevex.server as server
    from jevex import Budgets, RunBudget

    class Hangs(SQLiteStore):
        async def spend(self, **_: Any) -> float:
            await asyncio.sleep(10)
            return 0.0

    monkeypatch.setattr(server, "STORE_TIMEOUT_S", 0.01)
    store = Hangs(":memory:")
    service = Service(
        [Book], jev=fake_jev.client(), store=store, budgets=Budgets(run=RunBudget(max_spend=1))
    )
    await service.start()
    text = await service.render_metrics()
    assert "jevex_budget_remaining_usd" not in text
    assert "jevex_store_errors_total 1\n" in text
    await service.aclose()
    await store.aclose()


def test_metrics_report_drift_per_field(client: TestClient) -> None:
    client.post("/extract", json=body(["Book", "Author"]))
    client.post("/extract", json=body("Book"))
    text = client.get("/metrics").text
    for line in (
        "jevex_drift_documents 2",
        'jevex_field_records{field="Book.title"} 2',
        'jevex_field_records{field="Author.name"} 1',
        'jevex_field_none_rate{field="Book.title"} 0.0',
        'jevex_field_fallback_rate{field="Book.title"} 0.0',
        'jevex_field_confidence_mean{field="Book.title"} 0.9',
    ):
        assert line + "\n" in text
    # No learner and no spend caps: their series are left out.
    assert "jevex_learner_alive" not in text
    assert "jevex_budget_remaining_usd" not in text


async def test_metrics_report_budget_headroom(
    stage: FindValues, fake_jev: FakeJev, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jevex import Budgets, RunBudget
    from jevex.llm import reset_process_llm_cost
    from jevex.store import SpendEntry

    reset_process_llm_cost()  # other tests' calls
    monkeypatch.setenv("JEVEX_LLM_MAX_COST_USD", "2")
    store = open_store(":memory:")
    budgets = Budgets(run=RunBudget(max_spend=5.0, period="week", max_jev_spend=1.0))
    service = Service([Book], jev=fake_jev.client(), store=store, budgets=budgets)
    await service.start()
    await store.record_spend(SpendEntry(amount_usd=1.25, kind="llm", run_id="other"))
    text = await service.render_metrics()
    for line in (
        'jevex_budget_remaining_usd{scope="run",kind="llm",period="week"} 3.75',
        'jevex_budget_remaining_usd{scope="run",kind="jev",period="week"} 1.0',
        'jevex_budget_remaining_usd{scope="process",kind="llm",period="process"} 2.0',
        'jevex_budget_limit_usd{scope="run",kind="llm",period="week"} 5.0',
    ):
        assert line + "\n" in text
    await service.aclose()
    await store.aclose()


class BrokenStore(SQLiteStore):
    """A store that stopped answering reads."""

    async def spend(self, **_: Any) -> float:
        raise StoreError("database is locked")


def test_health_is_503_when_the_store_does_not_answer(stage: FindValues, fake_jev: FakeJev) -> None:
    from jevex import Budgets, RunBudget

    store = BrokenStore(":memory:")
    service = Service(
        [Book], jev=fake_jev.client(), store=store, budgets=Budgets(run=RunBudget(max_spend=1))
    )
    with TestClient(create_app(service)) as client:
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["status"] == "unhealthy"
        assert response.json()["checks"] == {"store": "StoreError: database is locked"}
        text = client.get("/metrics").text  # the headroom read fails too
    assert "jevex_store_errors_total 2\n" in text
    assert "jevex_budget_remaining_usd" not in text


PRICED = "Price: 9.1"


class Priced(BaseModel):
    """A priced item."""

    price: float = Field(description="Price in GBP")


def test_a_dead_learner_flips_health_and_its_gauge(stage: FindValues, fake_jev: FakeJev) -> None:
    stage.learn = VerifiedExample(
        id="ex-1",
        field="Priced.price",
        statement=PRICED,
        value=9.1,
        evidence=(7, 10),
        context={"heading_trail": [], "kind": "sentence"},
        source="llm",
        probability=0.99,
    )
    service = Service([Priced], jev=fake_jev.client(), generator_llm=raises(RuntimeError("bug")))
    client = TestClient(create_app(service))
    client.__enter__()
    assert client.get("/health").json()["checks"] == {"store": "ok"}  # no learner yet
    client.post("/extract", json=body("Priced"))
    response = client.get("/health")
    for _ in range(50):  # the worker learns in the background
        if response.status_code == 503:
            break
        response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["checks"] == {"store": "ok", "learner": "1 worker(s) died"}
    text = client.get("/metrics").text
    assert "jevex_learner_alive 0\n" in text
    assert "jevex_learner_worker_deaths_total 1\n" in text
    assert 'jevex_learner_outcomes_total{status="accepted"} 0\n' in text
    # Closing the service at shutdown raises the worker's error.
    with pytest.raises(RuntimeError, match="learner worker failed"):
        client.__exit__(None, None, None)


def test_learner_outcomes_are_counted(stage: FindValues, fake_jev: FakeJev) -> None:
    from jevex.llm import LLMError

    stage.learn = VerifiedExample(
        id="ex-1",
        field="Priced.price",
        statement=PRICED,
        value=9.1,
        evidence=(7, 10),
        context={"heading_trail": [], "kind": "sentence"},
        source="llm",
        probability=0.99,
    )
    service = Service([Priced], jev=fake_jev.client(), generator_llm=raises(LLMError("down")))
    with TestClient(create_app(service)) as client:
        client.post("/extract", json=body("Priced"))
        text = client.get("/metrics").text
        for _ in range(50):
            if 'jevex_learner_outcomes_total{status="llm_error"} 1\n' in text:
                break
            text = client.get("/metrics").text
        assert 'jevex_learner_outcomes_total{status="llm_error"} 1\n' in text
        assert "jevex_learner_alive 1\n" in text
        assert client.get("/health").status_code == 200
