import json
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import scrapy
from pydantic import BaseModel
from scrapy.crawler import Crawler
from scrapy.http import HtmlResponse, Response
from scrapy.utils.test import get_crawler

import jevex.contrib.scrapy as contrib
from jevex import Context, Document, Extractor, Field
from jevex.contrib.scrapy import AsyncioRequiredError, JevexPipeline, document_from_response
from jevex.errors import ExtractionError
from jevex.extractor import ExtractionResult
from jevex.jev import JevBudgetExceededError, JevClient, Noul
from jevex.pipeline import Pipeline
from jevex.results import FieldMeta
from jevex.store import SQLiteStore, VerifiedExample
from jevex.testing import FakeJev, FakeLLM

HTML = b"<html><body><h1>Dune</h1><p>Title: Dune</p></body></html>"
PDF = b"%PDF-1.4\n%fake\n"


class Book(BaseModel):
    """A book for sale."""

    title: str = Field(description="Book title")


class Author(BaseModel):
    """A book's author."""

    name: str = Field(description="Author name")


@dataclass
class FindTitle:
    """Stand-in for the real stages: asks Jev one question and records a title."""

    name: str = "select"
    fail: Exception | None = None
    skip: Exception | None = None
    seen: list[str | None] = field(default_factory=list[str | None])

    async def run(self, ctx: Context) -> None:
        self.seen.append(ctx.document.url)
        if self.fail is not None:
            raise self.fail
        await ctx.jev.ask("s", {"q": Noul(instructions="Is this a book page?")})
        for run in ctx.active:
            name = run.spec.fields[0].name
            run.set_field("document", name, FieldMeta(value="Dune", confidence=0.9, method="jev"))
        if self.skip is not None:
            ctx.part_failed("candidates", "generator", "gen-1", self.skip)


class FakePipeline(JevexPipeline):
    """The pipeline with a fake Jev and :class:`FindTitle` in place of the real stages."""

    stage: FindTitle
    closed: bool

    def make_extractor(self) -> Extractor:
        self.stage = FindTitle()
        self.closed = False
        extractor = Extractor(
            [Book],
            jev=JevClient(FakeJev().noul("Is this a book page?", p=0.9)),
            pipeline=Pipeline([self.stage]),
        )
        aclose = extractor.aclose

        async def close() -> None:
            self.closed = True
            await aclose()

        extractor.aclose = close
        return extractor


def crawler(**settings: Any) -> Crawler:
    return get_crawler(scrapy.Spider, settings)


async def opened[P: JevexPipeline](cls: type[P], **settings: Any) -> P:
    pipeline = cls.from_crawler(crawler(**settings))
    await pipeline.open_spider()
    return pipeline


def document(url: str = "https://books.example/dune") -> Document:
    return Document.from_bytes(HTML, url=url)


# --- document_from_response ----------------------------------------------------------


def test_document_from_response_takes_the_body_url_and_media_type() -> None:
    response = HtmlResponse(
        url="https://books.example/dune",
        body=HTML,
        headers={"Content-Type": "Text/HTML; charset=utf-8"},
    )
    before = datetime.now(UTC)

    doc = document_from_response(response, site="books")

    assert doc.content == HTML
    assert doc.url == "https://books.example/dune"
    assert doc.content_type == "text/html"
    assert doc.site == "books"
    assert doc.fetched_at is not None
    assert before <= doc.fetched_at <= datetime.now(UTC)


def test_document_from_response_trusts_a_specific_header_over_the_bytes() -> None:
    response = Response(
        url="https://books.example/notes.txt",
        body=HTML,
        headers={"Content-Type": "text/plain"},
    )

    assert document_from_response(response).content_type == "text/plain"


@pytest.mark.parametrize("header", [None, "application/octet-stream", "  "])
def test_document_from_response_sniffs_a_missing_or_generic_media_type(
    header: str | None,
) -> None:
    headers = {"Content-Type": header} if header is not None else {}
    response = Response(url="https://books.example/download?id=1", body=PDF, headers=headers)

    assert document_from_response(response).content_type == "application/pdf"


def test_document_from_response_falls_back_to_the_url_extension() -> None:
    response = Response(url="file:///tmp/spec.pdf", body=b"not magic")

    assert document_from_response(response).content_type == "application/pdf"


def test_document_from_response_takes_the_content_language() -> None:
    response = HtmlResponse(
        url="https://books.example/dune",
        body=HTML,
        headers={"Content-Type": "text/html", "Content-Language": "de-DE"},
    )
    assert document_from_response(response).content_language == "de-DE"
    plain = HtmlResponse(url="https://books.example/dune", body=HTML)
    assert document_from_response(plain).content_language is None


def test_document_from_response_keeps_octet_stream_when_nothing_says_more() -> None:
    response = Response(
        url="https://books.example/blob",
        body=b"\x00\x01",
        headers={"Content-Type": "application/octet-stream"},
    )
    assert document_from_response(response).content_type == "application/octet-stream"
    unknown = Response(url="https://books.example/blob", body=b"\x00")
    assert document_from_response(unknown).content_type == "application/octet-stream"


# --- JevexPipeline -------------------------------------------------------------------


async def test_pipeline_replaces_the_document_with_the_records() -> None:
    pipeline = await opened(FakePipeline)

    item = await pipeline.process_item({"url": "u", "document": document()})

    assert item == {
        "url": "u",
        "records": [{"schema": "Book", "entity": "document", "record": {"title": "Dune"}}],
    }
    assert pipeline.stage.seen == ["https://books.example/dune"]


async def test_pipeline_counts_documents_and_records_in_scrapy_stats() -> None:
    pipeline = await opened(FakePipeline, JEVEX_META=True)

    costs = [
        (await pipeline.process_item({"document": document(f"https://books.example/{i}")}))[
            "document_meta"
        ]["jev"]["cost"]
        for i in range(2)
    ]

    stats = pipeline.crawler.stats.get_stats()
    assert stats["jevex/documents"] == 2
    assert stats["jevex/records"] == 2
    assert stats["jevex/jev_requests"] == 2
    assert stats["jevex/llm_calls"] == 0
    assert stats["jevex/jev_cost_usd"] == pytest.approx(sum(costs))
    assert stats["jevex/jev_cost_usd"] > 0
    assert stats["jevex/llm_cost_usd"] == pytest.approx(0.0)
    assert "jevex/stopped" not in stats


async def test_pipeline_counts_documents_stopped_early() -> None:
    class Stopping(FakePipeline):
        def make_extractor(self) -> Extractor:
            extractor = super().make_extractor()

            @dataclass
            class Stop:
                name: str = "gate"

                async def run(self, ctx: Context) -> None:
                    ctx.stop(self.name, "not a book page")

            extractor.pipeline = Pipeline([Stop(), self.stage])
            return extractor

    pipeline = await opened(Stopping)

    item = await pipeline.process_item({"document": document()})

    assert item == {"records": []}
    assert pipeline.stage.seen == []
    assert pipeline.crawler.stats.get_value("jevex/stopped") == 1
    assert pipeline.crawler.stats.get_value("jevex/documents") == 1


async def test_pipeline_adds_meta_when_asked() -> None:
    pipeline = await opened(FakePipeline, JEVEX_META=True, JEVEX_KEEP_DOCUMENT=True)
    doc = document()

    item = await pipeline.process_item({"document": doc})

    assert item["document"] is doc
    (record,) = item["records"]
    assert record["record"] == {"title": "Dune"}
    assert record["meta"]["title"]["confidence"] == pytest.approx(0.9)
    assert record["meta"]["title"]["method"] == "jev"
    assert item["document_meta"]["url"] == "https://books.example/dune"
    assert item["document_meta"]["jev"]["requests"] == 1


async def test_pipeline_uses_the_configured_item_fields() -> None:
    pipeline = await opened(
        FakePipeline,
        JEVEX_DOCUMENT_FIELD="page",
        JEVEX_RECORDS_FIELD="books",
        JEVEX_META_FIELD="extraction",
        JEVEX_META=True,
    )

    item = await pipeline.process_item({"page": document(), "document": "untouched"})

    assert set(item) == {"document", "books", "extraction"}
    assert item["document"] == "untouched"


async def test_pipeline_fills_scrapy_items() -> None:
    class BookItem(scrapy.Item):
        url = scrapy.Field()
        document = scrapy.Field()
        records = scrapy.Field()

    pipeline = await opened(FakePipeline)

    item = await pipeline.process_item(BookItem(url="u", document=document()))

    assert isinstance(item, BookItem)
    assert dict(item) == {
        "url": "u",
        "records": [{"schema": "Book", "entity": "document", "record": {"title": "Dune"}}],
    }


async def test_pipeline_passes_items_without_a_document_through() -> None:
    pipeline = await opened(FakePipeline)
    plain = {"url": "u"}
    empty = {"url": "u", "document": None}

    assert await pipeline.process_item(plain) is plain
    assert await pipeline.process_item(empty) is empty
    assert plain == {"url": "u"}
    assert empty == {"url": "u", "document": None}
    assert pipeline.stage.seen == []
    assert pipeline.crawler.stats.get_value("jevex/documents") is None


async def test_pipeline_rejects_a_document_field_that_isnt_a_document() -> None:
    pipeline = await opened(FakePipeline)
    response = HtmlResponse(url="https://books.example/dune", body=HTML)

    with pytest.raises(TypeError, match=r"holds a HtmlResponse, not a jevex Document"):
        await pipeline.process_item({"document": response})


async def test_a_failed_extraction_fails_the_item() -> None:
    pipeline = await opened(FakePipeline)
    pipeline.stage.fail = RuntimeError("boom")
    item = {"document": document()}

    with pytest.raises(ExtractionError, match="select stage: RuntimeError: boom") as raised:
        await pipeline.process_item(item)

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert "records" not in item
    stats = pipeline.crawler.stats
    assert stats.get_value("jevex/documents") == 1
    assert stats.get_value("jevex/failed") == 1
    assert stats.get_value("jevex/errors/stage") == 1
    assert stats.get_value("jevex/errors/select/stage/-") == 1


async def test_a_partial_result_fills_the_item_and_is_counted() -> None:
    pipeline = await opened(FakePipeline)
    pipeline.stage.skip = RuntimeError("bad regex")

    item = await pipeline.process_item({"document": document()})

    assert item["records"][0]["record"] == {"title": "Dune"}
    stats = pipeline.crawler.stats
    assert stats.get_value("jevex/partial") == 1
    assert stats.get_value("jevex/errors/generator") == 1
    assert stats.get_value("jevex/errors/candidates/generator/gen-1") == 1
    assert stats.get_value("jevex/failed") is None
    assert stats.get_value("jevex/stage_seconds/select") > 0
    assert (
        stats.get_value("jevex/jev_rate_limited"),
        stats.get_value("jevex/llm_rate_limited"),
    ) == (
        0,
        0,
    )


async def test_a_spend_cap_fails_the_item_uncounted() -> None:
    pipeline = await opened(FakePipeline)
    pipeline.stage.fail = JevBudgetExceededError("capped")

    with pytest.raises(JevBudgetExceededError):
        await pipeline.process_item({"document": document()})

    assert pipeline.crawler.stats.get_value("jevex/documents") is None


async def test_fill_item_can_be_overridden_for_typed_items() -> None:
    class Typed(FakePipeline):
        def fill_item(self, item: Any, result: ExtractionResult) -> Any:
            return result.one(Book).strict()

    pipeline = await opened(Typed)

    assert await pipeline.process_item({"document": document()}) == Book(title="Dune")


async def test_pipeline_refuses_to_open_without_asyncio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contrib, "is_asyncio_available", lambda: False)
    pipeline = FakePipeline.from_crawler(crawler())

    with pytest.raises(AsyncioRequiredError, match="AsyncioSelectorReactor"):
        await pipeline.open_spider()

    with pytest.raises(RuntimeError, match="hasn't opened yet"):
        pipeline.extractor  # noqa: B018


async def test_close_spider_waits_for_learning_then_closes_the_extractor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = await opened(FakePipeline)
    calls: list[str] = []

    async def wait() -> None:
        calls.append("wait")

    monkeypatch.setattr(pipeline.extractor, "wait_for_learning", wait)
    await pipeline.close_spider()

    assert calls == ["wait"]
    assert pipeline.closed
    with pytest.raises(RuntimeError, match="hasn't opened yet"):
        pipeline.extractor  # noqa: B018
    await pipeline.close_spider()  # a second close does nothing


async def test_close_spider_can_skip_waiting_for_learning(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = await opened(FakePipeline, JEVEX_WAIT_FOR_LEARNING=False)

    async def wait() -> None:
        raise AssertionError("shouldn't wait")

    monkeypatch.setattr(pipeline.extractor, "wait_for_learning", wait)
    await pipeline.close_spider()

    assert pipeline.closed


async def test_close_spider_closes_the_extractor_when_learning_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = await opened(FakePipeline)

    async def wait() -> None:
        raise RuntimeError("learner crashed")

    monkeypatch.setattr(pipeline.extractor, "wait_for_learning", wait)
    with pytest.raises(RuntimeError, match="learner crashed"):
        await pipeline.close_spider()

    assert pipeline.closed


EXAMPLE = VerifiedExample(
    id="ex-1",
    field="Book.title",
    statement="Title: Dune",
    value="Dune",
    evidence=(7, 11),
    context={"heading_trail": [], "kind": "sentence"},
    source="llm",
    probability=0.99,
)


class Learning(FakePipeline):
    """A pipeline whose stage hands the learner an example."""

    def make_extractor(self) -> Extractor:
        extractor = super().make_extractor()
        extractor.generator_llm = FakeLLM(lambda _p, _s: {})
        stage = self.stage
        run = stage.run

        async def run_and_learn(ctx: Context) -> None:
            await run(ctx)
            assert ctx.learner is not None
            await ctx.learner.submit(EXAMPLE)

        stage.run = run_and_learn
        return extractor


async def test_close_spider_counts_the_learners_outcomes() -> None:
    pipeline = await opened(Learning)
    await pipeline.process_item({"document": document()})
    await pipeline.close_spider()
    stats = pipeline.crawler.stats
    assert stats.get_value("jevex/learner_alive") == 1
    assert stats.get_value("jevex/learner_deaths") == 0
    learned = {k: v for k, v in stats.get_stats().items() if k.startswith("jevex/learner/")}
    assert sum(learned.values()) == 1
    assert learned["jevex/learner/accepted"] == 0


async def test_close_spider_reports_a_dead_learner(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = await opened(Learning)
    learner = await pipeline.extractor.learner()
    assert learner is not None

    async def bug(_example: VerifiedExample) -> None:
        raise RuntimeError("bug")

    monkeypatch.setattr(learner, "learn", bug)
    await pipeline.process_item({"document": document()})
    with pytest.raises(RuntimeError, match="learner worker failed"):
        await pipeline.close_spider()
    assert pipeline.crawler.stats.get_value("jevex/learner_alive") == 0
    assert pipeline.crawler.stats.get_value("jevex/learner_deaths") == 1


async def test_close_spider_before_open_does_nothing() -> None:
    await JevexPipeline.from_crawler(crawler()).close_spider()


# --- the default extractor from settings ---------------------------------------------


async def test_default_extractor_comes_from_the_settings(tmp_path: Path) -> None:
    db = tmp_path / "jevex.db"
    pipeline = await opened(
        JevexPipeline,
        JEVEX_SCHEMAS=[Book, f"{__name__}:Author"],
        JEVEX_STORE=f"sqlite:///{db}",
        JEVEX_THRESHOLD=0.5,
    )
    extractor = pipeline.extractor

    assert [s.name for s in extractor.schemas] == ["Book", "Author"]
    assert extractor.threshold == 0.5
    assert isinstance(await extractor.store(), SQLiteStore)
    await pipeline.close_spider()
    assert db.exists()


async def test_default_extractor_takes_schemas_from_a_comma_separated_setting() -> None:
    pipeline = await opened(JevexPipeline, JEVEX_SCHEMAS=f"{__name__}:Book, {__name__}:Author")
    extractor = pipeline.extractor

    assert [s.name for s in extractor.schemas] == ["Book", "Author"]
    assert extractor.threshold == 0.0
    assert await extractor.store() is None
    await pipeline.close_spider()


async def test_default_extractor_takes_a_locale() -> None:
    pipeline = await opened(JevexPipeline, JEVEX_SCHEMAS=[Book], JEVEX_LOCALE=" de_ch ")
    assert pipeline.extractor.locale == "de-CH"
    await pipeline.close_spider()
    pipeline = await opened(JevexPipeline, JEVEX_SCHEMAS=[Book], JEVEX_LOCALE="")
    assert pipeline.extractor.locale is None
    await pipeline.close_spider()


@pytest.mark.parametrize("locale", ["Swiss German", 42])
async def test_default_extractor_rejects_a_bad_locale(locale: object) -> None:
    pipeline = JevexPipeline.from_crawler(crawler(JEVEX_SCHEMAS=[Book], JEVEX_LOCALE=locale))

    with pytest.raises(ValueError, match="JEVEX_LOCALE: locale must be a BCP 47 language tag"):
        await pipeline.open_spider()


async def test_default_extractor_needs_schemas() -> None:
    pipeline = JevexPipeline.from_crawler(crawler())

    with pytest.raises(ValueError, match="set JEVEX_SCHEMAS"):
        await pipeline.open_spider()


@pytest.mark.parametrize(
    ("schemas", "error", "message"),
    [
        ([f"{__name__}:Nope"], ValueError, r"JEVEX_SCHEMAS: .* has no attribute 'Nope'"),
        (["no_colon"], ValueError, r"JEVEX_SCHEMAS: .*module:Class"),
        ([f"{__name__}:HTML"], ValueError, r"JEVEX_SCHEMAS: .* is not a Pydantic model"),
        ([42], TypeError, r"JEVEX_SCHEMAS takes model classes or module:Class paths, not 42"),
        ([str], TypeError, r"not <class 'str'>"),
    ],
)
async def test_default_extractor_rejects_bad_schemas(
    schemas: list[Any], error: type[Exception], message: str
) -> None:
    pipeline = JevexPipeline.from_crawler(crawler(JEVEX_SCHEMAS=schemas))

    with pytest.raises(error, match=message):
        await pipeline.open_spider()


# --- a real crawl on the asyncio reactor ---------------------------------------------

CRAWL = """
    import sys

    import scrapy
    from scrapy.crawler import CrawlerProcess

    from jevex import Extractor
    from jevex.contrib.scrapy import JevexPipeline, document_from_response
    from jevex.jev import JevClient
    from jevex.pipeline import Pipeline
    from jevex.testing import FakeJev
    from test_contrib_scrapy import Book, FindTitle

    REACTOR = sys.argv[1]
    PAGES = sys.argv[2:-1]
    OUT = sys.argv[-1]


    class Fake(JevexPipeline):
        def make_extractor(self):
            return Extractor(
                [Book],
                jev=JevClient(FakeJev().noul("Is this a book page?", p=0.9)),
                pipeline=Pipeline([FindTitle()]),
            )


    class Pages(scrapy.Spider):
        name = "pages"
        start_urls = PAGES

        def parse(self, response):
            yield {"url": response.url, "document": document_from_response(response)}


    process = CrawlerProcess(
        {
            "ITEM_PIPELINES": {Fake: 300},
            "FEEDS": {OUT: {"format": "jsonlines"}},
            "TWISTED_REACTOR": REACTOR,
            "LOG_LEVEL": "WARNING",
            "TELNETCONSOLE_ENABLED": False,
        }
    )
    process.crawl(Pages)
    process.start()

    from twisted.internet import reactor

    print(type(reactor).__name__)
"""


ASYNCIO_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
SELECT_REACTOR = "twisted.internet.selectreactor.SelectReactor"


def crawl(tmp_path: Path, reactor: str) -> tuple[subprocess.CompletedProcess[str], list[str], Path]:
    """Crawl two saved pages in a fresh process (a reactor can't be restarted), with a
    fake Jev, writing the items to a JSON lines feed."""
    pages: list[str] = []
    for name in ("dune", "emma"):
        path = tmp_path / f"{name}.html"
        path.write_bytes(HTML)
        pages.append(path.as_uri())
    script = tmp_path / "crawl.py"
    script.write_text(textwrap.dedent(CRAWL))
    out = tmp_path / "out.jsonl"
    done = subprocess.run(
        [sys.executable, str(script), reactor, *pages, str(out)],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parent)},
        check=False,
    )
    return done, pages, out


def test_a_crawl_on_the_asyncio_reactor_writes_records_to_the_feed(tmp_path: Path) -> None:
    done, pages, out = crawl(tmp_path, ASYNCIO_REACTOR)

    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "AsyncioSelectorReactor"
    items = [json.loads(line) for line in out.read_text().splitlines()]
    assert sorted(item["url"] for item in items) == sorted(pages)
    for item in items:
        assert set(item) == {"url", "records"}
        assert item["records"] == [
            {"schema": "Book", "entity": "document", "record": {"title": "Dune"}}
        ]


def test_a_crawl_on_another_reactor_fails_to_open(tmp_path: Path) -> None:
    done, _, out = crawl(tmp_path, SELECT_REACTOR)

    assert done.stdout.strip() == "SelectReactor"
    assert "AsyncioRequiredError: JevexPipeline needs Scrapy's asyncio reactor" in done.stderr
    assert not out.exists() or not out.read_text()
