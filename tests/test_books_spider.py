"""The example spider, over saved books.toscrape.com pages (no network)."""

from pathlib import Path
from typing import Any

import pytest
from scrapy.http import HtmlResponse
from scrapy.utils.misc import load_object
from scrapy.utils.test import get_crawler
from test_smoke_books import FIXTURES, PAGES, pattern_jev

from jevex import Document
from jevex.clean import CleanStage
from jevex.examples.books import StarRatingCleaner
from jevex.examples.books_spider import BooksPipeline, BooksSpider
from jevex.jev import JevClient

CATALOGUE = b"""<html><body><ol class="row">
<li><article class="product_pod"><h3>
  <a href="catalogue/a-light-in-the-attic_1000/index.html" title="A Light...">A Light...</a>
</h3></article></li>
<li><article class="product_pod"><h3>
  <a href="catalogue/tipping-the-velvet_999/index.html" title="Tipping the Velvet">Tipping...</a>
</h3></article></li>
</ol>
<ul class="pager"><li class="current">Page 1 of 50</li>
<li class="next"><a href="catalogue/page-2.html">next</a></li></ul>
</body></html>"""


def catalogue(body: bytes = CATALOGUE) -> HtmlResponse:
    return HtmlResponse(url="https://books.toscrape.com/", body=body, encoding="utf-8")


def book_page(name: str) -> HtmlResponse:
    return HtmlResponse(
        url=f"https://books.toscrape.com/catalogue/{name}/index.html",
        body=(FIXTURES / f"{name}.html").read_bytes(),
        headers={"Content-Type": "text/html"},
    )


def test_parse_follows_each_book_then_the_next_page() -> None:
    spider = BooksSpider()

    requests = list(spider.parse(catalogue()))

    assert [(r.url, r.callback) for r in requests] == [
        (
            "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html",
            spider.parse_book,
        ),
        (
            "https://books.toscrape.com/catalogue/tipping-the-velvet_999/index.html",
            spider.parse_book,
        ),
        ("https://books.toscrape.com/catalogue/page-2.html", spider.parse),
    ]


def test_parse_stops_at_the_last_page() -> None:
    last = CATALOGUE.replace(b'<li class="next"><a href="catalogue/page-2.html">next</a></li>', b"")

    spider = BooksSpider()

    requests = list(spider.parse(catalogue(last)))

    assert [r.callback for r in requests] == [spider.parse_book, spider.parse_book]


def test_parse_book_yields_the_page_as_a_document() -> None:
    response = book_page("tipping-the-velvet_999")

    (item,) = BooksSpider().parse_book(response)

    assert set(item) == {"url", "document"}
    assert item["url"] == response.url
    doc = item["document"]
    assert isinstance(doc, Document)
    assert doc.content == response.body
    assert doc.content_type == "text/html"
    assert doc.url == response.url


def test_the_spider_crawls_politely_through_books_pipeline() -> None:
    settings: dict[str, Any] = BooksSpider.custom_settings or {}

    (path,) = settings["ITEM_PIPELINES"]
    assert load_object(path) is BooksPipeline
    assert settings["ROBOTSTXT_OBEY"] is True
    assert settings["DOWNLOAD_DELAY"] >= 1
    assert BooksSpider.allowed_domains == ["books.toscrape.com"]
    assert BooksSpider.start_urls == ["https://books.toscrape.com/"]


async def test_books_pipeline_uses_the_rating_cleaner_and_the_store_setting(
    tmp_path: Path,
) -> None:
    db = tmp_path / "jevex.db"
    crawler = get_crawler(BooksSpider, {"JEVEX_STORE": f"sqlite:///{db}", "JEVEX_THRESHOLD": 0.3})
    pipeline = BooksPipeline.from_crawler(crawler)
    await pipeline.open_spider()
    extractor = pipeline.extractor

    clean = next(s for s in extractor.pipeline if s.name == "clean")
    assert isinstance(clean, CleanStage)
    assert isinstance(clean.cleaner, StarRatingCleaner)
    assert [s.name for s in extractor.schemas] == ["Book"]
    assert extractor.threshold == 0.3
    assert await extractor.store() is not None
    await pipeline.close_spider()
    assert db.exists()


@pytest.mark.parametrize("name", list(PAGES))
async def test_a_book_page_becomes_a_book_record(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = pattern_jev(name)
    monkeypatch.setattr(JevClient, "from_env", lambda: fake.client())
    pipeline = BooksPipeline.from_crawler(get_crawler(BooksSpider))
    await pipeline.open_spider()

    (item,) = BooksSpider().parse_book(book_page(name))
    item = await pipeline.process_item(item)
    await pipeline.close_spider()

    (record,) = item["records"]
    assert record["schema"] == "Book"
    assert record["record"] == PAGES[name].model_dump(mode="json")
    assert pipeline.crawler.stats.get_value("jevex/documents") == 1
