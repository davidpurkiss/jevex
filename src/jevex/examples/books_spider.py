"""An example Scrapy spider: :class:`~jevex.examples.books.Book` records from
https://books.toscrape.com, a site made for scraping practice (``scrapy`` extra).

Run it with Scrapy (``TYPESAFE_API_KEY`` set; each book page costs Jev calls)::

    scrapy runspider src/jevex/examples/books_spider.py -O books.jsonl \\
        -s CLOSESPIDER_ITEMCOUNT=5 -s JEVEX_STORE=sqlite:///jevex.db

The spider only crawls: it follows the catalogue's book and next-page links and yields
each book page as a document. :class:`BooksPipeline` extracts the records, with the
example's rating cleaner, so each line of ``books.jsonl`` holds a page's URL and records.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import scrapy

from jevex.contrib.scrapy import JevexPipeline, document_from_response
from jevex.examples.books import Book, books_pipeline
from jevex.extractor import Extractor

if TYPE_CHECKING:
    from collections.abc import Iterator

    from scrapy.http import Response


class BooksPipeline(JevexPipeline):
    """:class:`JevexPipeline` with :func:`~jevex.examples.books.books_pipeline`, which
    reads the star ratings books.toscrape.com only gives as CSS classes."""

    def make_extractor(self) -> Extractor:
        return Extractor(
            [Book],
            pipeline=books_pipeline(),
            store=self.settings.get("JEVEX_STORE") or None,
            threshold=self.settings.getfloat("JEVEX_THRESHOLD", 0.0),
        )


class BooksSpider(scrapy.Spider):
    name = "books"
    allowed_domains: ClassVar[list[str]] = ["books.toscrape.com"]
    # Scrapy declares these as instance attributes, so they can't be ClassVars here.
    start_urls: list[str] = ["https://books.toscrape.com/"]  # noqa: RUF012
    custom_settings: dict[str, Any] | None = {  # noqa: RUF012
        "ITEM_PIPELINES": {f"{__name__}.BooksPipeline": 300},
        "ROBOTSTXT_OBEY": True,
        "DOWNLOAD_DELAY": 1.0,
    }

    def parse(self, response: Response, **kwargs: Any) -> Iterator[scrapy.Request]:
        """A catalogue page: its books, then the next page."""
        for href in response.css("article.product_pod h3 a::attr(href)").getall():
            yield response.follow(href, callback=self.parse_book)
        next_page = response.css("li.next a::attr(href)").get()
        if next_page is not None:
            yield response.follow(next_page, callback=self.parse)

    def parse_book(self, response: Response) -> Iterator[dict[str, Any]]:
        yield {"url": response.url, "document": document_from_response(response)}
